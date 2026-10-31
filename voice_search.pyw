import os
import sys

# ---------------- 1. CUDA DLL PATH RESOLUTION ----------------
def setup_cuda_paths():
    if sys.platform != "win32":
        return

    added_dirs = set()

    def register_dir(dll_dir):
        dll_dir = os.path.abspath(dll_dir)
        if os.path.isdir(dll_dir) and dll_dir not in added_dirs:
            os.add_dll_directory(dll_dir)
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ["PATH"]
            added_dirs.add(dll_dir)
            print(f"[CUDA] Added DLL path: {dll_dir}")

    for mod_name in ["nvidia.cublas.lib", "nvidia.cudnn.lib"]:
        try:
            mod = __import__(mod_name, fromlist=["__file__"])
            if getattr(mod, "__file__", None):
                bin_dir = os.path.join(os.path.dirname(mod.__file__), "..", "bin")
                register_dir(bin_dir)
        except ImportError:
            pass

    import site
    search_roots = site.getsitepackages() + [
        site.getusersitepackages(),
        os.path.join(sys.prefix, "Lib", "site-packages")
    ]
    for sp in search_roots:
        for sub in [("nvidia", "cublas", "bin"), ("nvidia", "cudnn", "bin")]:
            register_dir(os.path.join(sp, *sub))

    cuda_system_path = os.environ.get("CUDA_PATH")
    if cuda_system_path:
        register_dir(os.path.join(cuda_system_path, "bin"))

setup_cuda_paths()

# ---------------- 2. IMPORTS & CONFIGURATION ----------------
import math
import time
import signal
import threading
import urllib.parse
import webbrowser
import ctypes
import tkinter as tk
import numpy as np
import sounddevice as sd
import keyboard
from faster_whisper import WhisperModel

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

HOTKEY_SEARCH = "ctrl+alt+p"
HOTKEY_TYPE   = "ctrl+alt+o"
SAMPLE_RATE = 16000
MIN_DURATION_SEC = 0.4   # Ignore accidental micro-recordings under 400ms
MODEL_SIZE = "base.en"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
BORDER_THICKNESS = 6


# ---------------- 3. CLICK-THROUGH PULSING OVERLAY ----------------
class ScreenBorderOverlay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        
        self.width = self.root.winfo_screenwidth()
        self.height = self.root.winfo_screenheight()
        self.root.geometry(f"{self.width}x{self.height}+0+0")
        
        self.root.wm_attributes("-topmost", True)
        self.root.wm_attributes("-transparentcolor", "black")
        self.root.config(bg="black")

        self.canvas = tk.Canvas(
            self.root, width=self.width, height=self.height,
            bg="black", highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        half = BORDER_THICKNESS // 2
        self.rect = self.canvas.create_rectangle(
            half, half,
            self.width - half, self.height - half,
            outline="#0033aa", width=BORDER_THICKNESS
        )

        self.pulsing = False
        self.phase = 0.0
        self.mode = "search"
        
        self.root.update_idletasks()
        self._make_clickthrough()
        self.root.withdraw()

    def _make_clickthrough(self):
        hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
            
        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x00080000
        WS_EX_TRANSPARENT = 0x00000020
        WS_EX_TOOLWINDOW = 0x00000080
        WS_EX_NOACTIVATE = 0x08000000

        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        new_style = style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)

    def show(self, mode="search"):
        self.root.after(0, self._start_pulse, mode)

    def hide(self):
        self.root.after(0, self._stop_pulse)

    def _start_pulse(self, mode):
        self.mode = mode
        if not self.pulsing:
            self.pulsing = True
            self.phase = 0.0
            self.root.deiconify()
            self.root.attributes("-topmost", True)
            self._animate()

    def _stop_pulse(self):
        self.pulsing = False
        self.root.withdraw()

    def _animate(self):
        if not self.pulsing:
            return
        s = (math.sin(self.phase) + 1.0) / 2.0
        if self.mode == "type":
            # Emerald green pulse for direct keyboard dictation (ctrl+alt+o)
            r = int(0 + s * 20)
            g = int(70 + s * 165)
            b = int(30 + s * 60)
        else:
            # Royal blue pulse for Google search (ctrl+alt+p)
            r = int(0 + s * 15)
            g = int(20 + s * 80)
            b = int(90 + s * 165)
            
        color = f"#{r:02x}{g:02x}{b:02x}"
        self.canvas.itemconfig(self.rect, outline=color)
        self.phase += 0.15
        self.root.after(30, self._animate)

    def _signal_heartbeat(self):
        # Wakes Python's main thread every 200ms so Ctrl+C (SIGINT) works in PowerShell
        self.root.after(200, self._signal_heartbeat)

    def run(self):
        self._signal_heartbeat()
        self.root.mainloop()


# ---------------- 4. VOICE CAPTURE & GPU TRANSCRIPTION ----------------
print(f"Loading Whisper '{MODEL_SIZE}' on {DEVICE} ({COMPUTE_TYPE})...")
try:
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    list(model.transcribe(np.zeros(1600, dtype=np.float32), language="en")[0])
    print("Model and CUDA libraries verified and ready.")
except Exception as e:
    print(f"[WARNING] CUDA initialization failed ({e}). Falling back to CPU (int8)...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print("Model ready on CPU.")

overlay = ScreenBorderOverlay()
is_recording = False
current_mode = None      # 'search' or 'type'
audio_frames = []
hotkey_latched = False   # Prevents key-hold repeat spam

def audio_callback(indata, frames, time_info, status):
    if is_recording:
        audio_frames.append(indata.copy())

def launch_browser_to_foreground(url):
    try:
        ctypes.windll.user32.keybd_event(0x10, 0, 2, 0)
        ctypes.windll.shell32.ShellExecuteW(None, "open", url, None, None, 1)
    except Exception:
        webbrowser.open(url, new=2, autoraise=True)

def type_text_to_active_window(text):
    """Waits for modifier keys to be released, then types directly into the focused input."""
    deadline = time.time() + 1.5
    while time.time() < deadline:
        if not (keyboard.is_pressed("ctrl") or keyboard.is_pressed("alt") or keyboard.is_pressed("o")):
            break
        time.sleep(0.02)
    time.sleep(0.03)
    keyboard.write(text)

def start_recording(mode):
    global is_recording, current_mode, audio_frames
    if not is_recording:
        audio_frames = []
        current_mode = mode
        is_recording = True
        overlay.show(mode=mode)
        label = "SEARCH" if mode == "search" else "TYPE"
        print(f"\n[🎙️ Recording ({label})... Speak now]")

def stop_and_process():
    global is_recording, audio_frames, current_mode
    if not is_recording:
        return
    is_recording = False
    mode_to_run = current_mode
    overlay.hide()
    print("[⏹️ Recording stopped. Processing in memory...]")

    if not audio_frames:
        print("No audio captured.")
        return

    recording = np.concatenate(audio_frames, axis=0).flatten().astype(np.float32)
    audio_frames = []

    # Guard 1: Ignore accidental short taps (< 0.4s)
    duration = len(recording) / SAMPLE_RATE
    if duration < MIN_DURATION_SEC:
        print(f"[Ignored] Recording too short ({duration:.2f}s).")
        return

    def transcribe_worker(audio_data, mode):
        # vad_filter=True strips silence so Whisper doesn't hallucinate "..."
        segments, _ = model.transcribe(audio_data, beam_size=5, language="en", vad_filter=True)
        query_text = " ".join([seg.text for seg in segments]).strip()

        # Guard 2: Strip pure dot/whitespace hallucinations
        cleaned = query_text.replace(".", "").replace("…", "").strip()
        if not cleaned:
            print("No speech detected (silence/noise filtered).")
            return

        if mode == "search":
            print(f"[🔍 Searching]: \"{query_text}\"")
            encoded_query = urllib.parse.quote_plus(query_text)
            url = f"https://www.google.com/search?q={encoded_query}"
            launch_browser_to_foreground(url)
        elif mode == "type":
            print(f"[⌨️ Typing]: \"{query_text}\"")
            type_text_to_active_window(query_text)

    threading.Thread(target=transcribe_worker, args=(recording, mode_to_run), daemon=True).start()

def handle_hotkey(mode):
    global hotkey_latched
    if hotkey_latched:
        return
    hotkey_latched = True

    if not is_recording:
        start_recording(mode)
    else:
        stop_and_process()

def monitor_key_release():
    """Resets the latch only after the physical trigger keys ('p' and 'o') are released."""
    global hotkey_latched
    while True:
        if hotkey_latched:
            if not keyboard.is_pressed("p") and not keyboard.is_pressed("o"):
                hotkey_latched = False
        time.sleep(0.02)

def main():
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=audio_callback)
    stream.start()
    
    threading.Thread(target=monitor_key_release, daemon=True).start()

    def handle_sigint(sig, frame):
        print("\n[Exiting] Shutting down audio stream and overlay...")
        stream.stop()
        overlay.root.quit()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    print(f"\nHotkeys registered:")
    print(f"  • '{HOTKEY_SEARCH}' : Record & Google Search (Blue Border)")
    print(f"  • '{HOTKEY_TYPE}' : Record & Type into active cursor (Green Border)")
    print("Press Ctrl+C in this terminal to exit.")
    
    keyboard.add_hotkey(HOTKEY_SEARCH, lambda: handle_hotkey("search"))
    keyboard.add_hotkey(HOTKEY_TYPE, lambda: handle_hotkey("type"))
    
    try:
        overlay.run()
    except KeyboardInterrupt:
        handle_sigint(None, None)

if __name__ == "__main__":
    main()