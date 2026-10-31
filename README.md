# Local Voice Search & Dictation (Windows)

A lightweight, zero-disk-footprint Windows background utility that captures microphone audio via global hotkeys, transcribes it locally on NVIDIA GPUs using `faster-whisper`, and either runs a Google search or types the text directly into your active window.

## Features

- **100% Local & Private:** Audio is recorded strictly into RAM (`numpy` arrays), transcribed locally via CUDA (`float16`) with automatic CPU (`int8`) fallback, and immediately discarded. Zero audio files ever touch the disk.
- **Two Modes:**
  - **Voice Search (`Ctrl + Alt + P`):** Transcribes speech and opens a Google search in the foreground browser (even over fullscreen DX12/Vulkan games).
  - **Direct Dictation (`Ctrl + Alt + O`):** Transcribes speech and types it directly into the currently focused text input.
- **Game-Safe Visual Overlay:** Renders a click-through, non-activating pulsing screen border while recording: **Royal Blue** for search and **Emerald Green** for dictation.
- **Debounce & Silence Guards:** Latches hotkeys against OS key-repeat spam and filters out accidental micro-presses (<0.4s) and silence hallucinations via Silero VAD.

## Prerequisites

- Windows 10 or 11
- Python 3.10+
- NVIDIA GPU (optional, falls back to CPU automatically)

## Installation

Install the required Python packages and CUDA 12 runtime libraries:

```powershell
pip install faster-whisper sounddevice numpy keyboard nvidia-cublas-cu12 nvidia-cudnn-cu12
```

## Usage

### Run in Terminal (with debug logs)

```powershell
python voice_search.pyw
```

### Run Silently in Background

Double-click `voice_search.pyw` (executes via `pythonw.exe` without a console window).

To run automatically on Windows boot:

1. Press `Win + R`, type `shell:startup`, and press **Enter**.
2. Place a shortcut to `voice_search.pyw` inside the Startup folder.

## Controls

| Hotkey | Action | Visual Indicator |
| :--- | :--- | :--- |
| `Ctrl + Alt + P` | Press once to start recording; press again to stop and search Google | Pulsing Blue Border |
| `Ctrl + Alt + O` | Press once to start recording; press again to stop and type into active cursor | Pulsing Green Border |
| `Ctrl + C` | Exit program (when running in terminal) | None |

## Configuration

Edit the constants at the top of `voice_search.pyw` to customize behavior:

```python
HOTKEY_SEARCH = "ctrl+alt+p"
HOTKEY_TYPE   = "ctrl+alt+o"
MODEL_SIZE    = "base.en"    # Options: tiny.en, base.en, small.en, medium.en, large-v3
DEVICE        = "cuda"       # "cuda" or "cpu"
COMPUTE_TYPE  = "float16"    # "float16" for GPU, "int8" for CPU
```
