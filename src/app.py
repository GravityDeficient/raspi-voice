import argparse
import json
import os
import queue
import sys
import time
import wave
from datetime import datetime

import numpy as np
import sounddevice as sd
import webrtcvad
from rich.console import Console
from rich.panel import Panel
from vosk import KaldiRecognizer, Model


console = Console()


SAMPLE_RATE_DEFAULT = 16000  # Preferred
CHANNELS = 1
FRAME_DURATION_MS = 30  # valid for VAD: 10, 20, 30 ms
WAKE_WORDS = ["hey notes", "okay notes", "hey memo"]


def record_audio_to_queue(audio_q: queue.Queue, stop_flag: list[int], device=None, samplerate: int = SAMPLE_RATE_DEFAULT, blocksize: int | None = None):
    overflow_count = 0
    def callback(indata, frames, time_info, status):
        nonlocal overflow_count
        if status:
            try:
                if getattr(status, "input_overflow", False):
                    overflow_count += 1
                    if overflow_count % 25 == 0:
                        console.log("Audio callback: input overflow (throttled)")
                # Avoid logging all status messages every callback to reduce overhead
            except Exception:
                pass
        try:
            audio_q.put_nowait(indata.copy())
        except queue.Full:
            # Drop frame if consumer is slow; better than blocking audio callback
            pass
        if stop_flag and stop_flag[0]:
            raise sd.CallbackStop()

    with sd.InputStream(
        channels=CHANNELS,
        samplerate=samplerate,
        dtype="int16",
        blocksize=blocksize,
        callback=callback,
        device=device,  # default input device or user-provided
        latency="high",
    ):
        while True:
            time.sleep(0.1)


def frames_from_queue(audio_q: queue.Queue):
    while True:
        data = audio_q.get()
        if data is None:
            return
        yield data


def bytes_from_frames(frames):
    for frame in frames:
        yield frame.tobytes()


def detect_wake_word(model: Model, audio_q: queue.Queue, wake_words: list[str], sample_rate: int) -> bool:
    # Use grammar-based recognition constrained to wake words for faster, low-resource detection
    wake_words_lc = [w.strip().lower() for w in wake_words if w.strip()]
    grammar = json.dumps(wake_words_lc)
    rec = KaldiRecognizer(model, sample_rate, grammar)
    rec.SetWords(False)

    console.print(Panel("Listening for wake word: " + ", ".join(f'"{w}"' for w in wake_words), title="Wake Word"))

    for b in bytes_from_frames(frames_from_queue(audio_q)):
        if rec.AcceptWaveform(b):
            res = json.loads(rec.Result())
            text = (res.get("text") or "").strip()
            if text and any(text == w for w in wake_words_lc):
                console.print("Wake word detected: [bold green]" + text + "[/]")
                return True
        else:
            # partial results are ignored for grammar
            pass
    return False


def choose_input_device(preferred: str | None) -> int | None:
    """Return a valid input device index or None if no device available.

    Selection order:
    1) If preferred is an int index and valid for input, use it.
    2) If preferred is a substring, choose first device whose name contains it and has input channels.
    3) Use system default input device if it has input channels.
    4) Fallback to the first device with input channels.
    """
    try:
        devices = sd.query_devices()
    except Exception as e:
        console.print(f"[red]Failed to query audio devices:[/] {e}")
        return None

    # 1) numeric index
    if preferred is not None:
        try:
            idx = int(preferred)
            if 0 <= idx < len(devices) and devices[idx]["max_input_channels"] > 0:
                return idx
        except ValueError:
            pass

    # 2) substring match
    if preferred:
        low = preferred.lower()
        for idx, dev in enumerate(devices):
            try:
                if low in dev["name"].lower() and dev["max_input_channels"] > 0:
                    return idx
            except Exception:
                continue

    # 3) default input device
    try:
        default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
        if isinstance(default_in, int) and 0 <= default_in < len(devices) and devices[default_in]["max_input_channels"] > 0:
            return default_in
    except Exception:
        pass

    # 4) first with input
    for idx, dev in enumerate(devices):
        try:
            if dev["max_input_channels"] > 0:
                return idx
        except Exception:
            continue

    return None


def pick_working_samplerate(device_idx: int, preferred: int = SAMPLE_RATE_DEFAULT) -> tuple[int, int] | None:
    """Pick a sample rate supported by the device and return (rate, frame_size)."""
    # Candidates prioritized: preferred (16k), then common USB rates
    # WebRTC VAD supports 8000, 16000, 32000, 48000
    # Prefer lower rates first to reduce CPU and overflow risk
    candidates = [preferred, 32000, 48000, 8000]
    # De-duplicate while preserving order
    seen = set()
    ordered = []
    for r in candidates:
        if r not in seen:
            ordered.append(r)
            seen.add(r)
    for rate in ordered:
        try:
            sd.check_input_settings(device=device_idx, channels=CHANNELS, dtype="int16", samplerate=rate)
            frame_size = int(rate * FRAME_DURATION_MS / 1000)
            return rate, frame_size
        except Exception:
            continue
    return None


def collect_speech_until_silence(model: Model, audio_q: queue.Queue, aggressiveness: int = 2, max_silence_ms: int = 800, sample_rate: int = SAMPLE_RATE_DEFAULT):
    vad = webrtcvad.Vad(aggressiveness)
    rec = KaldiRecognizer(model, sample_rate)
    rec.SetWords(True)

    console.print(Panel("Speak your note. Pause to end.", title="Recording"))

    silence_limit_frames = max(1, int(max_silence_ms / FRAME_DURATION_MS))
    silence_counter = 0

    transcript_chunks = []
    wav_bytes = bytearray()

    for frame in frames_from_queue(audio_q):
        frame_bytes = frame.tobytes()
        wav_bytes.extend(frame_bytes)

        # VAD expects 16-bit mono 8/16/32/48kHz, frame length 10/20/30ms
        is_speech = vad.is_speech(frame_bytes, sample_rate)
        if is_speech:
            silence_counter = 0
        else:
            silence_counter += 1

        if rec.AcceptWaveform(frame_bytes):
            res = json.loads(rec.Result())
            if res.get("text"):
                transcript_chunks.append(res["text"])  # accumulate finalized chunks
        else:
            # Optional: could use partials to show live text
            pass

        if silence_counter >= silence_limit_frames:
            # finalize recognition
            final_res = json.loads(rec.FinalResult())
            if final_res.get("text"):
                transcript_chunks.append(final_res["text"])
            break

    transcript = " ".join(transcript_chunks).strip()
    return transcript, bytes(wav_bytes)


def save_markdown_note(text: str, notes_dir: str, vault_subdir: str | None = None) -> str:
    os.makedirs(notes_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"note_{timestamp}.md"
    path = os.path.join(notes_dir, filename)
    title = datetime.now().strftime("Voice Note %Y-%m-%d %H:%M:%S")
    content = f"---\nsource: raspi-voice\ncreated: {datetime.now().isoformat()}\n---\n\n# {title}\n\n{text}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def save_wav(data_bytes: bytes, audio_dir: str, sample_rate: int) -> str:
    os.makedirs(audio_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wav_path = os.path.join(audio_dir, f"note_{timestamp}.wav")
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(data_bytes)
    return wav_path


def ensure_model(model_dir: str, small: bool = True) -> str:
    # Download Vosk English small model if not present
    import tarfile
    import shutil
    import requests

    os.makedirs(model_dir, exist_ok=True)
    target = os.path.join(model_dir, "vosk-model-small-en-us") if small else os.path.join(model_dir, "vosk-model-en-us")
    if os.path.isdir(target) and os.path.isdir(os.path.join(target, "am")):
        return target

    console.print("Downloading Vosk model (first run only)... This may take a few minutes.")
    url = (
        "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    )
    # Prefer .zip to avoid tar on Windows; but we are targeting Pi, still okay to use here for cross-dev
    zip_path = os.path.join(model_dir, "vosk-model-small-en-us.zip")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        console.print(f"Downloading: {pct}%", end="\r")

    # Extract zip
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(model_dir)

    # Find extracted folder that starts with name
    extracted = None
    for name in os.listdir(model_dir):
        if name.startswith("vosk-model-small-en-us") and os.path.isdir(os.path.join(model_dir, name)):
            extracted = os.path.join(model_dir, name)
            break

    if not extracted:
        raise RuntimeError("Failed to extract Vosk model")

    # Normalize to target path
    if os.path.abspath(extracted) != os.path.abspath(target):
        if os.path.exists(target):
            shutil.rmtree(target)
        shutil.move(extracted, target)

    try:
        os.remove(zip_path)
    except OSError:
        pass

    return target


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Voice Assistant POC (wake word + STT to note)")
    parser.add_argument("--models", default=os.path.join(os.path.dirname(__file__), "..", "..", "models"), help="Models directory")
    parser.add_argument("--notes", default=os.path.join(os.path.dirname(__file__), "..", "..", "notes"), help="Notes output directory")
    parser.add_argument("--audio", default=os.path.join(os.path.dirname(__file__), "..", "..", "notes", "audio"), help="Audio output directory")
    parser.add_argument("--vad", type=int, default=2, help="WebRTC VAD aggressiveness 0-3")
    parser.add_argument("--silence", type=int, default=900, help="Silence ms to stop recording")
    parser.add_argument("--once", action="store_true", help="Exit after a single note")
    parser.add_argument("--device", default=None, help="Input device index or substring to select USB mic")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--wake-words", default=None, help='Comma-separated wake words, e.g., "hey notes, okay notes"')
    args = parser.parse_args()

    models_dir = os.path.abspath(args.models)
    notes_dir = os.path.abspath(args.notes)
    audio_dir = os.path.abspath(args.audio)

    model_path = ensure_model(models_dir, small=True)
    model = Model(model_path)

    audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=50)
    stop_flag = [0]

    if args.list_devices:
        console.print(Panel("Available audio devices", title="Audio Devices"))
        try:
            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                console.print(f"{idx}: {dev['name']} (in: {dev['max_input_channels']}, out: {dev['max_output_channels']})")
        except Exception as e:
            console.print(f"[red]Failed to query devices:[/] {e}")
        return

    selected_device = choose_input_device(args.device)
    if selected_device is None:
        console.print("[red]No valid input device found. Check microphone and ALSA/Pulse setup.[/]")
        # Show devices to aid debugging
        try:
            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                console.print(f"{idx}: {dev['name']} (in: {dev['max_input_channels']}, out: {dev['max_output_channels']})")
        except Exception:
            pass
        sys.exit(1)
    else:
        try:
            dev = sd.query_devices(selected_device)
            console.print(Panel(f"Using input device #{selected_device}: {dev['name']}", title="Audio Device"))
        except Exception:
            console.print(Panel(f"Using input device #{selected_device}", title="Audio Device"))

    # Determine a working sample rate and frame size for this device
    picked = pick_working_samplerate(selected_device, SAMPLE_RATE_DEFAULT)
    if not picked:
        console.print("[red]Could not find a supported sample rate for the selected device.[/]")
        sys.exit(1)
    sample_rate, frame_size = picked
    console.print(Panel(f"Sample rate: {sample_rate} Hz, Frame: {FRAME_DURATION_MS} ms ({frame_size} samples)", title="Audio Config"))

    console.print(Panel("Starting microphone stream", title="Audio"))
    try:
        with sd.InputStream(channels=CHANNELS, samplerate=sample_rate, dtype="int16", blocksize=frame_size, device=selected_device, latency="high"):
            pass
    except Exception as e:
        console.print(f"[red]Audio device error:[/] {e}")
        sys.exit(1)

    # Start background recording thread
    import threading

    t = threading.Thread(target=record_audio_to_queue, args=(audio_q, stop_flag, selected_device, sample_rate, frame_size), daemon=True)
    t.start()

    # Derive wake words list
    if args.wake_words:
        wake_words = [w.strip() for w in args.wake_words.split(",") if w.strip()]
        if not wake_words:
            wake_words = WAKE_WORDS
    else:
        wake_words = WAKE_WORDS

    ww_display = ", ".join(f'"{w}"' for w in wake_words)
    console.print(Panel(f"Ready. Say one of: {ww_display}", title="Status", subtitle="Ctrl+C to stop"))
    try:
        while True:
            if not detect_wake_word(model, audio_q, wake_words, sample_rate):
                continue
            transcript, wav_bytes = collect_speech_until_silence(model, audio_q, aggressiveness=args.vad, max_silence_ms=args.silence, sample_rate=sample_rate)
            if not transcript:
                console.print("[yellow]No speech captured.[/]")
                continue
            md_path = save_markdown_note(transcript, notes_dir)
            wav_path = save_wav(wav_bytes, audio_dir, sample_rate)
            console.print(Panel(f"Saved note to:\n{md_path}\nAudio:\n{wav_path}", title="Saved", border_style="green"))
            if args.once:
                break
    except KeyboardInterrupt:
        console.print("Exiting...")
    finally:
        stop_flag[0] = 1


if __name__ == "__main__":
    main()
