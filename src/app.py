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


SAMPLE_RATE = 16000  # 16kHz recommended for Vosk
CHANNELS = 1
FRAME_DURATION_MS = 30  # valid for VAD: 10, 20, 30 ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
WAKE_WORDS = ["hey notes", "okay notes", "hey memo"]


def record_audio_to_queue(audio_q: queue.Queue, stop_flag: list[int], device=None):
    def callback(indata, frames, time_info, status):
        if status:
            console.log(f"Audio callback status: {status}")
        try:
            audio_q.put_nowait(indata.copy())
        except queue.Full:
            # Drop frame if consumer is slow; better than blocking audio callback
            pass
        if stop_flag and stop_flag[0]:
            raise sd.CallbackStop()

    with sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype="int16",
        blocksize=FRAME_SIZE,
        callback=callback,
        device=device,  # default input device or user-provided
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


def detect_wake_word(model: Model, audio_q: queue.Queue, wake_words: list[str]) -> bool:
    # Use grammar-based recognition constrained to wake words for faster, low-resource detection
    wake_words_lc = [w.strip().lower() for w in wake_words if w.strip()]
    grammar = json.dumps(wake_words_lc)
    rec = KaldiRecognizer(model, SAMPLE_RATE, grammar)
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


def collect_speech_until_silence(model: Model, audio_q: queue.Queue, aggressiveness: int = 2, max_silence_ms: int = 800):
    vad = webrtcvad.Vad(aggressiveness)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
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
        is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)
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


def save_wav(data_bytes: bytes, audio_dir: str) -> str:
    os.makedirs(audio_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wav_path = os.path.join(audio_dir, f"note_{timestamp}.wav")
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16
        wf.setframerate(SAMPLE_RATE)
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

    selected_device = None
    if args.device is not None:
        # Allow index or substring match
        try:
            selected_device = int(args.device)
        except ValueError:
            # find first device containing substring and having input channels
            try:
                for idx, dev in enumerate(sd.query_devices()):
                    if args.device.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
                        selected_device = idx
                        break
            except Exception as e:
                console.print(f"[yellow]Device search failed:[/] {e}")

    console.print(Panel("Starting microphone stream", title="Audio"))
    try:
        with sd.InputStream(channels=CHANNELS, samplerate=SAMPLE_RATE, dtype="int16", blocksize=FRAME_SIZE, device=selected_device):
            pass
    except Exception as e:
        console.print(f"[red]Audio device error:[/] {e}")
        sys.exit(1)

    # Start background recording thread
    import threading

    t = threading.Thread(target=record_audio_to_queue, args=(audio_q, stop_flag, selected_device), daemon=True)
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
            if not detect_wake_word(model, audio_q, wake_words):
                continue
            transcript, wav_bytes = collect_speech_until_silence(model, audio_q, aggressiveness=args.vad, max_silence_ms=args.silence)
            if not transcript:
                console.print("[yellow]No speech captured.[/]")
                continue
            md_path = save_markdown_note(transcript, notes_dir)
            wav_path = save_wav(wav_bytes, audio_dir)
            console.print(Panel(f"Saved note to:\n{md_path}\nAudio:\n{wav_path}", title="Saved", border_style="green"))
            if args.once:
                break
    except KeyboardInterrupt:
        console.print("Exiting...")
    finally:
        stop_flag[0] = 1


if __name__ == "__main__":
    main()
