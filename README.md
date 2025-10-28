# raspi-voice (POC)

A lightweight Raspberry Pi 5 voice assistant proof-of-concept that runs on the Pi (not your PC):
- Listens for a wake word (grammar-based via Vosk) like "hey notes".
- After wake, captures speech until a brief pause using WebRTC VAD.
- Transcribes locally with Vosk and saves a Markdown note plus WAV audio.

No cloud services required. Designed for a USB microphone on Raspberry Pi 5.

## Prereqs (on Raspberry Pi OS)

- Python 3.11+ recommended
- System packages for audio and build tools:

```powershell
# PowerShell (Windows shown for reference); on Pi use bash/apt:
# sudo apt update
# sudo apt install -y python3-pip python3-venv portaudio19-dev libatlas-base-dev
```

On the Pi, run the apt commands with sudo in your shell. PortAudio is needed for PyAudio/sounddevice.

## Setup (on the Pi)

```bash
# On the Pi 5
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

First run will automatically download the small English Vosk model to `models/` (few tens of MBs).

## Run (interactive)

List devices to find your USB mic:

```bash
python -m src.app --list-devices
```

Then start listening (replace the device value with an index or name substring, e.g., "USB"):

```bash
python -m src.app --device "USB" --once
```

Change the wake word(s) on the fly with a comma-separated list:

```bash
python -m src.app --device "USB" --once --wake-words "hey notebook, okay notebook"
# Multiple accepted phrases are allowed (case-insensitive)
```

Say: "hey notes". Speak your note; pause for ~1 second to finish. The app saves:
- Markdown: `notes/note_YYYY-MM-DD_HH-MM-SS.md`
- Audio: `notes/audio/note_YYYY-MM-DD_HH-MM-SS.wav`

Drop the `--once` flag to keep it running for multiple notes.

Optional flags:
- `--vad 0..3` VAD aggressiveness (default 2; higher is stricter)
- `--silence <ms>` trailing silence to stop (default 900)
- `--models <dir>`, `--notes <dir>`, `--audio <dir>` to override paths

## Obsidian integration

Point `--notes` to your Obsidian vault's folder (or a subfolder) to have notes appear automatically. Example:

```bash
python -m src.app --notes "/home/pi/Obsidian/MyVault/Voice" --device "USB"
```

## Run on boot (systemd)

1) Adjust the service if needed (username, WorkingDirectory, device flag): `scripts/raspi-voice.service`.
2) Copy the repo to `/home/pi/raspi-voice`, create venv and install deps there.
3) Install the service:

```bash
sudo cp scripts/raspi-voice.service /etc/systemd/system/raspi-voice.service
sudo systemctl daemon-reload
sudo systemctl enable raspi-voice.service
sudo systemctl start raspi-voice.service
```

Check status:
```bash
sudo systemctl status raspi-voice.service
```

## Troubleshooting

- If audio fails to start, run `--list-devices` and pick a device with input channels. You can use an index or a name substring.
- On some USB mics, 16 kHz mono is supported natively; if not, try changing the default input format in system settings.
- If wake word misses, try speaking clearly or adjust wake phrases in `WAKE_WORDS` inside `src/app.py`.

## Future work

- Hotword models (e.g., Porcupine/Snowboy-like) for better wake detection
- Intent parsing to tag notes or route to tasks
- Background service via systemd

## License

MIT