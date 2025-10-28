#!/usr/bin/env bash
set -euo pipefail

SERVICE_FILE="$(cd "$(dirname "$0")" && pwd)/raspi-voice.service"

if [ ! -f "$SERVICE_FILE" ]; then
  echo "Service file not found: $SERVICE_FILE" >&2
  exit 1
fi

sudo cp "$SERVICE_FILE" /etc/systemd/system/raspi-voice.service
sudo systemctl daemon-reload
sudo systemctl enable raspi-voice.service
sudo systemctl start raspi-voice.service
echo "Service installed and started. Use: sudo systemctl status raspi-voice.service"
