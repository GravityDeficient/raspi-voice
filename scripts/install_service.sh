#!/usr/bin/env bash
set -euo pipefail

SERVICE_FILE="$(cd "$(dirname "$0")" && pwd)/raspivoice.service"

if [ ! -f "$SERVICE_FILE" ]; then
  echo "Service file not found: $SERVICE_FILE" >&2
  exit 1
fi

sudo mkdir -p /home/pi/raspi-voice
sudo cp -r "$(cd "$(dirname "$0")" && cd .. && pwd)" /home/pi/

sudo cp "$SERVICE_FILE" /etc/systemd/system/raspivoice.service
sudo systemctl daemon-reload
sudo systemctl enable raspivoice.service
sudo systemctl start raspivoice.service
echo "Service installed and started. Use: sudo systemctl status raspivoice.service"
