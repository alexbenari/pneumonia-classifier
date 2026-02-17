#!/usr/bin/env bash
# Quick start (from local machine):
# 1) Copy this script to EC2:
#    scp -i /path/to/key.pem C:/dev/pneumonia-classifier/start-pneumonia.sh ec2-user@<EC2_PUBLIC_IP>:~/start-pneumonia.sh
# 2) SSH into EC2:
#    ssh -i /path/to/key.pem ec2-user@<EC2_PUBLIC_IP>
# 3) Make it executable and run:
#    chmod +x ~/start-pneumonia.sh
#    ~/start-pneumonia.sh
#
# If your AMI uses 'ubuntu' instead of 'ec2-user', replace the username.
set -euo pipefail

source /opt/pytorch/bin/activate

DEVICE="/dev/nvme1n1"
MOUNTPOINT="/data"
REPO_URL="https://github.com/alexbenari/pneumonia-classifier.git"
REPO_DIR="$MOUNTPOINT/pneumonia-classifier"

# Format only if no filesystem exists on the device.
if ! sudo blkid "$DEVICE" >/dev/null 2>&1; then
  sudo mkfs.xfs -f "$DEVICE"
fi

sudo mkdir -p "$MOUNTPOINT"
if ! mountpoint -q "$MOUNTPOINT"; then
  sudo mount "$DEVICE" "$MOUNTPOINT"
fi

sudo chown -R ec2-user:ec2-user "$MOUNTPOINT"
cd "$MOUNTPOINT"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL"
fi

cd "$REPO_DIR"
echo "Ready in: $(pwd)"

