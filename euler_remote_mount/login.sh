#!/bin/bash
KEY_PATH="$HOME/.ssh/id_ed25519"
MOUNT_HOME="/home/sebi/MSc/3.Sem/semester_thesis/local-git/semester_project_autobrep/euler_remote_mount/home"
MOUNT_SCRATCH="/home/sebi/MSc/3.Sem/semester_thesis/local-git/semester_project_autobrep/euler_remote_mount/scratch"

# Ensure mount directories exist and are empty
mkdir -p "$MOUNT_HOME" "$MOUNT_SCRATCH"

# Mount home and scratch with SSH key
sshfs -o 'ssh_command=ssh -i '"$KEY_PATH" slochmann@euler.ethz.ch:/cluster/home/slochmann "$MOUNT_HOME" &
sshfs -o 'ssh_command=ssh -i '"$KEY_PATH" slochmann@euler.ethz.ch:/cluster/scratch/slochmann "$MOUNT_SCRATCH" &

# Connect terminal via SSH
ssh -i "$KEY_PATH" slochmann@euler.ethz.ch
