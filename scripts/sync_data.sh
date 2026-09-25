#!/usr/bin/env bash
# ============================================================
# sync_data.sh — Transfer large files between Laptop ↔ Desktop
# Uses a shared folder (Google Drive, USB, or network share)
# ============================================================
#
# Usage:
#   bash sync_data.sh save    # Copy data OUT to shared location
#   bash sync_data.sh load    # Copy data IN from shared location
#
# Setup: Set SHARED_DIR to your sync location (pick one):
#   - Google Drive:  ~/Google_Drive/school_data
#   - USB drive:     /media/dung/USB_NAME/school_data
#   - Network share: /mnt/shared/school_data
#   - Same WiFi:     Use rsync (see below)
# ============================================================

set -euo pipefail

# ──── CONFIGURE THIS ────
SHARED_DIR="${SCHOOL_DATA_DIR:-$HOME/Google_Drive/school_data}"
# ─────────────────────────

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ACTION="${1:-help}"

# Folders to sync (large files not in git)
SYNC_DIRS=(
    "Smart_system/Smart_system_ass3/data"
    "Smart_system/Smart_system_ass3/trained"
    "Smart_system/Smart_system_ass4/data"
    "Smart_system/Smart_system_ass4/trained"
    "Smart_system/Smart_system_ass4/results/models"
)

RED='\033[91m'
GREEN='\033[92m'
CYAN='\033[96m'
RESET='\033[0m'

case "$ACTION" in
    save)
        echo -e "${CYAN}Saving data to: $SHARED_DIR${RESET}"
        mkdir -p "$SHARED_DIR"
        for dir in "${SYNC_DIRS[@]}"; do
            src="$PROJECT_ROOT/$dir"
            if [ -d "$src" ]; then
                dest="$SHARED_DIR/$dir"
                mkdir -p "$(dirname "$dest")"
                echo -e "  📤 $dir → $(du -sh "$src" | cut -f1)"
                rsync -a --progress "$src/" "$dest/"
            else
                echo -e "  ⏭️  $dir (not found, skipping)"
            fi
        done
        echo -e "${GREEN}✅ Save complete!${RESET}"
        ;;

    load)
        echo -e "${CYAN}Loading data from: $SHARED_DIR${RESET}"
        for dir in "${SYNC_DIRS[@]}"; do
            src="$SHARED_DIR/$dir"
            if [ -d "$src" ]; then
                dest="$PROJECT_ROOT/$dir"
                mkdir -p "$dest"
                echo -e "  📥 $dir ← $(du -sh "$src" | cut -f1)"
                rsync -a --progress "$src/" "$dest/"
            else
                echo -e "  ⏭️  $dir (not in shared, skipping)"
            fi
        done
        echo -e "${GREEN}✅ Load complete!${RESET}"
        ;;

    help|*)
        echo "Usage: bash sync_data.sh [save|load]"
        echo ""
        echo "  save   Copy data from this machine → shared location"
        echo "  load   Copy data from shared location → this machine"
        echo ""
        echo "Shared location: $SHARED_DIR"
        echo "Override with: export SCHOOL_DATA_DIR=/path/to/shared"
        echo ""
        echo "Synced folders:"
        for dir in "${SYNC_DIRS[@]}"; do
            echo "  - $dir"
        done
        ;;
esac
