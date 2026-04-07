#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# provision_runpod.sh — Download & extract Google Drive archives on RunPod
#
# Usage:  bash provision_runpod.sh <digits>
#
#   digits = any combination of 1 2 3 4 indicating which archives
#            to download.
#
#   Examples:
#     bash provision_runpod.sh 1234   # download all four
#     bash provision_runpod.sh 14     # download #1 and #4 only
#     bash provision_runpod.sh 2      # download #2 only
#
# Flow per archive:
#   1. gdown → /dev/shm  (download to RAM)
#   2. extract  → target dir
#   3. rm archive from /dev/shm
# ──────────────────────────────────────────────────────────────────────

ROOT="/workspace/309"
DATA_ROOT="${ROOT}/data"

# ── Load secrets from .env (data archive links) ─────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a; source "${REPO_ROOT}/.env"; set +a
else
    echo "ERROR: .env not found — copy .env.example and fill in DATA_LINK1..3"
    exit 1
fi

LINK1="${DATA_LINK1:?Set DATA_LINK1 in .env}"
LINK2="${DATA_LINK2:?Set DATA_LINK2 in .env}"
LINK3="${DATA_LINK3:?Set DATA_LINK3 in .env}"

# ── Extraction target dirs (created automatically) ───────────────────
DIR1="${ROOT}"
DIR2="${DATA_ROOT}"
DIR3="${DATA_ROOT}"

# ──────────────────────────────────────────────────────────────────────

if [[ $# -ne 1 ]] || [[ ! "$1" =~ ^[1-3]+$ ]]; then
    echo "Usage: bash provision_runpod.sh <digits>   (digits = any combo of 1 2 3)"
    echo "  e.g.  bash provision_runpod.sh 123"
    exit 1
fi

DIGITS="$1"

mkdir -p "${DATA_ROOT}"
pip install -q gdown
apt-get update -qq && apt-get install -y unzip
apt-get update -qq && apt-get install -y nano

download_and_extract() {
    local idx="$1" link="$2" dest="$3"

    local ram_path="/dev/shm/_dl_${idx}.zip"

    echo "────────────────────────────────────────"
    echo "[${idx}] Downloading → ${ram_path}"
    gdown "$link" -O "$ram_path" --fuzzy

    local fname
    fname="$(basename "$ram_path")"

    mkdir -p "$dest"
    echo "[${idx}] Extracting ${fname} → ${dest}"

    case "$fname" in
        *.tar.gz|*.tgz)  tar -xzf "$ram_path" -C "$dest" ;;
        *.tar.bz2)       tar -xjf "$ram_path" -C "$dest" ;;
        *.tar)           tar -xf  "$ram_path" -C "$dest" ;;
        *.zip)           unzip -qo "$ram_path" -d "$dest" ;;
        *)               echo "[${idx}] Unknown format, copying as-is"
                         cp "$ram_path" "$dest/" ;;
    esac

    rm -f "$ram_path"
    echo "[${idx}] Cleaned /dev/shm/${fname}"
}

for (( i=0; i<${#DIGITS}; i++ )); do
    d="${DIGITS:$i:1}"
    case "$d" in
        1) download_and_extract 1 "$LINK1" "$DIR1" ;;
        2) download_and_extract 2 "$LINK2" "$DIR2" ;;
        3) download_and_extract 3 "$LINK3" "$DIR3" ;;
    esac
done

echo "════════════════════════════════════════"
echo "Done. Contents of ${DATA_ROOT}:"
ls -lh "${DATA_ROOT}"
