#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

log() { printf '%s\n' "$*" >&2; }
die() { log "Error: $*"; exit 1; }

# --- Locate project root ---
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
PROJECT_ROOT="$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel 2>/dev/null || (cd "$SCRIPT_DIR/.." && pwd -P))"

BIODATA_DIR="$PROJECT_ROOT/biodata_files"
BIODATA_RAW="$PROJECT_ROOT/biodata_raw.txt"

# --- Ensure biodata_files folder exists ---
[[ -d "$BIODATA_DIR" ]] || die "biodata_files directory not found at: $BIODATA_DIR"

# --- Ensure biodata_raw.txt exists (create if absent) ---
if [[ ! -f "$BIODATA_RAW" ]]; then
  log "biodata_raw.txt not found — creating: $BIODATA_RAW"
  touch "$BIODATA_RAW"
fi

# --- Collect all readable files in biodata_files ---
mapfile -t BIODATA_FILES < <(find "$BIODATA_DIR" -maxdepth 1 -type f | sort)

if [[ ${#BIODATA_FILES[@]} -eq 0 ]]; then
  log "Warning: biodata_files directory is empty — nothing to process."
else
  log "Processing ${#BIODATA_FILES[@]} file(s) from: $BIODATA_DIR"

  for filepath in "${BIODATA_FILES[@]}"; do
    filename="$(basename "$filepath")"
    log "  → $filename"

    appended=0
    skipped=0

    while IFS= read -r line || [[ -n "$line" ]]; do
      [[ -z "$line" ]] && continue

      if grep -qxF "$line" "$BIODATA_RAW"; then
        (( skipped++ )) || true
      else
        printf '%s\n' "$line" >> "$BIODATA_RAW"
        (( appended++ )) || true
      fi
    done < "$filepath"

    log "    appended=$appended  skipped(duplicate)=$skipped"
  done

  log "✓ Biodata ingestion complete → $BIODATA_RAW"
fi