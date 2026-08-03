#!/usr/bin/env bash
# Build a Canvas/Colab-friendly export zip for Divya's FER2013 P6 sections.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

OUT_DIR="${1:-P6_export}"
ZIP_NAME="${OUT_DIR}.zip"

rm -rf "$OUT_DIR" "$ZIP_NAME"
mkdir -p "$OUT_DIR"

# Core code required by the writeup / Colab instructions
cp -r models "$OUT_DIR/"
cp config.py preprocess.py train.py train_transfer.py export_dataset.py \
   show_examples.py player.py game.py gui.py run.py requirements.txt "$OUT_DIR/"

# Trained artifacts + plots + reports (if present)
mkdir -p "$OUT_DIR/results" "$OUT_DIR/reports"
if [[ -f results/best_basic_model.keras ]]; then
  cp results/best_basic_model.keras "$OUT_DIR/results/"
fi
if [[ -f results/best_basic_model.npy ]]; then
  cp results/best_basic_model.npy "$OUT_DIR/results/"
fi
if [[ -f results/basic_model_history.png ]]; then
  cp results/basic_model_history.png "$OUT_DIR/results/"
fi
if [[ -f results/basic_model_metrics.json ]]; then
  cp results/basic_model_metrics.json "$OUT_DIR/results/"
fi
cp reports/*.pdf "$OUT_DIR/reports/" 2>/dev/null || true

# Exported FER2013 subset (happy/neutral/surprise) — not the full kaggle dump
if [[ -d train && -d test ]]; then
  cp -r train test "$OUT_DIR/"
fi

# Helper scripts
cp generate_reports.py make_export.sh "$OUT_DIR/" 2>/dev/null || true
[[ -f README_P6.md ]] && cp README_P6.md "$OUT_DIR/README.md"

# Drop caches
find "$OUT_DIR" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

zip -qr "$ZIP_NAME" "$OUT_DIR"
echo "Created $ZIP_NAME"
du -sh "$ZIP_NAME" "$OUT_DIR"
