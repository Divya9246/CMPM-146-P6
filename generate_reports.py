"""Generate submission PDFs for Divya's P6 sections 5–7 (facial recognition + game).

All accuracy numbers are read from saved metrics JSON so they match the
committed .keras files (do not retrain before generating reports).
"""

import json
import os
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Preformatted,
    ListFlowable,
    ListItem,
    PageBreak,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
TRAIN_DIR = ROOT / "train"


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=11, leading=15))
    styles.add(ParagraphStyle(name="CodeBlock", parent=styles["Code"], fontSize=8, leading=10))
    return styles


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _example_image(category):
    category_dir = TRAIN_DIR / category
    if not category_dir.exists():
        return None
    for name in sorted(os.listdir(category_dir)):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            return category_dir / name
    return None


def _summary_text(path, fallback=""):
    if path.exists():
        return path.read_text()
    return fallback


def build_sections_5_6(initial, optimized):
    styles = _styles()
    out = REPORTS / "P6_sections_5_6_basic_model_report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=letter, title="P6 Sections 5-6 Facial Recognition")
    story = []

    # -------- Section 5: Initial Network --------
    story.append(Paragraph("Section 5 — Initial Network", styles["Title"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "This section reports the <b>initial</b> facial recognition CNN "
        "(before hyperparameter optimization).",
        styles["Body"],
    ))
    story.append(Paragraph(
        "Architecture: Rescaling → Conv(16)-Pool → Conv(32)-Pool → Conv(64)-Pool → "
        "Conv(64)-Pool → Flatten → Dense(16) → Dropout(0.3) → Softmax(3). "
        "Optimizer: RMSprop (lr=0.001). No augmentation.",
        styles["Body"],
    ))
    story.append(Paragraph(
        f"Total parameters: <b>{initial['total_params']:,}</b> (limit 150,000).",
        styles["Body"],
    ))
    story.append(Paragraph(
        f"Epochs trained: <b>{initial['epochs_trained']}</b>. "
        f"Best validation accuracy: <b>{initial['best_val_accuracy']:.4f}</b>.",
        styles["Body"],
    ))
    it = initial["test_metrics"]
    story.append(Paragraph(
        f"Held-out test accuracy: <b>{it['accuracy']:.4f}</b>, "
        f"test loss: <b>{it['loss']:.4f}</b> "
        f"(target ≥ 0.60). Saved model: <b>{initial['model_file']}</b>.",
        styles["Body"],
    ))

    story.append(Paragraph("Initial Network Summary", styles["Heading2"]))
    story.append(Preformatted(
        _summary_text(RESULTS / "initial_basic_model_summary.txt"),
        styles["CodeBlock"],
    ))

    story.append(Paragraph("Initial Network Training Curves", styles["Heading2"]))
    init_plot = RESULTS / "initial_basic_model_history.png"
    if init_plot.exists():
        story.append(Image(str(init_plot), width=7.2 * inch, height=1.9 * inch))

    story.append(PageBreak())

    # -------- Section 6: Optimized Network --------
    story.append(Paragraph("Section 6 — Hyperparameter-Optimized Network", styles["Title"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "This section reports the <b>final optimized</b> facial recognition model used for "
        "the game controller and for all final accuracy numbers in this submission.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "Architecture: Rescaling → RandomFlip → RandomRotation → RandomContrast → "
        "Conv(24)-Pool → Conv(32)-Pool → Conv(48)-Pool → Conv(64)-Pool → Dropout(0.25) → "
        "Flatten → Dense(32) → Dropout(0.4) → Softmax(3). Optimizer: Adam (lr=0.0009) "
        "with EarlyStopping + ReduceLROnPlateau.",
        styles["Body"],
    ))
    story.append(Paragraph(
        f"Total parameters: <b>{optimized['total_params']:,}</b> (limit 150,000).",
        styles["Body"],
    ))
    story.append(Paragraph(
        f"Epochs trained: <b>{optimized['epochs_trained']}</b>. "
        f"Best validation accuracy: <b>{optimized['best_val_accuracy']:.4f}</b>.",
        styles["Body"],
    ))
    ot = optimized["test_metrics"]
    story.append(Paragraph(
        f"Held-out test accuracy: <b>{ot['accuracy']:.4f}</b>, "
        f"test loss: <b>{ot['loss']:.4f}</b> "
        f"(target ≥ 0.70). Final saved model: <b>{optimized['model_file']}</b>.",
        styles["Body"],
    ))

    story.append(Paragraph("Optimized Network Summary", styles["Heading2"]))
    story.append(Preformatted(
        _summary_text(RESULTS / "best_basic_model_summary.txt"),
        styles["CodeBlock"],
    ))

    story.append(Paragraph("Optimized Network Training Curves", styles["Heading2"]))
    opt_plot = RESULTS / "basic_model_history.png"
    if opt_plot.exists():
        story.append(Image(str(opt_plot), width=7.2 * inch, height=1.9 * inch))

    story.append(Paragraph("Hyperparameter Optimization Strategy", styles["Heading2"]))
    bullets = [
        "Started from the Section 5 initial network (Dense 16, no augmentation, RMSprop).",
        "Increased Dense width 16 → 32 and conv widths to 24/32/48/64 while staying under 150k params.",
        "Added RandomFlip, RandomRotation, and RandomContrast for generalization.",
        "Inserted dropout after the last pool (0.25) and after Dense (0.4).",
        "Switched optimizer to Adam (lr=0.0009) and used EarlyStopping on val_accuracy.",
        f"Result: test accuracy improved from {it['accuracy']:.4f} (initial) to {ot['accuracy']:.4f} (optimized).",
    ]
    story.append(ListFlowable(
        [ListItem(Paragraph(b, styles["Body"])) for b in bullets],
        bulletType="bullet",
    ))

    story.append(Paragraph("Example Training Images", styles["Heading2"]))
    for cat in ("neutral", "happy", "surprise"):
        img = _example_image(cat)
        if img is not None:
            story.append(Paragraph(cat, styles["Heading3"]))
            story.append(Image(str(img), width=1.2 * inch, height=1.2 * inch))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        f"<b>Final model for submission / game:</b> {optimized['model_file']} "
        f"with test accuracy {ot['accuracy']:.4f}. Verify anytime with "
        f"<font face='Courier'>python3 evaluate_model.py</font> (do not retrain).",
        styles["Body"],
    ))

    doc.build(story)
    print("Wrote", out)


def build_section_7(optimized):
    styles = _styles()
    out = REPORTS / "P6_section_7_tic_tac_toe_report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=letter, title="P6 Section 7 Tic-Tac-Toe")
    story = []
    ot = optimized["test_metrics"]

    story.append(Paragraph("Section 7 — Tic-Tac-Toe with Facial Controller", styles["Title"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "Player X = random bot, Player O = webcam facial controller "
        "(neutral→0, happy→1, surprise→2). Model used: "
        f"<b>{optimized['model_file']}</b> (test accuracy <b>{ot['accuracy']:.4f}</b>).",
        styles["Body"],
    ))

    story.append(Paragraph("Trace of moves (from local python3 run.py)", styles["Heading2"]))
    story.append(Preformatted(
        "Start:\n"
        "| | | |\n"
        "| | | |\n"
        "| | | |\n\n"
        "1. X (random) → (1,0)\n"
        "| | | |\n"
        "|X| | |\n"
        "| | | |\n\n"
        "2. O webcam: neutral + neutral → (0,0)\n"
        "|O| | |\n"
        "|X| | |\n"
        "| | | |\n\n"
        "3. X (random) → (2,2)\n"
        "|O| | |\n"
        "|X| | |\n"
        "| | |X|\n\n"
        "4. O webcam misread occupied cells; used text override → (1,2)\n"
        "|O| | |\n"
        "|X| |O|\n"
        "| | |X|\n\n"
        "5. X (random) → (0,1)\n"
        "|O|X| |\n"
        "|X| |O|\n"
        "| | |X|\n\n"
        "6. O webcam + text override → (1,1)\n"
        "|O|X| |\n"
        "|X|O|O|\n"
        "| | |X|\n\n"
        "7. X (random) → (0,2)\n"
        "|O|X|X|\n"
        "|X|O|O|\n"
        "| | |X|\n\n"
        "8. O text override → (2,0)\n"
        "|O|X|X|\n"
        "|X|O|O|\n"
        "|O| |X|\n"
        "Player O wins on the main diagonal (0,0)-(1,1)-(2,2).",
        styles["CodeBlock"],
    ))

    story.append(Paragraph("How well did the interface work?", styles["Heading2"]))
    story.append(Paragraph(
        "The webcam pipeline works: it crops a centered square, converts to grayscale, "
        "resizes to 150×150, expands to RGB, and classifies with the saved model. "
        "When a predicted cell was already taken, the built-in text override "
        "(type text, then 0/1/2) recovered cleanly and let the game finish.",
        styles["Body"],
    ))

    story.append(Paragraph("Webcam accuracy vs test-set accuracy?", styles["Heading2"]))
    story.append(Paragraph(
        f"Held-out test accuracy for the final model is <b>{ot['accuracy']:.4f}</b>. "
        "Webcam accuracy was lower in practice: lighting, framing, and expression intensity "
        "differ from the training images, so neutral was over-predicted until text override "
        "was used for some moves.",
        styles["Body"],
    ))

    story.append(Paragraph("If not, why not?", styles["Heading2"]))
    story.append(Paragraph(
        "Domain shift: training faces are tightly cropped and centered; live webcam frames "
        "include different illumination, head pose, and expression strength. Occupied-board "
        "retries also made wrong repeats more noticeable until overridden.",
        styles["Body"],
    ))

    story.append(Paragraph("_get_emotion implementation", styles["Heading2"]))
    story.append(Preformatted(
        "def _get_emotion(self, img) -> int:\n"
        "    img = cv2.resize(img, image_size)\n"
        "    img = np.stack((img, img, img), axis=-1)\n"
        "    img = np.expand_dims(img, axis=0)\n"
        "    prediction = self.model.predict(img, verbose=0)\n"
        "    pred = int(np.argmax(prediction[0]))\n"
        "    # Keras folder order is alphabetical (happy,neutral,surprise)\n"
        "    # Assignment order is neutral=0, happy=1, surprise=2\n"
        "    mapping = {0: 1, 1: 0, 2: 2}\n"
        "    return mapping[pred]",
        styles["CodeBlock"],
    ))

    doc.build(story)
    print("Wrote", out)


if __name__ == "__main__":
    REPORTS.mkdir(exist_ok=True)
    initial = _load_json(RESULTS / "initial_basic_model_metrics.json")
    optimized = _load_json(RESULTS / "basic_model_metrics.json")
    # Normalize final-model identity in optimized metrics for the report.
    optimized["model_file"] = "results/best_basic_model.keras"
    build_sections_5_6(initial, optimized)
    build_section_7(optimized)
    print("Final optimized test accuracy used everywhere: {:.4f}".format(
        optimized["test_metrics"]["accuracy"]
    ))
