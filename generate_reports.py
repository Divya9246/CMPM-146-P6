"""Generate PDF reports for Divya's P6 sections 5-7 only (FER2013 + game)."""

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


def _load_metrics():
    path = RESULTS / "basic_model_metrics.json"
    if not path.exists():
        return None
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


def build_section5_6_report(metrics):
    styles = _styles()
    out = REPORTS / "P6_sections_5_6_basic_model_report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=letter, title="P6 Sections 5-6 Basic Model")
    story = []

    story.append(Paragraph("P6 Sections 5 &amp; 6 — Facial Emotion CNN (FER2013)", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "Dataset: Kaggle FER2013 (happy / neutral / surprise), 5000 training images via export_dataset.py.",
        styles["Body"],
    ))

    story.append(Paragraph("Network Summary", styles["Heading2"]))
    story.append(Paragraph(
        "Rescaling → RandomFlip → RandomRotation → Conv(24)-Pool → Conv(32)-Pool → "
        "Conv(48)-Pool → Conv(64)-Pool → Dropout(0.25) → Flatten → Dense(32) → "
        "Dropout(0.4) → Softmax(3). Optimizer: Adam (lr=0.0009).",
        styles["Body"],
    ))
    if metrics:
        story.append(Paragraph(
            f"Total parameters: <b>{metrics.get('total_params', 0):,}</b> (limit 150,000).",
            styles["Body"],
        ))
        story.append(Paragraph(
            f"Epochs trained: <b>{metrics.get('epochs_trained', 'N/A')}</b> "
            f"(EarlyStopping + ReduceLROnPlateau).",
            styles["Body"],
        ))
        story.append(Paragraph(
            f"Best validation accuracy: <b>{metrics.get('best_val_accuracy', 0):.4f}</b>.",
            styles["Body"],
        ))
        test = metrics.get("test_metrics", {})
        story.append(Paragraph(
            f"Held-out test accuracy: <b>{test.get('accuracy', 0):.4f}</b>, "
            f"test loss: <b>{test.get('loss', 0):.4f}</b>.",
            styles["Body"],
        ))
        story.append(Paragraph(
            f"Saved model: <b>{metrics.get('model_file', 'results/best_basic_model.keras')}</b>.",
            styles["Body"],
        ))

    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Training / Validation Curves", styles["Heading2"]))
    plot_path = RESULTS / "basic_model_history.png"
    if plot_path.exists():
        story.append(Image(str(plot_path), width=7.2 * inch, height=1.9 * inch))

    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Hyperparameter Optimization Strategy (Section 6)", styles["Heading2"]))
    bullets = [
        "Swept convolutional widths (16–64) while staying under 150k parameters.",
        "Compared Dense sizes 16 vs 32; Dense(32) improved capacity without exceeding the limit.",
        "Inserted dropout after the last conv stack (0.25) and after Dense (0.4).",
        "Added light augmentation (horizontal flip + small rotation).",
        "Switched optimizer to Adam (lr=0.0009) with ReduceLROnPlateau.",
        "Used EarlyStopping on val_accuracy to keep the best epoch before overfitting.",
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

    doc.build(story)
    print("Wrote", out)


def build_section7_report(metrics):
    styles = _styles()
    out = REPORTS / "P6_section_7_tic_tac_toe_report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=letter, title="P6 Section 7 Tic-Tac-Toe")
    story = []

    story.append(Paragraph("P6 Section 7 — Tic-Tac-Toe with Facial Controller", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "Emotions map to board indices: neutral→0, happy→1, surprise→2 "
        "(row then column). Run with `python run.py`.",
        styles["Body"],
    ))

    story.append(Paragraph("How well did the interface work?", styles["Heading2"]))
    story.append(Paragraph(
        "The webcam path crops a centered square, converts to grayscale, resizes to 150×150, "
        "expands to 3 channels, and runs `results/best_basic_model.keras`. "
        "The `text` override recovers from wrong detections.",
        styles["Body"],
    ))

    story.append(Paragraph("Webcam accuracy vs test-set accuracy?", styles["Heading2"]))
    test_acc = (metrics or {}).get("test_metrics", {}).get("accuracy")
    if test_acc is not None:
        story.append(Paragraph(
            f"Held-out FER2013 test accuracy is <b>{test_acc:.4f}</b>. "
            "Webcam accuracy is usually a bit lower because of lighting/pose/framing domain shift.",
            styles["Body"],
        ))

    story.append(Paragraph("If not, why not?", styles["Heading2"]))
    story.append(Paragraph(
        "FER2013 faces are tightly cropped and centered; webcam frames differ in illumination, "
        "expression intensity, and alignment.",
        styles["Body"],
    ))

    story.append(Paragraph("_get_emotion implementation", styles["Heading2"]))
    code = '''def _get_emotion(self, img) -> int:
    img = cv2.resize(img, image_size)
    img = np.stack((img, img, img), axis=-1)
    img = np.expand_dims(img, axis=0)
    prediction = self.model.predict(img, verbose=0)
    pred = int(np.argmax(prediction[0]))
    # Keras dir order is alphabetical (happy,neutral,surprise)
    # Assignment order is neutral=0, happy=1, surprise=2
    mapping = {0: 1, 1: 0, 2: 2}
    return mapping[pred]'''
    story.append(Preformatted(code, styles["CodeBlock"]))

    story.append(Paragraph("Example move trace", styles["Heading2"]))
    story.append(Preformatted(
        "Move 1: X happy(1)+neutral(0) → (1,0)\n"
        "Move 2: O random → (0,1)\n"
        "Move 3: X surprise(2)+happy(1) → (2,1)\n"
        "Move 4: O random → (0,0)\n"
        "Move 5: X happy(1)+happy(1) → (1,1)\n"
        "Move 6: O random → (2,0)\n"
        "Move 7: X happy(1)+surprise(2) → (1,2) → X wins\n\n"
        "Replace this with your live webcam trace before Canvas submit.",
        styles["CodeBlock"],
    ))

    doc.build(story)
    print("Wrote", out)


if __name__ == "__main__":
    REPORTS.mkdir(exist_ok=True)
    metrics = _load_metrics()
    build_section5_6_report(metrics)
    build_section7_report(metrics)
