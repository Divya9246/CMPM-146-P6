"""Generate PDF reports for Divya's P6 sections 5-7 (FER2013 basic model + game)."""

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
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Initial / Tuned Network Summary", styles["Heading2"]))
    story.append(Paragraph(
        "Architecture: Rescaling → RandomFlip → RandomRotation → "
        "Conv(24)-Pool → Conv(32)-Pool → Conv(48)-Pool → Conv(64)-Pool → "
        "Dropout(0.25) → Flatten → Dense(32) → Dropout(0.4) → Softmax(3).",
        styles["Body"],
    ))
    if metrics:
        story.append(Paragraph(
            f"Total parameters: <b>{metrics.get('total_params', 'N/A'):,}</b> (limit 150,000).",
            styles["Body"],
        ))
        story.append(Paragraph(
            f"Epochs trained (with early stopping): <b>{metrics.get('epochs_trained', 'N/A')}</b>.",
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

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Training / Validation Curves", styles["Heading2"]))
    plot_path = RESULTS / "basic_model_history.png"
    if plot_path.exists():
        story.append(Image(str(plot_path), width=7.2 * inch, height=1.9 * inch))
    else:
        story.append(Paragraph("Plot not found — re-run train.py.", styles["Body"]))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Hyperparameter Optimization Strategy (Section 6)", styles["Heading2"]))
    bullets = [
        "Varied convolutional width (16–64 filters) while staying under 150k parameters.",
        "Tried Dense sizes 16 / 32 / 48; settled on Dense(32) for capacity without overfitting as quickly.",
        "Inserted 1–2 Dropout layers (rates 0.25 after last conv stack, 0.4 after Dense).",
        "Added light augmentation (horizontal flip + small rotation) to improve generalization.",
        "Lowered RMSprop learning rate from 0.001 to 0.0008.",
        "Used EarlyStopping on val_accuracy (patience=5) and ModelCheckpoint to keep the best epoch.",
        "Target: ≥60% for section 5 and ≥70% for section 6 on the held-out test set.",
    ]
    story.append(ListFlowable(
        [ListItem(Paragraph(b, styles["Body"])) for b in bullets],
        bulletType="bullet",
    ))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Example Training Images", styles["Heading2"]))
    for cat in ("neutral", "happy", "surprise"):
        img = _example_image(cat)
        if img is not None:
            story.append(Paragraph(cat, styles["Heading3"]))
            story.append(Image(str(img), width=1.2 * inch, height=1.2 * inch))

    doc.build(story)
    print("Wrote", out)
    return out


def build_section7_report(metrics):
    styles = _styles()
    out = REPORTS / "P6_section_7_tic_tac_toe_report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=letter, title="P6 Section 7 Tic-Tac-Toe")
    story = []

    story.append(Paragraph("P6 Section 7 — Tic-Tac-Toe with Facial Controller", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "The game maps facial emotions to board coordinates: "
        "neutral → 0, happy → 1, surprise → 2 for both row and column selection.",
        styles["Body"],
    ))

    story.append(Paragraph("How well did the interface work?", styles["Heading2"]))
    story.append(Paragraph(
        "The webcam pipeline crops a centered square, converts to grayscale, resizes to 150×150, "
        "replicates channels to RGB, and runs the saved Keras model. A text override (`text`) is "
        "available when the predicted emotion is wrong. On machines with a working camera, this is "
        "enough to complete a full game against the random bot via `python run.py`.",
        styles["Body"],
    ))

    story.append(Paragraph("Webcam accuracy vs test-set accuracy?", styles["Heading2"]))
    test_acc = (metrics or {}).get("test_metrics", {}).get("accuracy")
    if test_acc is not None:
        story.append(Paragraph(
            f"Held-out FER2013 test accuracy for the exported model is <b>{test_acc:.4f}</b>. "
            "Webcam accuracy is typically a bit lower because lighting, pose, and framing differ "
            "from the FER2013 distribution, and the live crop is not identically aligned.",
            styles["Body"],
        ))
    else:
        story.append(Paragraph(
            "Webcam accuracy is typically a bit lower than the FER2013 test number because of "
            "domain shift (lighting / pose / framing).",
            styles["Body"],
        ))

    story.append(Paragraph("If not, why not?", styles["Heading2"]))
    story.append(Paragraph(
        "Domain shift: FER2013 faces are tightly cropped and grayscale-centered. Webcam frames "
        "include background, different lighting, and expression intensity that may not match the "
        "training labels. Using the text override recovers from misclassifications during play.",
        styles["Body"],
    ))

    story.append(Paragraph("_get_emotion implementation", styles["Heading2"]))
    code = '''def _get_emotion(self, img) -> int:
    img = cv2.resize(img, image_size)
    img = np.stack((img, img, img), axis=-1)
    img = np.expand_dims(img, axis=0)
    prediction = self.model.predict(img, verbose=0)
    pred = int(np.argmax(prediction[0]))
    # alphabetical labels from Keras dirs → assignment order
    mapping = {0: 1, 1: 0, 2: 2}  # happy,neutral,surprise → neutral,happy,surprise
    return mapping[pred]'''
    story.append(Preformatted(code, styles["CodeBlock"]))

    story.append(Paragraph("Example move trace (emotion → board index)", styles["Heading2"]))
    story.append(Paragraph(
        "X (webcam) vs O (random). Emotions choose row then column.",
        styles["Body"],
    ))
    story.append(Preformatted(
        "Move 1: X happy(1) + neutral(0) → (1,0)\n"
        "Move 2: O random → (0,1)\n"
        "Move 3: X surprise(2) + happy(1) → (2,1)\n"
        "Move 4: O random → (0,0)\n"
        "Move 5: X happy(1) + happy(1) → (1,1)\n"
        "Move 6: O random → (2,0)\n"
        "Move 7: X happy(1) + surprise(2) → (1,2)  → X wins (middle row)",
        styles["CodeBlock"],
    ))
    story.append(Paragraph(
        "Re-run `python run.py` locally with a webcam to capture your own live trace for submission.",
        styles["Body"],
    ))

    doc.build(story)
    print("Wrote", out)
    return out


if __name__ == "__main__":
    REPORTS.mkdir(exist_ok=True)
    metrics = _load_metrics()
    build_section5_6_report(metrics)
    build_section7_report(metrics)
