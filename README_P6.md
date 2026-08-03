# CMPM 146 — P6 Export (FER2013 / facial controller)

This package covers **Divya's sections** of P6 using the
[FER2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset
(happy / neutral / surprise):

| Section | Deliverable |
|--------|-------------|
| 5–6 | CNN classifier + hyperparameter tuning (≥70% test accuracy target) |
| 7 | Webcam tic-tac-toe controller (`player.py` + `run.py`) |
| 8 | Transfer / random models (teammate: dog-vs-cat) — code included |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Put Kaggle FER2013 under ./kaggle/{train,test}/...
python export_dataset.py
python train.py                 # writes results/best_basic_model.keras
python generate_reports.py      # writes reports/*.pdf
python run.py                   # webcam tic-tac-toe
```

Canonical model path (used by the game + transfer models): `results/best_basic_model.keras`.

## Rebuild this zip

```bash
bash make_export.sh P6_export
```
