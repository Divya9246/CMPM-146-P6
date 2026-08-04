# CMPM 146 — Project 6 (Facial Recognition + Tic-Tac-Toe + Transfer Learning)

**Team**
- **Divya Machiraju** (`Divya9246`) — Sections 5–7: facial CNN, optimization, evaluation, and webcam tic-tac-toe
- **Yuhui Zhen / YHZ** (`yzhen174`) — Section 8: dog-versus-cat transfer learning and randomized comparison model

---

## What this project is

We trained a facial emotion classifier (neutral / happy / surprise), used the webcam so faces can pick tic-tac-toe moves, and Yuhui reused the CNN backbone for a dog-vs-cat transfer learning experiment (plus a random-weight control).

---

## Datasets

**Facial emotions (Sections 5–7)**  
[FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013?resource=download)  
Classes: `neutral`, `happy`, `surprise`. Images are resized to 150×150.

**Transfer learning (Section 8)**  
[Dog vs Cat Images Data on Kaggle](https://www.kaggle.com/datasets/kunalgupta2616/dog-vs-cat-images-data)

---

## Official models (do not mix these up)

| What | File | Test accuracy | Params |
|------|------|---------------|--------|
| Section 5 initial | `results/section5_initial_model.keras` | **~72.62%** | 110,755 |
| Section 6/7 final | `results/section6_final_model.keras` | **~77.75%** | 149,683 |

Older tuning runs live under `results/attempts/` as `attempt_1_…`, `attempt_2_…`, etc. They are **not** submission models. See `results/attempts/README.txt`.

**Locked final accuracy:** use **77.75%** everywhere (`0.777488…` in `results/section6_final_model_metrics.json`). Do not retrain before submitting if you want that number to stay. Check it with:

```bash
python3 evaluate_model.py
```

---

## Section 5 — Initial network

Simpler baseline CNN (no augmentation):

- Rescaling → Conv(16)-Pool → Conv(32)-Pool → Conv(64)-Pool → Conv(64)-Pool → Flatten → Dense(16) → Dropout(0.3) → Softmax(3)
- Optimizer: RMSprop (lr=0.001)
- Under the **150,000** parameter limit (**110,755** params)
- Best validation accuracy ≈ **0.704**
- **Test accuracy ≈ 72.62%** (assignment asks for ≥ 60%)

Files: `results/section5_initial_model.keras` (+ history plot, metrics JSON, summary text)

```bash
python3 train_initial.py
```

---

## Section 6 — Optimized network

This is the model used for evaluation and the webcam game.

### Architecture

- Rescaling + light augmentation: horizontal flip, small rotation, contrast
- Conv(24)-Pool → Conv(32)-Pool → Conv(48)-Pool → Conv(64)-Pool → Dropout(0.25)
- Flatten (named `"flatten"` for transfer compatibility) → Dense(32) → Dropout(0.4) → Softmax(3)
- Optimizer: Adam (lr=0.0009)
- **149,683** parameters (under the 150,000 limit)

### Training setup

- Seeds (`SEED = 42`) for Python, NumPy, and TensorFlow so runs are more repeatable across machines
- EarlyStopping + ModelCheckpoint + ReduceLROnPlateau
- Trained up to 35 epochs (best checkpoint kept)

### Results (official / locked)

- Best validation accuracy ≈ **0.786**
- **Test accuracy ≈ 77.75%**
- Confusion matrix shape (from evaluation of the saved model): strong diagonal for all three classes; some confusion between neutral and happy, which also shows up a bit on webcam

What we tried while tuning: more epochs, BatchNorm (didn’t help enough / was abandoned), different dense sizes, etc. Those extras are renamed under `results/attempts/`.

```bash
python3 train.py              # overwrites the final model — only if you mean to
python3 evaluate_model.py     # score the saved final model without retraining
```

---

## Section 7 — Webcam tic-tac-toe (real game trace)

### How the controller works

- `player.py` → `UserWebcamPlayer` loads `results/section6_final_model.keras`
- Webcam frame → center crop → grayscale → resize 150×150 → stack to RGB → predict
- Keras folder order is alphabetical (`happy`, `neutral`, `surprise`), but the assignment wants `neutral=0`, `happy=1`, `surprise=2`, so we remap `{0: 1, 1: 0, 2: 2}`
- If a predicted cell is already taken (or the face read is wrong), type `text` then `0` / `1` / `2` to override

### Webcam vs test-set accuracy

Held-out test accuracy for the final model is **~77.75%**. Live webcam accuracy was lower: lighting, framing, and expression strength differ from the training images, so neutral was over-predicted sometimes until the text override was used.

### Real game we played (Player X = random bot, Player O = webcam)

This is a full game from an actual `python3 run.py` session. **Player X won across the top row.**

```text
Start:
| | | |
| | | |
| | | |

1. X (random) → (0,1)
| |X| |
| | | |
| | | |

   O tried neutral + happy → (0,1)
   Position already taken.

2. O surprise + happy → (2,1)
| |X| |
| | | |
| |O| |

3. X (random) → (0,0)
|X|X| |
| | | |
| |O| |

   O tried neutral + happy → (0,1)
   Position already taken.

   O tried surprise + happy → (2,1)
   Position already taken.

4. O happy + happy → (1,1)
|X|X| |
| |O| |
| |O| |

5. X (random) → (1,0)
|X|X| |
|X|O| |
| |O| |

6. O surprise + neutral → (2,0)
|X|X| |
|X|O| |
|O|O| |

7. X (random) → (0,2)
|X|X|X|
|X|O| |
|O|O| |

Player X won across the top row (0,0)-(0,1)-(0,2).
```

### How well the interface worked

The webcam pipeline works end-to-end. Occupied-cell retries and a couple of wrong emotion reads were handled with the built-in text override so the game could finish. The board above is the real terminal outcome, not a made-up win for O.

```bash
python3 run.py
```

---

## Section 8 — Transfer + random (Yuhui)

Yuhui owns the detailed numbers / PDF for this section. High level:

- Transfer model: load trained facial base, freeze backbone, keep `flatten`, add a new dense head for dog vs cat
- Random model: same idea but randomize weights (control)
- In Yuhui’s tests, transfer looked stronger in the first few epochs; random could catch up later overall. Transfer still showed the early validation accuracy the assignment wants after epoch 1
- Base-model accuracy can differ by machine / retrain (~10% difference showed up once), which can affect transfer/random since they load that base model

```bash
python3 train_transfer.py
```

---

## Setup

The original assignment used **TensorFlow 2.12**. Our compatibility imports also support newer TensorFlow/Keras versions we tested on our machines. We do **not** guarantee every modern TensorFlow version will work.

```bash
pip install -r requirements.txt

# FER-2013 → train/ and test/ with neutral, happy, surprise folders
# Dog-vs-cat data where preprocess / train_transfer expect it
```

Useful commands:

```bash
python3 train_initial.py      # section 5 baseline
python3 train.py              # section 6 optimized (overwrites final model — careful)
python3 evaluate_model.py     # score saved final model (keep ~77.75%)
python3 run.py                # tic-tac-toe with webcam
python3 train_transfer.py     # Yuhui’s transfer / random training
```

On Mac use `python3`. If Rescaling / augmentation imports fail, the code tries both the older `experimental.preprocessing` path and the newer Keras path.

---

## Submission checklist (Divya sections 5–7)

- Final facial model under **150k** params (**149,683**) and above 60% test accuracy (**~77.75%**)
- Initial network saved and reported separately for section 5 (**~72.62%**)
- Webcam game works with emotion → board mapping + real move trace above
- Seeds in train scripts for more repeatable runs
- Do **not** retrain right before submit if you want the locked **77.75%** — use `evaluate_model.py` and the committed `.keras` file
- Yuhui attaches / completes the section 8 transfer + random writeup on their side
