# CMPM 146 — Project 6 (Facial Recognition + Tic-Tac-Toe + Transfer Learning)

Hey! This is our P6 repo.

**Team**
- Divya Machiraju (`Divya9246`) — facial recognition CNN, training, webcam tic-tac-toe (sections 5–7)
- Yuhui Zhen / YHZ (`yzhen174`) — dog-vs-cat transfer model + random model (section 8)

We used Discord to split the work early on, shared datasets, and kept checking each other’s pushes / PRs so both parts would plug together.

---

## What this project is

We built a facial emotion classifier (neutral / happy / surprise), hooked it up so a webcam can play tic-tac-toe by making faces, and then Yuhui reused the CNN backbone for a dog-vs-cat transfer learning experiment (plus a random-weight control).

---

## Datasets we used

**Facial emotions (Divya’s part)**  
[FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013?resource=download)  
We use the three classes the assignment asks for: `neutral`, `happy`, `surprise`. Images get resized to 150×150 for training / prediction.

**Transfer learning (Yuhui’s part)**  
[Dog vs Cat Images Data on Kaggle](https://www.kaggle.com/datasets/kunalgupta2616/dog-vs-cat-images-data)  
Train / val / test folders for cats and dogs.

---

## How we split the work (short story)

1. Divya sent the P6 writeup / plan; Yuhui said it looked good and shared GitHub (`yzhen174`). Invite went out, repo shared.
2. Divya used FER-2013; Yuhui later picked the dog-vs-cat set for transfer.
3. Divya finished the CNN + training + webcam game first (~74.7% at that point), pushed, and Yuhui pulled once the PR was actually merged (classic “empty pull” scare — it was just the PR not merged yet).
4. Yuhui finished transfer + random models, noted transfer looked stronger early (first few epochs) and random caught up later overall — but transfer still had the early val accuracy bump the assignment cares about. Yuhui also saw ~64% when retraining the basic model on their machine (about 10% off Divya’s number), which matters because transfer/random load that base model.
5. Divya kept tuning the facial model and locked seeds so results stay more consistent across machines. Final locked test accuracy on the shared weights: **~77.75%**.
6. Yuhui later tweaked transfer/random (epochs / dense size) when a run didn’t meet requirements, and shared their dog-vs-cat PDF report. Divya’s focus stayed on sections 5–7; Yuhui owns the section 8 writeup / numbers.

---

## Divya’s part (sections 5–7) — the main focus here

### Section 5 — Initial network

Simpler baseline CNN (no augmentation):

- Rescaling → Conv(16)-Pool → Conv(32)-Pool → Conv(64)-Pool → Conv(64)-Pool → Flatten → Dense(16) → Dropout(0.3) → Softmax(3)
- Optimizer: RMSprop (lr=0.001)
- Under the **150,000** parameter limit (~110,755 params)
- Official file: `results/section5_initial_model.keras`
- **Test accuracy ≈ 72.62%** (assignment asks for ≥ 60%)

Train it with:

```bash
python3 train_initial.py
```

### Section 6 — Optimized network

This is the model we actually use for the game / submission.

What we changed / tried along the way:
- Light data augmentation (horizontal flip, small rotation, contrast)
- Adam instead of plain RMSprop, slightly tuned LR
- A bit more capacity in the conv / dense layers, but still **under 150k** (final: **149,683** params)
- Dropout to reduce overfitting
- EarlyStopping + ModelCheckpoint + ReduceLROnPlateau
- Seeds everywhere (`SEED = 42`) so reruns are closer to the same starting point (same idea from an ML class Divya took before)

We did multiple training attempts (25 epochs, 30, 35, a BatchNorm experiment that didn’t pan out, etc.). Those leftovers are renamed under `results/attempts/` as `attempt_1_…`, `attempt_2_…`, etc., so they don’t get confused with the real submission files.

**Official final model**
- File: `results/section6_final_model.keras`
- **Test accuracy ≈ 0.7775 (77.75%)**
- Val accuracy peaked around ~0.786 during training
- Example evaluate output on Divya’s machine:

```text
{'accuracy': 0.7754038572311401, 'loss': 0.5710193514823914}
Confusion Matrix for basic_model
[[1303  321  150]
 [ 147  959  127]
 [  45   72  714]]
```

(That ~77.5% evaluate pass is the same family of weights; the locked metrics JSON for the committed final model is **0.7775**.)

**Important:** if you retrain with `train.py`, TensorFlow can give a slightly different number even with seeds (different OS / TF version). To check the **same** accuracy as the submission without overwriting weights, run:

```bash
python3 evaluate_model.py
```

That loads `results/section6_final_model.keras` and only evaluates.

### Section 7 — Webcam tic-tac-toe

- `player.py` → `UserWebcamPlayer` loads the final model
- Webcam frame → center crop → grayscale → resize 150×150 → stack to RGB → predict
- Keras folder order is alphabetical (`happy`, `neutral`, `surprise`), but the assignment wants `neutral=0`, `happy=1`, `surprise=2`, so we remap `{0:1, 1:0, 2:2}`
- Live webcam accuracy is a bit worse than the test set (lighting / framing / expression strength), so the built-in text override (`text` then `0/1/2`) is useful when a cell is already taken or the face read is wrong
- We ran a full game with Player X = random bot, Player O = webcam; O won on the main diagonal (trace is in the section 7 PDF)

Play:

```bash
python3 run.py
```

---

## Yuhui’s part (section 8) — transfer + random (summary)

Yuhui owns the details / PDF numbers; this is the high-level version from our chat and the code in the repo.

- Loads a trained facial base model, freezes the backbone for transfer, keeps the `flatten` layer, then adds a new dense head for dog vs cat
- Random model uses the same architecture idea but randomizes weights (control)
- When Yuhui tested: transfer had higher accuracy for the first few epochs; later random could pull ahead overall. Transfer still showed the early validation accuracy the assignment wants after epoch 1
- Yuhui recommended Divya also verify transfer on their machine, because the base model accuracy can differ by ~10% depending who trains it
- Yuhui later changed epochs / dense size when a run didn’t meet requirements, and shared `dog vs cat transfer model report.pdf`

Train transfer side:

```bash
python3 train_transfer.py
```

(Yuhui’s scripts expect the dog-vs-cat data laid out the way `preprocess.py` / `train_transfer.py` describe.)

---

## Official files vs old attempts (so nobody gets confused)

We had a bunch of files named `best_*` / `initial_*` from tuning. That got messy. Now it’s just:

| What | File | Approx test acc |
|------|------|-----------------|
| Section 5 initial | `results/section5_initial_model.keras` | ~72.6% |
| Section 6/7 final | `results/section6_final_model.keras` | **77.75%** |

Everything else from older runs lives in `results/attempts/` (`attempt_1_25epochs`, `attempt_2_30epochs`, `attempt_3_35epochs`, `attempt_failed_bn_…`, etc.). See `results/attempts/README.txt`.

Matching history plots / metrics / summaries are next to the official `.keras` files (same name prefix).

---

## Reports (PDFs)

- `reports/P6_sections_5_6_basic_model_report.pdf` — initial + optimized facial models
- `reports/P6_section_7_tic_tac_toe_report.pdf` — webcam game writeup + real move trace
- Yuhui’s dog-vs-cat transfer / random PDF (shared in Discord; add to the final zip / Canvas upload if it’s not already in the repo)

Regenerate Divya’s PDFs from the saved metrics (no retrain needed):

```bash
python3 generate_reports.py
```

---

## Setup (quick)

```bash
# Python 3 + a modern TensorFlow 2.x (course mentioned 2.12; newer 2.x is fine on current Mac/Python)
pip install -r requirements.txt

# Put FER-2013 into train/ and test/ (neutral, happy, surprise folders)
# Put dog-vs-cat data where Yuhui’s transfer preprocess expects it
```

Useful commands:

```bash
python3 train_initial.py      # section 5 baseline
python3 train.py              # section 6 optimized (overwrites final model — careful)
python3 evaluate_model.py     # score the saved final model without retraining
python3 run.py                # tic-tac-toe with webcam
python3 train_transfer.py     # Yuhui’s transfer / random training
python3 generate_reports.py   # rebuild Divya’s section PDFs
```

On Mac we use `python3`. If Rescaling / augmentation imports fail, the code already tries both the older `experimental.preprocessing` path and the newer Keras path.

---

## What to hand in / remember

- Final facial model is **under 150k params** and **above 60%** test accuracy (we’re at **~77.75%**)
- Initial network is also saved and reported separately for section 5
- Webcam game works end-to-end with emotion → board move mapping
- Seeds are in the train scripts so runs are more repeatable
- Don’t retrain right before submitting if you want the locked **77.75%** number — just `evaluate_model.py` and the committed `.keras` file
- Yuhui will fill in / attach more for the transfer + random writeup on their side

That’s basically everything we did for P6. Thanks Yuhui for the transfer side and for catching the PR pull thing early 🙂
