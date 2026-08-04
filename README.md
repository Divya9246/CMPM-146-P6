# CMPM 146 — Project 6

**Team**
- **Divya Machiraju** (`Divya9246`) — Sections 5–7: facial CNN, optimization, evaluation, webcam tic-tac-toe
- **Yuhui Zhen / YHZ** (`yzhen174`) — Section 8: dog-vs-cat transfer learning and random comparison model

---

## How we split the work

We planned the project over Discord and stuck to a clear split.

1. **Kickoff (late July)**  
   Divya shared the P6 plan / writeup. Yuhui agreed it looked good, shared GitHub (`yzhen174`), and Divya sent a repo invite.

2. **Datasets**  
   - Divya: [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) for facial emotions (`neutral`, `happy`, `surprise`).  
   - Yuhui: [Dog vs Cat Images Data](https://www.kaggle.com/datasets/kunalgupta2616/dog-vs-cat-images-data) for transfer learning.

3. **Divya — Sections 5–7 first**  
   Finished the facial CNN, training, and webcam tic-tac-toe integration. Early working version was about **74.7%** test accuracy (above the ≥60% requirement, under the 150k param limit), and the full webcam game worked. Pushed via PR; Yuhui pulled once the PR was merged.

4. **Yuhui — Section 8**  
   Built the transfer model and random model on dog vs cat. On Yuhui’s runs: transfer had higher accuracy for the first few epochs; later random could pull ahead overall. Transfer still had the early validation accuracy after epoch 1 that the assignment cares about. Yuhui also noted that retraining the basic model on their machine gave about **64%**, roughly **10%** below Divya’s facial model — important because transfer/random load that base model, so results can differ by machine. Divya was asked to verify transfer on their side as well.

5. **More tuning**  
   - Divya kept improving the facial model, added **seeds** so runs stay more consistent across machines, and locked the final shared weights at **~77.75%** test accuracy.  
   - Yuhui later adjusted transfer/random (epochs → 9, dense → 18 when an earlier run didn’t meet requirements) and shared the dog-vs-cat transfer PDF report.

6. **Who owns what for hand-in**  
   - Divya: sections 5–7 writeup (architecture, metrics, real webcam game trace), official facial models, evaluate/webcam code.  
   - Yuhui: section 8 writeup / PDF and transfer + random model details.

---

## Official models

| What | File | Test accuracy | Params |
|------|------|---------------|--------|
| Section 5 initial | `results/section5_initial_model.keras` | **~72.62%** | 110,755 |
| Section 6/7 final | `results/section6_final_model.keras` | **~77.75%** | 149,683 |

Older tuning runs are under `results/attempts/` (`attempt_1_…`, etc.) and are **not** submission models.

Locked final accuracy: **77.75%** (`0.777488…` in `results/section6_final_model_metrics.json`). Do not retrain before submit if you want that number. Check with:

```bash
python3 evaluate_model.py
```

---

## Section 5 — Initial network

No augmentation.

- Rescaling → Conv(16)-Pool → Conv(32)-Pool → Conv(64)-Pool → Conv(64)-Pool → Flatten → Dense(16) → Dropout(0.3) → Softmax(3)
- Optimizer: RMSprop (lr=0.001)
- **110,755** params (under 150,000)
- Best val accuracy ≈ **0.704**
- **Test accuracy ≈ 72.62%** (requirement ≥ 60%)

```bash
python3 train_initial.py
```

---

## Section 6 — Optimized network

Model used for evaluation and the webcam game.

- Augmentation: horizontal flip, small rotation, contrast
- Conv(24)-Pool → Conv(32)-Pool → Conv(48)-Pool → Conv(64)-Pool → Dropout(0.25)
- Flatten `"flatten"` (for transfer compatibility) → Dense(32) → Dropout(0.4) → Softmax(3)
- Adam (lr=0.0009), seed **42**, EarlyStopping + ModelCheckpoint + ReduceLROnPlateau
- **149,683** params (under 150,000)
- Best val accuracy ≈ **0.786**
- **Test accuracy ≈ 77.75%**

```bash
python3 train.py              # overwrites final model — careful
python3 evaluate_model.py     # score saved model only
```

---

## Section 7 — Webcam tic-tac-toe

- `UserWebcamPlayer` loads `results/section6_final_model.keras`
- Crop → grayscale → resize 150×150 → RGB → predict
- Remap Keras alphabetical labels to assignment order: `{0:1, 1:0, 2:2}` (neutral=0, happy=1, surprise=2)
- Text override (`text` then `0`/`1`/`2`) if a cell is taken or the face read is wrong
- Test-set accuracy **~77.75%**; live webcam is lower (lighting / framing / expression)

### Real game (X = random, O = webcam) — X won top row

From an actual `python3 run.py` session:

```text
1. X → (0,1)
   O tried neutral+happy → (0,1) already taken
2. O surprise+happy → (2,1)
3. X → (0,0)
   O tried neutral+happy → (0,1) already taken
   O tried surprise+happy → (2,1) already taken
4. O happy+happy → (1,1)
5. X → (1,0)
6. O surprise+neutral → (2,0)
7. X → (0,2)

Final:
|X|X|X|
|X|O| |
|O|O| |

Player X won across the top row (0,0)-(0,1)-(0,2).
```

Occupied cells and some wrong emotion reads were handled with the text override so the game could finish.

```bash
python3 run.py
```

---

## Section 8 — Yuhui (summary)

Transfer: load facial base, freeze backbone, keep `flatten`, new dense head for dog vs cat.  
Random: same structure with randomized weights (control).  
Details and numbers are in Yuhui’s dog-vs-cat PDF.

```bash
python3 train_transfer.py
```

---

## Setup

Assignment used TensorFlow **2.12**. Compatibility imports also support newer TF/Keras we tested; not every version is guaranteed.

```bash
pip install -r requirements.txt
python3 train_initial.py
python3 evaluate_model.py
python3 run.py
```
