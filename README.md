## CMPM 146 P6

Team:
Divya Machiraju: Sections 5,6,7 facial CNN, optimization, evaluation, webcam tic-tac-toe
Yuhui Zhen: Section 8 dog vs cat transfer learning and random comparison model

## How we split the work

We planned the project over Discord and stuck to a clear split.

Datasets used: 
Divya: [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) for facial emotions (`neutral`, `happy`, `surprise`).  
Yuhui: [Dog vs Cat Images Data](https://www.kaggle.com/datasets/kunalgupta2616/dog-vs-cat-images-data) for transfer learning.

Divya: Sections 5,6,7   
Finished the facial CNN, training, and webcam tic-tac-toe integration. Early working version was about 74.7% test accuracy which was above the greater than the 60% requirement, under the 150k param limit, and the full webcam game worked.

Yuhui: Section 8 
Built the transfer model and random model on dog vs cat. On Yuhui’s runs transfer had higher accuracy for the first few epochs, later random could pull ahead overall. Transfer still had the early validation accuracy after epoch 1 that the assignment cares about. Yuhui also observed that retraining the basic model on their machine gave about 64%, roughly 10% below Divya’s facial model which was important because transfer/random load that base model, so results can differ by machine.

Divya kept improving the facial model, added seeds so runs stay more consistent across machines, and the final shared weights was 77.75% test accuracy.  
Yuhui later adjusted transfer/random epochs to 9, dense to 18 when an earlier run didn’t meet requirements.

Divya: sections 5,6,7 writeup, architecture, metrics, real webcam game trace, official facial models, evaluate/webcam code.  
Yuhui: section 8 writeup, PDF and transfer + random model details.


## Models

Section 5 initial is in file `results/section5_initial_model.keras` with test accuracy 72.62% and params 110,755
Section 6/7 final is in file `results/section6_final_model.keras` with test accuracy 77.75% and params 149,683

Older tuning runs are under `results/attempts/` (`attempt_1_…`, etc.) and are not submission models, they re just included for reference.

77.75% is in in `results/section6_final_model_metrics.json`. 

You can check with python3 evaluate_model.py


## Section 5 - Initial network

No augmentation.
Rescaling → Conv(16)-Pool → Conv(32)-Pool → Conv(64)-Pool → Conv(64)-Pool → Flatten → Dense(16) → Dropout(0.3) → Softmax(3)
Optimizer: RMSprop (lr=0.001)
110,755 params (under 150,000)
Best val accuracy 0.704
Test accuracy 72.62% (requirement greater than 60%)

python3 train_initial.py

## Section 6 - Optimized network

Model used for evaluation and the webcam game.

Augmentation: horizontal flip, small rotation, contrast
Conv(24)-Pool → Conv(32)-Pool → Conv(48)-Pool → Conv(64)-Pool → Dropout(0.25)
Flatten `"flatten"` (for transfer compatibility) → Dense(32) → Dropout(0.4) → Softmax(3)
Adam (lr=0.0009), seed 42, EarlyStopping + ModelCheckpoint + ReduceLROnPlateau
149,683 params (under 150,000)
Best val accuracy 0.786
Test accuracy 77.75%

python3 train.py overwrites final model
python3 evaluate_model.py has the score saved model 

## Section 7 - Webcam tic-tac-toe

`UserWebcamPlayer` loads `results/section6_final_model.keras`
Crop → grayscale → resize 150×150 → RGB → predict
Remap Keras alphabetical labels to assignment order: `{0:1, 1:0, 2:2}` (neutral=0, happy=1, surprise=2)
Text override (`text` then `0`/`1`/`2`) if a cell is taken or the face read is wrong
Test set accuracy 77.75%; live webcam is lower (lighting / framing / expression)

### Real game (X = random, O = webcam) — X won top row

From an actual `python3 run.py` session:

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
Occupied cells and some wrong emotion reads were handled with the text override so the game could finish.

python3 run.py


## Section 8 — Yuhui

Transfer: load facial base, freeze backbone, keep `flatten`, new dense head for dog vs cat.  
Random: same structure with randomized weights (control).  
Details and numbers are in Yuhui’s dog-vs-cat PDF.

python3 train_transfer.py