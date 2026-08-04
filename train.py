import os
import random
import time
import json

# ---- Reproducibility (seed everyone to the same starting point) ----
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

from preprocess import get_datasets
from models.basic_model import BasicModel
from models.model import Model
from config import image_size
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

input_shape = (image_size[0], image_size[1], 3)
categories_count = 3
BEST_MODEL_PATH = "results/best_basic_model.keras"

models = {
    'basic_model': BasicModel,
}

def _make_deterministic(dataset):
    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.with_options(options)

def plot_history(history, save_path=None):
    hist = history.history if hasattr(history, "history") else history
    acc = hist['accuracy']
    val_acc = hist['val_accuracy']
    loss = hist['loss']
    val_loss = hist['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize = (24, 6))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'b', label = 'Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.title('Accuracy')

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'b', label = 'Training Loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.title('Loss')

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print('* Saved training plot to {}'.format(save_path))
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    # Hyperparameter-tuned training for sections 5-6 (facial recognition).
    epochs = 30
    os.makedirs("results", exist_ok=True)
    print('* Using random seed {}'.format(SEED))
    print('* Data preprocessing')
    train_dataset, validation_dataset, test_dataset = get_datasets()
    train_dataset = _make_deterministic(train_dataset)
    validation_dataset = _make_deterministic(validation_dataset)
    test_dataset = _make_deterministic(test_dataset)

    name = 'basic_model'
    model_class = models[name]
    print('* Training {} for up to {} epochs'.format(name, epochs))
    model = model_class(input_shape, categories_count)
    model.print_summary()

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            mode="max",
        ),
        ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    history = model.model.fit(
        x=train_dataset,
        epochs=epochs,
        verbose="auto",
        validation_data=validation_dataset,
        callbacks=callbacks,
        class_weight={0: 1.0, 1: 1.15, 2: 1.25},  # happy, neutral, surprise (alpha order)
    )

    print('* Evaluating {}'.format(name))
    test_metrics = model.model.evaluate(test_dataset, return_dict=True)
    print(test_metrics)
    print('* Confusion Matrix for {}'.format(name))
    print(model.get_confusion_matrix(test_dataset))

    model_name = '{}_{}_epochs_timestamp_{}'.format(
        name, len(history.history["accuracy"]), int(time.time())
    )
    filename = 'results/{}.keras'.format(model_name)
    model.save_model(filename)
    # Keep a stable filename for the webcam game controller.
    model.save_model(BEST_MODEL_PATH)
    np.save('results/{}.npy'.format(model_name), history.history)
    np.save('results/best_basic_model.npy', history.history)
    plot_history(history, save_path="results/basic_model_history.png")

    metrics = {
        "model_file": BEST_MODEL_PATH,
        "timestamped_model_file": filename,
        "seed": SEED,
        "epochs_trained": len(history.history["accuracy"]),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "total_params": int(model.model.count_params()),
    }
    with open("results/basic_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print('* Model saved as {}'.format(filename))
    print('* Best model saved as {}'.format(BEST_MODEL_PATH))
    print('* Test accuracy: {:.4f}'.format(metrics["test_metrics"].get("accuracy", 0.0)))
    print('* For the same accuracy on every device, share results/best_basic_model.keras')
    print('*   and run: python3 evaluate_model.py  (do not retrain)')
