"""Train the Section 5 initial facial recognition network (simpler baseline)."""
import os
import random
import json
import time

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

try:
    from tensorflow.keras.layers.experimental.preprocessing import Rescaling
except ImportError:
    from tensorflow.keras.layers import Rescaling

from preprocess import get_datasets
from config import image_size

INITIAL_MODEL_PATH = "results/initial_basic_model.keras"
SUMMARY_PATH = "results/initial_basic_model_summary.txt"


def plot_history(history, save_path):
    hist = history.history
    epochs = range(1, len(hist["accuracy"]) + 1)
    plt.figure(figsize=(24, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist["accuracy"], "b", label="Training Accuracy")
    plt.plot(epochs, hist["val_accuracy"], "r", label="Validation Accuracy")
    plt.grid(True)
    plt.legend()
    plt.xlabel("Epoch")
    plt.title("Initial Network — Accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist["loss"], "b", label="Training Loss")
    plt.plot(epochs, hist["val_loss"], "r", label="Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.xlabel("Epoch")
    plt.title("Initial Network — Loss")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    print("* Section 5: training INITIAL facial recognition network")
    train_ds, val_ds, test_ds = get_datasets()

    model = Sequential([
        Rescaling(1. / 255, input_shape=(image_size[0], image_size[1], 3)),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(name="flatten"),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    with open(SUMMARY_PATH, "w") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, mode="max"),
        ModelCheckpoint(INITIAL_MODEL_PATH, monitor="val_accuracy", save_best_only=True, mode="max"),
    ]
    history = model.fit(
        train_ds,
        epochs=15,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose="auto",
    )
    test_metrics = model.evaluate(test_ds, return_dict=True)
    print(test_metrics)
    model.save(INITIAL_MODEL_PATH)
    np.save("results/initial_basic_model.npy", history.history)
    plot_history(history, "results/initial_basic_model_history.png")

    metrics = {
        "label": "initial_network_section_5",
        "model_file": INITIAL_MODEL_PATH,
        "epochs_trained": len(history.history["accuracy"]),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "total_params": int(model.count_params()),
        "notes": "Simpler CNN: no augmentation, Dense(16), RMSprop(0.001). Target >= 60% test accuracy.",
    }
    with open("results/initial_basic_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("* Initial test accuracy: {:.4f}".format(metrics["test_metrics"]["accuracy"]))
    print("* Saved", INITIAL_MODEL_PATH)
