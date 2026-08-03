import numpy as np
from preprocess import get_datasets
from models.basic_model import BasicModel
from models.model import Model
from config import image_size, basic_model_path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import os
import json
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

input_shape = (image_size[0], image_size[1], 3)
categories_count = 3

models = {
    'basic_model': BasicModel,
}

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
        print("* Saved training plot to {}".format(save_path))
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    # if you want to load your model later, you can use:
    # model = Model.load_model("name_of_your_model.keras")
    # to load your history and plot it again, you can use:
    # history = np.load('results/name_of_your_model.npy',allow_pickle='TRUE').item()
    # plot_history(history)
    #
    # Hyperparameter-tuned run: more epochs + early stopping on val accuracy.
    epochs = 25
    os.makedirs("results", exist_ok=True)
    print('* Data preprocessing')
    train_dataset, validation_dataset, test_dataset = get_datasets()
    name = 'basic_model'
    model_class = models[name]
    print('* Training {} for up to {} epochs'.format(name, epochs))
    model = model_class(input_shape, categories_count)
    model.print_summary()

    checkpoint_path = "results/best_checkpoint.keras"
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            mode="max",
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
    ]

    history = model.model.fit(
        x=train_dataset,
        epochs=epochs,
        verbose="auto",
        validation_data=validation_dataset,
        callbacks=callbacks,
    )

    print('* Evaluating {}'.format(name))
    test_metrics = model.model.evaluate(test_dataset, return_dict=True)
    print(test_metrics)
    print('* Confusion Matrix for {}'.format(name))
    print(model.get_confusion_matrix(test_dataset))

    model_name = '{}_{}_epochs_timestamp_{}'.format(name, len(history.history['accuracy']), int(time.time()))
    filename = 'results/{}.keras'.format(model_name)
    model.save_model(filename)
    model.save_model(basic_model_path)
    np.save('results/{}.npy'.format(model_name), history.history)
    np.save('results/best_basic_model.npy', history.history)
    plot_history(history, save_path="results/basic_model_history.png")

    metrics = {
        "model_file": basic_model_path,
        "timestamped_model_file": filename,
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
    print('* Canonical model saved as {}'.format(basic_model_path))
    print('* Test accuracy: {:.4f}'.format(metrics["test_metrics"].get("accuracy", 0.0)))
