from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

try:
    from tensorflow.keras.layers.experimental.preprocessing import (
        Rescaling, RandomFlip, RandomRotation, RandomContrast,
    )
except ImportError:
    from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation, RandomContrast


class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Facial recognition CNN (neutral / happy / surprise), kept under 150k params.
        # Keep a Flatten layer named "flatten" for the transfer-learning code path.
        self.model = Sequential([
            Rescaling(1. / 255, input_shape=input_shape),
            RandomFlip("horizontal"),
            RandomRotation(0.05),
            RandomContrast(0.1),

            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(40, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(48, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(56, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.3),

            layers.Flatten(name="flatten"),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.45),
            layers.Dense(categories_count, activation="softmax"),
        ])

    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossentropy(label_smoothing=0.05),
            metrics=["accuracy"],
        )
