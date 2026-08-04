from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam

try:
    from tensorflow.keras.layers.experimental.preprocessing import (
        Rescaling, RandomFlip, RandomRotation, RandomContrast,
    )
except ImportError:
    from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation, RandomContrast


class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Facial recognition CNN (neutral / happy / surprise), kept under 150k params.
        # Keep Flatten named "flatten" for transfer-learning compatibility.
        self.model = Sequential([
            Rescaling(1. / 255, input_shape=input_shape),
            RandomFlip("horizontal"),
            RandomRotation(0.04),
            RandomContrast(0.08),

            layers.Conv2D(24, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(48, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            layers.Flatten(name="flatten"),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(categories_count, activation="softmax"),
        ])

    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=0.0009),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
