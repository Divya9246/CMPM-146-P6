from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import RMSprop, Adam

try:
    from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomFlip, RandomRotation
except ImportError:
    from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation


class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Hyperparameter-tuned CNN for FER2013 (neutral / happy / surprise).
        # Stays under the 150,000-parameter limit while targeting >=70% test accuracy.
        self.model = Sequential([
            Rescaling(1. / 255, input_shape=input_shape),
            RandomFlip("horizontal"),
            RandomRotation(0.05),

            layers.Conv2D(24, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(48, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(categories_count, activation="softmax"),
        ])

    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.0008),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
