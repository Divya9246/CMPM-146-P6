from models.model import Model
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import RMSprop, Adam
from config import basic_model_path
import tensorflow as tf


def _backbone_from_basic(basemodel, input_shape):
    """Build a Keras model that ends at the Flatten layer of the basic model.

    Works on both TF 2.12 (course) and newer Keras 3 by reusing layer objects
    instead of relying on Sequential.input before the model has been called.
    """
    inp = layers.Input(shape=input_shape)
    x = inp
    for layer in basemodel.layers:
        x = layer(x)
        if isinstance(layer, layers.Flatten) or layer.name == "flatten":
            break
    return inp, x


class TransferedModel(Model):
    def _define_model(self, input_shape, categories_count):
        # load your basic model with keras's load_model function
        # freeze the weights of the loaded model to make sure the training doesn't affect them
        # use this model by removing the last layer, adding dense layers and an output layer
        basemodel = models.load_model(basic_model_path)
        for layer in basemodel.layers:
            layer.trainable = False

        inp, x = _backbone_from_basic(basemodel, input_shape)
        x = layers.Dense(18, activation="relu", name="trans_dense")(x)
        x = layers.Dropout(0.3, name="trans_dropout")(x)
        output = layers.Dense(categories_count, activation="softmax", name="transfer_output")(x)

        self.model = models.Model(inputs=inp, outputs=output)
    
    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
