from models.model import Model
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import RMSprop, Adam
from config import basic_model_path
from models.transfered_model import _backbone_from_basic


class RandomModel(Model):
    def _define_model(self, input_shape, categories_count):
        # very similar to transfered_model.py, the only difference is that you should randomize the weights
        # load your basic model, randomize weights, keep them trainable, replace the head
        basemodel = models.load_model(basic_model_path)
        self._randomize_layers(basemodel)
        for layer in basemodel.layers:
            layer.trainable = True

        inp, x = _backbone_from_basic(basemodel, input_shape)
        x = layers.Dense(18, activation="relu", name="trans_dense")(x)
        x = layers.Dropout(0.3, name="trans_dropout")(x)
        output = layers.Dense(categories_count, activation="softmax", name="random_output")(x)

        self.model = models.Model(inputs=inp, outputs=output)

    
    def _compile_model(self):
       self.model.compile(
                   optimizer=RMSprop(learning_rate=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy']
               )

    @staticmethod
    def _randomize_layers(model):
       for layer in model.layers:
            if hasattr(layer, "kernel") and hasattr(layer, "kernel_initializer"):
                layer.kernel.assign(
                layer.kernel_initializer(
                    shape=layer.kernel.shape,
                    dtype=layer.kernel.dtype
                )
            )

            if hasattr(layer, "bias") and layer.bias is not None:
                layer.bias.assign(
                layer.bias_initializer(
                    shape=layer.bias.shape,
                    dtype=layer.bias.dtype
                )
            )
            layer.trainable = True
