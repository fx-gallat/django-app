from keras.layers import Dense, Input, Dropout
from keras.models import Model
from .keras_model import KerasModel


class KerasMnistModel(KerasModel):

    def __init__(self, pretrained=False):
        super(KerasMnistModel, self).__init__()

        self.pretrained = pretrained

        if not self.pretrained:
            self.model, _ = self._create_network()
        else:
            self.model = self.load_model("../../../data/models/keras_models/ewc_mnist_model.h5")

    def _create_network(self):

        input_layer = Input(batch_shape=(None, 784), name='input')
        dense_0, w_in, b_in = self._create_dense_layer(input_layer, 128, activation="relu")
        dropout_0 = Dropout(0.5)(dense_0)
        dense_1, w_1, b_1 = self._create_dense_layer(dropout_0, 128, activation="relu")
        dropout_1 = Dropout(0.5)(dense_1)
        output_layer, w_out, b_out = self._create_dense_layer(dropout_1, 10, activation="linear")
        weights = [w_in, b_in, w_1, b_1, w_out, b_out]
        model = Model(inputs=input_layer, outputs=output_layer)

        return model, weights

    def _create_dense_layer(self,input_tensor, layer_size, activation):

        layer_1 = Dense(layer_size, activation=activation)
        y = layer_1(input_tensor)
        w_1, b_1 = layer_1.trainable_weights[0], layer_1.trainable_weights[1]
        return y, w_1, b_1
