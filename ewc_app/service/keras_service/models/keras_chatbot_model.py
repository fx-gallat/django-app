
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Reshape, MaxPool2D, Conv2D, Dense, Concatenate, Flatten, Dropout, Embedding
from .keras_model import KerasModel
from keras import regularizers
from ewc_app.const.constants import MODEL_PATH
import os


class KerasChatBotModel(KerasModel):

    def __init__(self, embedding_chatbot, pretrained=False):
        super(KerasChatBotModel, self).__init__()

        if not pretrained:
            self.model, _ = self._create_network(embedding_chatbot)
        else:
            self.model = self.load_model(os.path.join(MODEL_PATH, "keras_models/ewc_cnn_model.h5"))

    def _create_network(self, embedding_matrix):

        MAX_SEQUENCE_LENGTH = 111
        EMBEDDING_DIM = 300
        FILTER_SIZES = [3, 4, 5]
        CONV_ACTIVATION = 'relu'
        NUM_FILTERS = 512
        DROPOUT_RATE = 0.5
        OUTPUT_ACTIVATION = 'softmax'
        REGUL_PARAM_L2 = 0.01
        REGUL_PARAM_L1 = 0.01
        OUTPUT_DIM = 59

        inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

        embedding_layer = Embedding(
            embedding_matrix.shape[0],
            embedding_matrix.shape[1],
            input_length=MAX_SEQUENCE_LENGTH,
            trainable=False,
            weights=[embedding_matrix],
            name='embedding')(inputs)

        reshape = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1),name="reshape")(embedding_layer)

        with K.name_scope('Conv_MP_0'):
            conv_0 = Conv2D(
                NUM_FILTERS,
                kernel_size=(FILTER_SIZES[0], EMBEDDING_DIM),
                padding='valid',
                kernel_initializer='normal',
                activation=CONV_ACTIVATION,
                trainable=True,
                name='conv_0')(reshape)

            maxpool_0 = MaxPool2D(
                pool_size=(MAX_SEQUENCE_LENGTH - FILTER_SIZES[0] + 1, 1),
                strides=(1, 1),
                padding='valid',
                name='maxpool_0')(conv_0)

        with K.name_scope('Conv_MP_1'):
            conv_1 = Conv2D(
                NUM_FILTERS,
                kernel_size=(FILTER_SIZES[1], EMBEDDING_DIM),
                padding='valid',
                kernel_initializer='normal',
                activation=CONV_ACTIVATION,
                trainable=True,
                name='conv_1')(reshape)

            maxpool_1 = MaxPool2D(
                pool_size=(MAX_SEQUENCE_LENGTH - FILTER_SIZES[1] + 1, 1),
                strides=(1, 1),
                padding='valid',
                name='maxpool_1')(conv_1)

        with K.name_scope('Conv_MP_2'):
            conv_2 = Conv2D(
                NUM_FILTERS,
                kernel_size=(FILTER_SIZES[2], EMBEDDING_DIM),
                padding='valid',
                trainable=True,
                kernel_initializer='normal',
                activation=CONV_ACTIVATION,
                name='conv_2')(reshape)

            maxpool_2 = MaxPool2D(
                pool_size=(MAX_SEQUENCE_LENGTH - FILTER_SIZES[2] + 1, 1),
                strides=(1, 1),
                padding='valid',
                name='maxpool_2')(conv_2)

        concatenated_tensor = Concatenate(axis=1, name="concatenate")([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten(name="flatten")(concatenated_tensor)
        dropout = Dropout(DROPOUT_RATE, name="dropout")(flatten)
        output = Dense(
            units=OUTPUT_DIM,
            activation=OUTPUT_ACTIVATION,
            kernel_regularizer=regularizers.l2(REGUL_PARAM_L2),
            activity_regularizer=regularizers.l1(REGUL_PARAM_L1),
            name="dense")(dropout)

        model = Model(inputs=inputs, outputs=output)

        return model
