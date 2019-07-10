from keras.models import load_model
from keras import backend as K
import numpy as np
import tensorflow as tf


class KerasModel:

    def __init__(self):
        self.model = None

    def _create_network(self):
        raise NotImplementedError

    def compile(self, optimizer, loss, metrics):
        return self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def compute_fisher(self, sess, features, labels, placeholder_label, sample_size=200):

        ind = np.random.choice(features.shape[0], sample_size)
        training_features = features[ind]
        training_labels = labels[ind]

        fisher = []
        for v in range(len(self.model.trainable_weights)):
            fisher.append(np.zeros(self.model.trainable_weights[v].get_shape().as_list()))

        log_likelihood = tf.nn.softmax_cross_entropy_with_logits_v2(placeholder_label, self.model.output)
        log_likelihood = tf.reduce_sum(log_likelihood)
        optimal_weights = self.model.trainable_weights
        gradients = K.gradients(-log_likelihood, optimal_weights)

        for sentence, label in zip(training_features, training_labels):
            der = sess.run(gradients, feed_dict={self.model.input: sentence.reshape((1, -1)),
                                                 placeholder_label: label.reshape((1, -1))})
            for v in range(len(fisher)):
                fisher[v] += tf.square(der[v])

        for v in range(len(fisher)):
            fisher[v] /= sample_size

        return fisher

    def fit(self, features, labels, batch_size=64, epochs=10, validation_data=None, shuffle=True, verbose=0):
        return self.model.fit(features, labels, batch_size=batch_size, epochs=epochs,
                              validation_data=validation_data, shuffle=shuffle, verbose=verbose)

    def predict(self, features):
        return self.model.predict(features)

    def evaluate(self, features, labels):
        return self.model.evaluate(features, labels, verbose=0)

    def trainable_weights(self):
        return self.model.trainable_weights

    def load_model(self, model_path):
        self.model = load_model(model_path)
        return self.model
