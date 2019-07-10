import unittest

from ewc_app.service.keras_service.models.keras_mnist_model import KerasMnistModel
from ewc_app.service.keras_service.models.loss_model import LossModel
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
from copy import deepcopy


class TestModel(unittest.TestCase):

    def setUp(self):
        self.model = KerasMnistModel()
        self.loss = LossModel()
        self.data = self._create_data_set()
        self.permutated_data = self._permute_data(self.data)

    def _create_data_set(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        return mnist

    def _create_permutated_data(self):
        return self._permute_data(self.data)

    def _permute_data(self, mnist):
        perm_inds = list(range(mnist.train.images.shape[1]))
        np.random.shuffle(perm_inds)
        mnist2 = deepcopy(mnist)
        sets = ["train", "validation", "test"]
        for set_name in sets:
            this_set = getattr(mnist2, set_name)
            this_set._images = np.transpose(np.array([this_set.images[:, c] for c in perm_inds]))
        return mnist2


class TestInit(TestModel):

    def test_model_creation(self):
        self.assertIsNotNone(self.model)

    def test_data_generation(self):
        self.assertIsNotNone(self.data)
        self.assertEqual(3,len(self.data))

    def test_loss_generation(self):
        self.assertIsNotNone(self.loss)

    def test_simple_model_fit(self):

        self.model.compile(optimizer="adam", loss=self.loss.standard_loss(), metrics=["accuracy"])
        self.model.fit(self.data.train.images, self.data.train.labels, batch_size=64, epochs=10,
                       validation_data=(self.data.validation.images, self.data.validation.labels), shuffle=True, verbose=0)
        _, accuracy = self.model.evaluate(self.data.test.images, self.data.test.labels)
        self.assertGreater(accuracy, 0.9)
        return self.model

    def test_simple_model_fit_permutated_standard_loss(self):

        self.model = self.test_simple_model_fit()
        _, accuracy = self.model.evaluate(self.data.test.images, self.data.test.labels)
        self.assertGreater(accuracy, 0.9)

        self.model.compile(optimizer="adam", loss=self.loss.standard_loss(), metrics=["accuracy"])
        self.model.fit(self.permutated_data.train.images, self.permutated_data.train.labels, batch_size=64, epochs=10,
                       validation_data=(self.permutated_data.validation.images, self.permutated_data.validation.labels),
                       shuffle=True, verbose=0)
        _, accuracy = self.model.evaluate(self.data.test.images, self.data.test.labels)
        self.assertLess(accuracy, 0.85)

    def test_simple_model_fit_permutated_ewc_loss(self):

        g = tf.Graph()
        with g.as_default():
            with tf.Session() as sess:

                self.model = KerasMnistModel()
                self.model.compile(optimizer="adam", loss=self.loss.standard_loss(), metrics=["accuracy"])
                self.model.fit(self.data.train.images, self.data.train.labels, batch_size=64,
                               epochs=10,
                               validation_data=(self.data.validation.images, self.data.validation.labels),
                               shuffle=True, verbose=0)

                _, accuracy = self.model.evaluate(self.data.test.images, self.data.test.labels)
                self.assertGreater(accuracy, 0.9, msg="initial training")

                optimal_weights = [weight.eval() for weight in self.model.trainable_weights()]
                labels_ph = tf.placeholder(tf.int32, shape=(None, 10))
                fisher_tf = self.model.compute_fisher(sess, self.data.train.images,
                                           self.data.test.labels, labels_ph, sample_size=100)

                fisher = [layer.eval() for layer in fisher_tf]

                loss = self.loss.ewc_loss(self.model, optimal_weights, lam=[1,15], fisher=fisher)
                self.model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
                self.model.fit(self.permutated_data.train.images, self.permutated_data.train.labels, batch_size=64, epochs=10,
                               validation_data=(self.permutated_data.validation.images, self.permutated_data.validation.labels),
                               shuffle=True, verbose=0)
                _, accuracy = self.model.evaluate(self.permutated_data.test.images, self.permutated_data.test.labels)
                _, accuracy_2 = self.model.evaluate(self.data.test.images, self.data.test.labels)

                self.assertGreater(accuracy, 0.90, msg="permutated data")
                self.assertGreater(accuracy_2, 0.90, msg="initial data")
