import tensorflow as tf


class LossModel:

    @staticmethod
    def standard_loss():
        def custom_loss(y_true, y_pred):
            loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
            return loss
        return custom_loss

    @staticmethod
    def ewc_loss(model, optimal_weights=None, lam=[1, 0], fisher=None):
        def custom_loss(y_true, y_pred):

            ewc_loss = tf.losses.softmax_cross_entropy(y_true, y_pred)

            if lam[1] != 0 and fisher is not None and optimal_weights is not None:

                for v in range(len(optimal_weights)):
                    ewc_loss += (lam[1] / 2) * tf.reduce_sum(
                        tf.multiply(fisher[v],
                                    tf.square(tf.subtract(model.trainable_weights[v], optimal_weights[v]))))
            return ewc_loss
        return custom_loss
