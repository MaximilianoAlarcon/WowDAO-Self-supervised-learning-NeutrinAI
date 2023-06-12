# %% Imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


# %% Von Mises Fisher 3D Loss
class VonMisesFisher3DLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        switch_exact: float=100.0,
        eps: float=tf.keras.backend.epsilon(),
        name: str="vmf_3d_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.switch_exact = switch_exact
        self.switch_exact_tf = tf.constant(self.switch_exact, dtype=tf.float32)
        self.eps = eps

        self.offset = self.log_c3k_exact(self.switch_exact_tf)
        self.offset -= self.log_c3k_approx(self.switch_exact_tf)

    def tfp_log_iv(self, m, kappa):
        v = m / 2.0 - 1
        ive = tfp.math.bessel_ive(v, kappa)
        log_ive = tf.math.log(ive)
        log_iv = log_ive + kappa
        return log_iv

    def log_c3k_approx(self, kappa):
        """Calculate $log C_{3}(k)$ term in von Mises-Fisher loss approx."""
        # v = 1, b = 0 for C_3(k)
        a = tf.sqrt(4 + kappa**2)
        return -a

    def log_c3k_exact(self, kappa):
        m = 3.0
        return (
            (m / 2.0 - 1) * tf.math.log(kappa)
            - self.tfp_log_iv(m, kappa)
            - (m / 2) * np.math.log(2 * np.pi)
        )

    def log_c3k(self, kappa):
        return tf.where(
            kappa < self.switch_exact_tf,
            self.log_c3k_exact(kappa),
            self.log_c3k_approx(kappa) + self.offset,
        )

    def call(self, y_true, y_pred):
        # y_true is the azimuth and zenith of true label.
        # convert it into direction vector
        direction_true = tf.stack(
            [
                tf.cos(y_true[:, 0]) * tf.sin(y_true[:, 1]),
                tf.sin(y_true[:, 0]) * tf.sin(y_true[:, 1]),
                tf.cos(y_true[:, 1]),
            ],
            axis=1,
        )

        # y_pred is the vec_x, vec_y and vec_z prediction.
        # kappa is the vector norm
        kappa = tf.norm(y_pred, axis=1) + self.eps
        direction_pred = y_pred / tf.expand_dims(kappa, axis=1)

        # calculate the loss
        loss = -tf.reduce_mean(
            self.log_c3k(kappa)
            + kappa * tf.reduce_sum(direction_true * direction_pred, axis=1)
        )
        return loss
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'switch_exact': self.switch_exact,
            'eps': self.eps,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% Main function is for testing
if __name__ == "__main__":
    # start of main
    print("\n\nLosses.py is running as main\n")

    # Test VonMisesFisher3DLoss
    print("Test VonMisesFisher3DLoss")
    loss = VonMisesFisher3DLoss()
    y_true = tf.constant([[0.0, 0.0], [0.0, np.pi / 2.0]], dtype=tf.float32)
    y_pred = tf.constant([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=tf.float32)
    print("loss(y_true, y_pred) = ", loss(y_true, y_pred))

    # end of main
    print("\n\nLosses.py finished running as main\n")

# %% END-OF-FILE
