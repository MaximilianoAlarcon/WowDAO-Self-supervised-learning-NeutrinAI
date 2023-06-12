# %% Imports
import numpy as np
import tensorflow as tf


# %% Define angular distance score
def angular_dist_score(az_true, zen_true, az_pred, zen_pred, return_sample=False):
    if not (
        np.all(np.isfinite(az_true))
        and np.all(np.isfinite(zen_true))
        and np.all(np.isfinite(az_pred))
        and np.all(np.isfinite(zen_pred))
    ):
        raise ValueError("All arguments must be finite")

    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)

    sa2 = np.sin(az_pred)
    ca2 = np.cos(az_pred)
    sz2 = np.sin(zen_pred)
    cz2 = np.cos(zen_pred)

    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)
    scalar_prod = np.clip(scalar_prod, -1, 1)

    if return_sample:
        return np.abs(np.arccos(scalar_prod))
    else:
        return np.average(np.abs(np.arccos(scalar_prod)))


# %% angular dist score in tensorflow
class AngularDistScore(tf.keras.metrics.Metric):
    def __init__(
        self, eps=tf.keras.backend.epsilon(), name="angular_dist_score", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.angular_distance_sum = self.add_weight(name="ads", initializer="zeros")
        self.number_of_events = self.add_weight(name="noe", initializer="zeros")
        self.eps = eps

    def update_state(self, y_true, y_pred, sample_weight=None):
        az_true = y_true[:, 0]
        zen_true = y_true[:, 1]

        vec_x_true = tf.cos(az_true) * tf.sin(zen_true)
        vec_y_true = tf.sin(az_true) * tf.sin(zen_true)
        vec_z_true = tf.cos(zen_true)

        kappa_pred = tf.norm(y_pred, axis=1) + self.eps
        vec_x_pred = y_pred[:, 0] / kappa_pred
        vec_y_pred = y_pred[:, 1] / kappa_pred
        vec_z_pred = y_pred[:, 2] / kappa_pred

        # dot product
        scalar_prod = vec_x_true * vec_x_pred
        scalar_prod += vec_y_true * vec_y_pred
        scalar_prod += vec_z_true * vec_z_pred
        scalar_prod = tf.clip_by_value(scalar_prod, -1, 1)

        angular_distance = tf.abs(tf.acos(scalar_prod))
        self.angular_distance_sum.assign_add(tf.reduce_sum(angular_distance))
        self.number_of_events.assign_add(
            tf.cast(tf.size(angular_distance), tf.float32)
        )  # NOTE: float16

    def result(self):
        res = self.angular_distance_sum / self.number_of_events
        if tf.math.is_nan(res):
            return tf.zeros_like(res)
        else:
            return res

    def reset_state(self):
        self.angular_distance_sum.assign(0)
        self.number_of_events.assign(0)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"eps": self.eps})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% Main function is for testing
if __name__ == "__main__":
    # start of main
    print("\n\nMetrics.py is running as main\n")

    # Test angular_dist_score
    print("Testing angular_dist_score")

    az_true = np.random.uniform(0, 2 * np.pi, 100)
    zen_true = np.random.uniform(0, np.pi, 100)

    az_pred = az_true + np.random.uniform(-0.1, 0.1, 100)
    az_pred = np.mod(az_pred, 2 * np.pi)
    zen_pred = zen_true + np.random.uniform(-0.1, 0.1, 100)
    zen_pred = np.clip(zen_pred, 0, np.pi)

    print("angular_dist_score(az_true, zen_true, az_pred, zen_pred):")
    print(angular_dist_score(az_true, zen_true, az_pred, zen_pred))

    # Test AngularDistScore
    print("\nTesting AngularDistScore")

    az_true = tf.random.uniform((100,), 0, 2 * np.pi)
    zen_true = tf.random.uniform((100,), 0, np.pi)

    az_pred = az_true + tf.random.uniform((100,), -0.1, 0.1)
    az_pred = tf.math.mod(az_pred, 2 * np.pi)
    zen_pred = zen_true + tf.random.uniform((100,), -0.1, 0.1)
    zen_pred = tf.clip_by_value(zen_pred, 0, np.pi)

    ads = AngularDistScore()
    ads.update_state(
        tf.stack([az_true, zen_true], axis=1),
        tf.stack(
            [
                tf.cos(az_pred) * tf.sin(zen_pred),
                tf.sin(az_pred) * tf.sin(zen_pred),
                tf.cos(zen_pred),
            ],
            axis=1,
        ),
    )
    print("AngularDistScore(az_true, zen_true, az_pred, zen_pred):")
    print(ads.result())

    # end of main
    print("\n\nMetrics.py finished running as main\n")

# %% END-OF-FILE
