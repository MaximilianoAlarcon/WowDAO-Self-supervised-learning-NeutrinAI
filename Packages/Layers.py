# %% Import
from typing import List

import tensorflow as tf

try:
    from . import Utils as _utils
except:
    import Utils as _utils


# %% H-Layer: low level shape layer. (B, N, F) -> (B, N, K, H(=F_in))
class HLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        h_list: List[str] = [
            "t",
            "x",
            "y",
            "z",
            "c",
            "a",
            "dt",
            "dx",
            "dy",
            "dz",
            "euclidean",
            "lorentz",
        ],
        K: int = 9,
        **kwargs
    ):
        """H-Layer which generates low-level geometric features from raw inputs.
        (B, N, F) -> (B, N, K, H)

        Args:
                h_list (List[str]): list of geometric features to generate
                K (int): number of nearest neighbors to use. (K-1) / 2 neighbors on each side
        """
        super(HLayer, self).__init__(**kwargs)
        assert len(h_list) > 0, "h_list must not be empty"
        assert K % 2 == 1, "K must be odd"

        self.h_list = h_list
        self.K = K
        self.k_self = (K - 1) // 2

        self.h_to_func = {
            "t": self.gf_t,
            "x": self.gf_x,
            "y": self.gf_y,
            "z": self.gf_z,
            "c": self.gf_c,
            "a": self.gf_a,
            "deep": self.gf_deep,
            "dt": self.gf_dt,
            "dx": self.gf_dx,
            "dy": self.gf_dy,
            "dz": self.gf_dz,
            "euclidean": self.gf_euclidean,
            "lorentz": self.gf_lorentz,
        }

        assert all(
            [h in self.h_to_func.keys() for h in self.h_list]
        ), "h_list contains invalid geometric features"

        # speed of light, but time in us and position in 1./600. m
        self.speed_of_light = 299792458.0  # m/s
        self.speed_of_light /= 1e6  # m/us
        self.speed_of_light /= 600.0  # 1./600. m/us
        self.speed_of_light = tf.constant(self.speed_of_light, dtype=tf.float32)

    def call(self, inputs):
        #  inputs: (B, N, F)
        #       t: inputs[:, :, 0]
        #  charge: inputs[:, :, 1]
        #     aux: inputs[:, :, 2]
        #       x: inputs[:, :, 3]
        #       y: inputs[:, :, 4]
        #       z: inputs[:, :, 5]
        # outputs: (B, N, K, H)
        #   where K = the # of NNs
        #         H = len(h_list)

        # initialize outputs
        self.outputs = list()

        for h in self.h_list:
            self.outputs.append(self.h_to_func[h](inputs))

        # concatenate outputs
        self.outputs = tf.stack(self.outputs, axis=3)  # (B, N, K, H)

        return self.outputs

    # geometric feature functions
    # time
    def gf_t(self, inputs):
        #    inputs: (B, N, F)
        #   outputs: (B, N, K)
        # [:, :, k]: k-th neighbor's time

        # get time and initialize output `time_k`
        time = inputs[:, :, 0]  # (B, N)
        time_k = list()
        for k in range(self.K):
            shift = k - self.k_self
            if shift < 0:
                time_k.append(tf.pad(time[:, :shift], [[0, 0], [-shift, 0]]))
            elif shift == 0:
                time_k.append(time)
            else:  # shift > 0
                time_k.append(tf.pad(time[:, shift:], [[0, 0], [0, shift]]))

        # concatenate time_k
        time_k = tf.stack(time_k, axis=2)  # (B, N, K)

        return time_k

    # position x
    def gf_x(self, inputs):
        #    inputs: (B, N, F)
        #   outputs: (B, N, K)
        # [:, :, k]: k-th neighbor's x

        # get x and initialize output `x_k`
        x = inputs[:, :, 3]  # (B, N)
        x_k = list()
        for k in range(self.K):
            shift = k - self.k_self
            if shift < 0:
                x_k.append(tf.pad(x[:, :shift], [[0, 0], [-shift, 0]]))
            elif shift == 0:
                x_k.append(x)
            else:  # shift > 0
                x_k.append(tf.pad(x[:, shift:], [[0, 0], [0, shift]]))

        # concatenate x_k
        x_k = tf.stack(x_k, axis=2)  # (B, N, K)

        return x_k

    # position y
    def gf_y(self, inputs):
        #    inputs: (B, N, F)
        #   outputs: (B, N, K)
        # [:, :, k]: k-th neighbor's y

        # get y and initialize output `y_k`
        y = inputs[:, :, 4]  # (B, N)
        y_k = list()
        for k in range(self.K):
            shift = k - self.k_self
            if shift < 0:
                y_k.append(tf.pad(y[:, :shift], [[0, 0], [-shift, 0]]))
            elif shift == 0:
                y_k.append(y)
            else:  # shift > 0
                y_k.append(tf.pad(y[:, shift:], [[0, 0], [0, shift]]))

        # concatenate y_k
        y_k = tf.stack(y_k, axis=2)  # (B, N, K)

        return y_k

    # position z
    def gf_z(self, inputs):
        #    inputs: (B, N, F)
        #   outputs: (B, N, K)
        # [:, :, k]: k-th neighbor's z

        # get z and initialize output `z_k`
        z = inputs[:, :, 5]  # (B, N)
        z_k = list()
        for k in range(self.K):
            shift = k - self.k_self
            if shift < 0:
                z_k.append(tf.pad(z[:, :shift], [[0, 0], [-shift, 0]]))
            elif shift == 0:
                z_k.append(z)
            else:  # shift > 0
                z_k.append(tf.pad(z[:, shift:], [[0, 0], [0, shift]]))

        # concatenate z_k
        z_k = tf.stack(z_k, axis=2)  # (B, N, K)

        return z_k

    # charge c
    def gf_c(self, inputs):
        #    inputs: (B, N, F)
        #   outputs: (B, N, K)
        # [:, :, k]: k-th neighbor's charge

        # get charge and initialize output `charge_k`
        charge = inputs[:, :, 1]  # (B, N)
        charge_k = list()
        for k in range(self.K):
            shift = k - self.k_self
            if shift < 0:
                charge_k.append(tf.pad(charge[:, :shift], [[0, 0], [-shift, 0]]))
            elif shift == 0:
                charge_k.append(charge)
            else:  # shift > 0
                charge_k.append(tf.pad(charge[:, shift:], [[0, 0], [0, shift]]))

        # concatenate charge_k
        charge_k = tf.stack(charge_k, axis=2)  # (B, N, K)

        return charge_k

    # auxiliary a
    def gf_a(self, inputs):
        #    inputs: (B, N, F)
        #   outputs: (B, N, K)
        # [:, :, k]: k-th neighbor's auxiliary

        # get auxiliary and initialize output `auxiliary_k`
        auxiliary = inputs[:, :, 2]  # (B, N)
        auxiliary_k = list()
        for k in range(self.K):
            shift = k - self.k_self
            if shift < 0:
                auxiliary_k.append(tf.pad(auxiliary[:, :shift], [[0, 0], [-shift, 0]]))
            elif shift == 0:
                auxiliary_k.append(auxiliary)
            else:  # shift > 0
                auxiliary_k.append(tf.pad(auxiliary[:, shift:], [[0, 0], [0, shift]]))

        # concatenate auxiliary_k
        auxiliary_k = tf.stack(auxiliary_k, axis=2)  # (B, N, K)

        return auxiliary_k
    
    # is_deep deep
    def gf_deep(self, inputs):
        #    inputs: (B, N, F)
        #   outputs: (B, N, K)
        # [:, :, k]: k-th neighbor's deep

        # get deep and initialize output `deep_k`
        deep = inputs[:, :, 6]
        deep_k = list()
        for k in range(self.K):
            shift = k - self.k_self
            if shift < 0:
                deep_k.append(tf.pad(deep[:, :shift], [[0, 0], [-shift, 0]]))
            elif shift == 0:
                deep_k.append(deep)
            else:  # shift > 0
                deep_k.append(tf.pad(deep[:, shift:], [[0, 0], [0, shift]]))

        # concatenate deep_k
        deep_k = tf.stack(deep_k, axis=2)  # (B, N, K)

        return deep_k

    # dt: time difference
    def gf_dt(self, inputs):
        #    inputs: (B, N, F)
        #   outputs: (B, N, K)
        # [:, :, k]: k-th neighbor's time difference

        # get time and initialize output `dt_k`
        time = inputs[:, :, 0]  # (B, N)
        dt_k = list()
        for k in range(self.K):
            shift = k - self.k_self
            if shift < 0:
                dt_k.append(
                    tf.pad(time[:, :shift] - time[:, -shift:], [[0, 0], [-shift, 0]])
                )
            elif shift == 0:
                dt_k.append(tf.zeros_like(time))
            else:  # shift > 0
                dt_k.append(
                    tf.pad(time[:, shift:] - time[:, :-shift], [[0, 0], [0, shift]])
                )

        # concatenate dt_k
        dt_k = tf.stack(dt_k, axis=2)  # (B, N, K)

        return dt_k

    # dx: position x difference
    def gf_dx(self, inputs):
        #    inputs: (B, N, F)
        #   outputs: (B, N, K)
        # [:, :, k]: k-th neighbor's x difference

        # get x and initialize output `dx_k`
        x = inputs[:, :, 3]  # (B, N)
        dx_k = list()
        for k in range(self.K):
            shift = k - self.k_self
            if shift < 0:
                dx_k.append(tf.pad(x[:, :shift] - x[:, -shift:], [[0, 0], [-shift, 0]]))
            elif shift == 0:
                dx_k.append(tf.zeros_like(x))
            else:  # shift > 0
                dx_k.append(tf.pad(x[:, shift:] - x[:, :-shift], [[0, 0], [0, shift]]))

        # concatenate dx_k
        dx_k = tf.stack(dx_k, axis=2)  # (B, N, K)

        return dx_k

    # dy: position y difference
    def gf_dy(self, inputs):
        #    inputs: (B, N, F)
        #   outputs: (B, N, K)
        # [:, :, k]: k-th neighbor's y difference

        # get y and initialize output `dy_k`
        y = inputs[:, :, 4]  # (B, N)
        dy_k = list()
        for k in range(self.K):
            shift = k - self.k_self
            if shift < 0:
                dy_k.append(tf.pad(y[:, :shift] - y[:, -shift:], [[0, 0], [-shift, 0]]))
            elif shift == 0:
                dy_k.append(tf.zeros_like(y))
            else:  # shift > 0
                dy_k.append(tf.pad(y[:, shift:] - y[:, :-shift], [[0, 0], [0, shift]]))

        # concatenate dy_k
        dy_k = tf.stack(dy_k, axis=2)  # (B, N, K)

        return dy_k

    # dz: position z difference
    def gf_dz(self, inputs):
        #    inputs: (B, N, F)
        #   outputs: (B, N, K)
        # [:, :, k]: k-th neighbor's z difference

        # get z and initialize output `dz_k`
        z = inputs[:, :, 5]  # (B, N)
        dz_k = list()
        for k in range(self.K):
            shift = k - self.k_self
            if shift < 0:
                dz_k.append(tf.pad(z[:, :shift] - z[:, -shift:], [[0, 0], [-shift, 0]]))
            elif shift == 0:
                dz_k.append(tf.zeros_like(z))
            else:  # shift > 0
                dz_k.append(tf.pad(z[:, shift:] - z[:, :-shift], [[0, 0], [0, shift]]))

        # concatenate dz_k
        dz_k = tf.stack(dz_k, axis=2)  # (B, N, K)

        return dz_k

    # euclidean distance: sqrt(dx**2 + dy**2 + dz**2)
    def gf_euclidean(self, inputs):
        #    inputs: (B, N, F)
        #   outputs: (B, N, K)
        # [:, :, k]: k-th neighbor's euclidean distance

        # get dx, dy, dz and initialize output `euclidean_k`
        dx_k = self.gf_dx(inputs)  # (B, N, K)
        dy_k = self.gf_dy(inputs)  # (B, N, K)
        dz_k = self.gf_dz(inputs)  # (B, N, K)
        euclidean_k = tf.sqrt(dx_k**2 + dy_k**2 + dz_k**2)

        return euclidean_k

    def gf_lorentz(self, inputs):
        """Lorentz invariant distance with sign.

            ds**2 = dt**2 - dx/c**2 - dy/c**2 - dz/c**2

            ds = sign(ds**2) sqrt(abs(ds**2))

            here, c is the speed of light, 299792458 m/s,
            but t in us, x, y, z in m / 600

        Args:
                inputs (tf.Tensor): (B, N, F)

        Returns:
                lorentz_k: (B, N, K). [:, :, k]: k-th neighbor's lorentz invariant distance
        """
        # get dt, dx, dy, dz and initialize output `lorentz_k`
        dt_k = self.gf_dt(inputs)  # (B, N, K)
        dx_k = self.gf_dx(inputs) / self.speed_of_light  # (B, N, K)
        dy_k = self.gf_dy(inputs) / self.speed_of_light  # (B, N, K)
        dz_k = self.gf_dz(inputs) / self.speed_of_light  # (B, N, K)

        # lorentz invariant distance
        ds2_k = dt_k**2 - dx_k**2 - dy_k**2 - dz_k**2  # (B, N, K)
        lorentz_k = tf.sign(ds2_k) * tf.sqrt(tf.abs(ds2_k))

        return lorentz_k

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "K": self.K,
                "h_list": self.h_list,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% M-layer: mapping. (B, N, K, F_in) -> (B, N, K, F_out)
class MLayer(tf.keras.Model):
    # input_shape: (B, N, K, F_in)
    # after series of Conv2D, output_shape: (B, N, K, F_out)
    def __init__(self, filters_list: List[int], activation: str = "relu", **kwargs):
        super(MLayer, self).__init__(**kwargs)
        self.filters_list = filters_list
        self.activation = activation

        # define Conv2D layers
        self.conv2d_layers = list()
        for i, filters in enumerate(filters_list):
            self.conv2d_layers.append(
                tf.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="same",
                    activation=activation,
                    kernel_initializer="he_normal",
                    name=self.name + "_conv2d_{}".format(i + 1),
                )
            )

        # define BatchNormalization layers
        self.bn_layers = list()
        for i in range(len(filters_list)):
            self.bn_layers.append(
                tf.keras.layers.BatchNormalization(
                    name=self.name + "_bn_{}".format(i + 1),
                )
            )

    def call(self, inputs, training=None):
        # inputs: (B, N, K, F_in)
        # outputs: (B, N, K, F_out)

        # pass through Conv2D layers
        x = inputs
        for i, conv2d_layer in enumerate(self.conv2d_layers):
            x = conv2d_layer(x)
            x = self.bn_layers[i](x, training=training)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "filters_list": self.filters_list,
                "activation": self.activation,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% FR-layer: feature raising layer. (B, N, F) -> (B, N, F_out)
class FRLayer(tf.keras.Model):
    # input_shape: (B, N, F)
    # after a series of Conv1D, output_shape: (B, N, F_out)
    def __init__(self, filters_list: List[int], activation: str = "relu", **kwargs):
        super(FRLayer, self).__init__(**kwargs)
        self.filters_list = filters_list
        self.activation = activation

        # define Conv1D layers
        self.conv1d_layers = list()
        for i, filters in enumerate(filters_list):
            self.conv1d_layers.append(
                tf.keras.layers.Conv1D(
                    filters=filters,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    activation=activation,
                    kernel_initializer="he_normal",
                    name=self.name + "_conv1d_{}".format(i + 1),
                )
            )

        # define BatchNormalization layers
        self.bn_layers = list()
        for i in range(len(filters_list)):
            self.bn_layers.append(
                tf.keras.layers.BatchNormalization(
                    name=self.name + "_bn_{}".format(i + 1),
                )
            )

    def call(self, inputs, training=None):
        # inputs: (B, N, F)
        # outputs: (B, N, F_out)

        # pass through Conv1D layers
        x = inputs
        for i, conv1d_layer in enumerate(self.conv1d_layers):
            x = conv1d_layer(x)
            x = self.bn_layers[i](x, training=training)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "filters_list": self.filters_list,
                "activation": self.activation,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Feature raising layer, but using RNN instead of Conv
class FRLayerRnn(tf.keras.Model):
    # input_shape: (B, N, F)
    # after a series of RNN, output_shape: (B, N, F_out)
    def __init__(self, units_list: List[int], rnn_name: str = "gru", use_bidirectional: bool = True, **kwargs):
        super(FRLayerRnn, self).__init__(**kwargs)
        self.units_list = units_list
        self.rnn_name = rnn_name
        self.use_bidirectional = use_bidirectional

        # define RNN layers
        self.rnn_layers = list()
        for i, units in enumerate(units_list):
            if use_bidirectional:
                self.rnn_layers.append(
                    tf.keras.layers.Bidirectional(
                        getattr(tf.keras.layers, rnn_name.upper())(
                            units=units,
                            return_sequences=True,
                            name=self.name + "_{}_{}".format(rnn_name, i + 1),
                        )
                    )
                )
            else:
                self.rnn_layers.append(
                    getattr(tf.keras.layers, rnn_name.upper())(
                        units=units,
                        return_sequences=True,
                        name=self.name + "_{}_{}".format(rnn_name, i + 1),
                    )
                )

        # define BatchNormalization layers
        self.bn_layers = list()
        for i in range(len(units_list)):
            self.bn_layers.append(
                tf.keras.layers.BatchNormalization(
                    name=self.name + "_bn_{}".format(i + 1),
                )
            )
    
    def call(self, inputs, training=None):
        # inputs: (B, N, F)
        # outputs: (B, N, F_out)

        # pass through RNN layers
        x = inputs
        for i, rnn_layer in enumerate(self.rnn_layers):
            x = rnn_layer(x)
            x = self.bn_layers[i](x, training=training)

        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "units_list": self.units_list,
                "rnn_name": self.rnn_name,
                "use_bidirectional": self.use_bidirectional,
            }
        )
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% A-layer: local neighbors aggregation layer. (B, N, K, F_out) + (B, N, F_out) -> (B, N, F_out)
class ALayer(tf.keras.layers.Layer):
    # from m_nkf = (B, N, K, F_out) and fr_nf = (B, N, F_out),
    # make T_nkf = m_nkf * fr_(n+k, f)
    # and aggregate T_nkf to (B, N, F_out) by max-pooling or average-pooling
    def __init__(self, aggregation: str = "max", **kwargs):
        super(ALayer, self).__init__(**kwargs)
        assert aggregation in ["max", "avg"], "aggregation must be max or avg"
        self.aggregation = aggregation

    def call(self, inputs, training=None):
        #  inputs: (m_nkf, fr_nf)
        # outputs: (B, N, F_out)

        # get m_nkf and fr_nf
        m_nkf = inputs[0]  # (B, N, K, F_out)
        fr_nf = inputs[1]  # (B, N, F_out)

        # get B, N, K, F_out
        B, N, K, F_out = m_nkf.shape

        assert K % 2 == 1, "K must be odd"
        k_self = (K - 1) // 2

        # get T_nkf = m_nkf * fr_(n+k, f)
        T_nkf = list()
        for k in range(K):
            # T_nkf.append(m_nkf[:, :, k, :] * fr_nf[:, :, shift, :])
            shift = k - k_self
            if shift < 0:
                T_nkf.append(
                    m_nkf[:, :, k, :]
                    * tf.pad(fr_nf[:, :, :shift], [[0, 0], [0, 0], [-shift, 0]])
                )
            elif shift == 0:
                T_nkf.append(m_nkf[:, :, k, :] * fr_nf)
            else:  # shift > 0
                T_nkf.append(
                    m_nkf[:, :, k, :]
                    * tf.pad(fr_nf[:, :, shift:], [[0, 0], [0, 0], [0, shift]])
                )

        T_nkf = tf.stack(T_nkf, axis=2)  # (B, N, K, F_out)

        # aggregate T_nkf to A_nf (B, N, F_out)
        if self.aggregation == "max":
            A_nf = tf.reduce_max(T_nkf, axis=2)
        elif self.aggregation == "avg":
            A_nf = tf.reduce_mean(T_nkf, axis=2)
        else:
            raise ValueError("Invalid aggregation: {}".format(self.aggregation))

        return A_nf

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "aggregation": self.aggregation,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% CR-layer: channel raising layer. (B, N, F_out) -> (B, N, F_raise)
class CRLayer(tf.keras.Model):
    # input_shape: (B, N, F_out)
    # after a series of Conv1D, output_shape: (B, N, F_raise)
    def __init__(self, filters_list: List[int], activation: str = "relu", **kwargs):
        super(CRLayer, self).__init__(**kwargs)
        self.filters_list = filters_list
        self.activation = activation

        # define Conv1D layers
        self.conv1d_layers = list()
        for i, filters in enumerate(filters_list):
            self.conv1d_layers.append(
                tf.keras.layers.Conv1D(
                    filters=filters,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    activation=activation,
                    kernel_initializer="he_normal",
                    name=self.name + "_conv1d_{}".format(i + 1),
                )
            )

        # define BatchNormalization layers
        self.bn_layers = list()
        for i in range(len(filters_list)):
            self.bn_layers.append(
                tf.keras.layers.BatchNormalization(
                    name=self.name + "_bn_{}".format(i + 1),
                )
            )

    def call(self, inputs, training=None):
        #  inputs: (B, N, F_out)
        # outputs: (B, N, F_raise)

        # pass through Conv1D layers
        x = inputs
        for i, conv1d_layer in enumerate(self.conv1d_layers):
            x = conv1d_layer(x)
            x = self.bn_layers[i](x, training=training)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "filters_list": self.filters_list,
                "activation": self.activation,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Channel raising layer, but using RNN instead of Conv
class CRLayerRNN(tf.keras.Model):
    # input_shape: (B, N, F_out)
    # after a series of RNN, output_shape: (B, N, F_raise)
    def __init__(self, units_list: List[int], rnn_name: str = "gru", use_bidirectional: bool = True, **kwargs):
        super(CRLayerRNN, self).__init__(**kwargs)
        self.units_list = units_list
        self.rnn_name = rnn_name
        self.use_bidirectional = use_bidirectional

        # define RNN layers
        self.rnn_layers = list()
        for i, units in enumerate(units_list):
            if use_bidirectional:
                self.rnn_layers.append(
                    tf.keras.layers.Bidirectional(
                        getattr(tf.keras.layers, rnn_name.upper())(
                            units=units,
                            return_sequences=True,
                            name=self.name + "_{}_{}".format(rnn_name, i + 1),
                        )
                    )
                )
            else:
                self.rnn_layers.append(
                    getattr(tf.keras.layers, rnn_name.upper())(
                        units=units,
                        return_sequences=True,
                        name=self.name + "_{}_{}".format(rnn_name, i + 1),
                    )
                )
                
        # define BatchNormalization layers
        self.bn_layers = list()
        for i in range(len(units_list)):
            self.bn_layers.append(
                tf.keras.layers.BatchNormalization(
                    name=self.name + "_bn_{}".format(i + 1),
                )
            )
    
    def call(self, inputs, training=None):
        # inputs: (B, N, F_out)
        # outputs: (B, N, F_raise)

        # pass through RNN layers
        x = inputs
        for i, rnn_layer in enumerate(self.rnn_layers):
            x = rnn_layer(x)
            x = self.bn_layers[i](x, training=training)

        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "units_list": self.units_list,
                "rnn_name": self.rnn_name,
                "use_bidirectional": self.use_bidirectional,
            }
        )
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% RS-Block: relation shape cnn block. (B, N, F) + (B, N, F_before) -> (B, N, F) + (B, N, F_raise)
class RSBlock(tf.keras.Model):
    def __init__(
        self,
        input_block: bool,
        H_h_list: List[str],
        H_K: int,
        M_filters_list: List[int],
        M_activation: str,
        FR_filters_list: List[int],
        FR_activation: str,
        A_aggregation: str,
        CR_filters_list: List[int],
        CR_activation: str,
        **kwargs
    ):
        super(RSBlock, self).__init__(**kwargs)
        self.input_block = input_block

        assert (
            M_filters_list[-1] == FR_filters_list[-1]
        ), "M_filters_list[-1] must be equal to FR_filters_list[-1]"

        self.H_h_list = H_h_list
        self.H_K = H_K

        self.M_filters_list = M_filters_list
        self.M_activation = M_activation

        self.FR_filters_list = FR_filters_list
        self.FR_activation = FR_activation

        self.A_aggregation = A_aggregation

        self.CR_filters_list = CR_filters_list
        self.CR_activation = CR_activation

        # define H-layer
        self.h_layer = HLayer(
            h_list=H_h_list,
            K=H_K,
            name=self.name + "_HLayer",
        )

        # define M-layer
        self.m_layer = MLayer(
            filters_list=M_filters_list,
            activation=M_activation,
            name=self.name + "_MLayer",
        )

        # define FR-layer
        self.fr_layer = FRLayer(
            filters_list=FR_filters_list,
            activation=FR_activation,
            name=self.name + "_FRLayer",
        )

        # define A-layer
        self.a_layer = ALayer(
            aggregation=A_aggregation,
            name=self.name + "_ALayer",
        )

        # define CR-layer
        self.cr_layer = CRLayer(
            filters_list=CR_filters_list,
            activation=CR_activation,
            name=self.name + "_CRLayer",
        )

    def call(self, inputs, training=None):
        #   inputs: (data, features)
        #     data: (B, N, F)
        # features: (B, N, F) for the input block, (B, N, F_raise_before) for the rest
        #
        #  outputs: (data, cr(a(m(h(d)), fr(f))))
        #        a: (B, N, F_raise))

        # unpack inputs
        if self.input_block:
            data = inputs
            features = data
        else:
            data = inputs[0]
            features = inputs[1]

        # H-layer
        h = self.h_layer(data)

        # M-layer
        m = self.m_layer(h)

        # FR-layer
        fr = self.fr_layer(features)

        # A-layer
        a = self.a_layer((m, fr))

        # CR-layer
        cr = self.cr_layer(a)

        return (data, cr)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "input_block": self.input_block,
                "H_h_list": self.H_h_list,
                "H_K": self.H_K,
                "M_filters_list": self.M_filters_list,
                "M_activation": self.M_activation,
                "FR_filters_list": self.FR_filters_list,
                "FR_activation": self.FR_activation,
                "A_aggregation": self.A_aggregation,
                "CR_filters_list": self.CR_filters_list,
                "CR_activation": self.CR_activation,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% GA-Block: global aggregation layer. (B, N, F_concat) -> (B, output)
class GABlockResRNN(tf.keras.Model):
    # (B, N, F_concat) -> (B, output)
    def __init__(
        self,
        rnn_stacks: int = 3,
        dense_units_list: List[int] = [256],
        rnn_name: str = "gru",
        dense_activation: str = "relu",
        dense_dropout: float = 0.0,
        use_bidirectional: bool = True,
        **kwargs
    ):
        super(GABlockResRNN, self).__init__(**kwargs)
        self.rnn_stacks = rnn_stacks
        self.dense_units_list = dense_units_list
        self.rnn_name = rnn_name
        self.dense_activation = dense_activation
        self.dense_dropout = dense_dropout
        self.use_bidirectional = use_bidirectional


    def build(self, input_shape):
        # input_shape: (B, N, F_concat)
        if self.use_bidirectional:
            assert (input_shape[-1] % 2 == 0), "input_shape[-1] must be even when use_bidirectional=True"
            self.rnn_units = input_shape[-1] // 2
        else:
            self.rnn_units = input_shape[-1]

        # define RNN layer
        self.rnn_layers = list()
        for i in range(self.rnn_stacks):
            if i == self.rnn_stacks - 1:
                return_sequences = False
            else:
                return_sequences = True

            if self.use_bidirectional:
                rnn_layer = tf.keras.layers.Bidirectional(
                    getattr(tf.keras.layers, self.rnn_name.upper())(
                        units=self.rnn_units,
                        return_sequences=return_sequences,
                        name=self.name + "_RNNLayer_" + str(i),
                    )
                )
            else:
                rnn_layer = getattr(tf.keras.layers, self.rnn_name.upper())(
                    units=self.rnn_units,
                    return_sequences=return_sequences,
                    name=self.name + "_RNNLayer_" + str(i),
                )
            self.rnn_layers.append(rnn_layer)

        # define dense layers and dropout layers
        self.dense_layers = list()
        self.dropout_layers = list()
        for i, units in enumerate(self.dense_units_list):
            dense_layer = tf.keras.layers.Dense(
                units=units,
                activation=self.dense_activation,
                name=self.name + "_DenseLayer_" + str(i),
            )
            self.dense_layers.append(dense_layer)

            dropout_layer = tf.keras.layers.Dropout(
                rate=self.dense_dropout,
                name=self.name + "_DropoutLayer_" + str(i),
            )
            self.dropout_layers.append(dropout_layer)

        # define output layer
        self.output_layer = tf.keras.layers.Dense(
            units=3,
            activation="linear",
            name=self.name + "_OutputLayer",
        )

    def call(self, inputs, training=None):
        # inputs: (B, N, F_concat)
        # outputs: (B, 3)

        # RNN layers
        x = inputs
        for i, rnn_layer in enumerate(self.rnn_layers[:-1]):
            x_new = rnn_layer(x)
            x = x + x_new
        
        x = self.rnn_layers[-1](x)

        # dense layers
        for i, dense_layer in enumerate(self.dense_layers):
            x = dense_layer(x)
            x = self.dropout_layers[i](x, training=training)

        # output layer
        outputs = self.output_layer(x)

        return outputs

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "rnn_stacks": self.rnn_stacks,
                "dense_units_list": self.dense_units_list,
                "rnn_name": self.rnn_name,
                "dense_activation": self.dense_activation,
                "dense_dropout": self.dense_dropout,
                "use_bidirectional": self.use_bidirectional,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% Main function is for testing
if __name__ == "__main__":
    # start of main
    print("\n\nLayers.py is running as main\n")
    _utils.GpuMemoryManagement(verbose=True)
    print("")

    # pseudo data with shape (B=batch, N=time, F=feature)
    B = 32
    N = 128
    F = 6
    data = tf.random.normal(shape=(B, N, F))

    print("data.shape =", data.shape)

    # define H-Layer
    h_list = [
        "t",
        "x",
        "y",
        "z",
        "c",
        "a",
        "dt",
        "dx",
        "dy",
        "dz",
        "euclidean",
        "lorentz",
    ]
    K = 5
    h_layer = HLayer(h_list, K)

    # call H-Layer
    h = h_layer(data)
    print("h.shape =", h.shape)

    # # print h
    # print("")
    # print("data[0, :7, :] = ", data[0, :7, :])

    # for h_i, h_name in enumerate(h_list):
    #     print("")
    #     print("input " + h_name + ": h[0, :7, :, " + str(h_i) + "] = ", h[0, :7, :, h_i])

    # define and call M-Layer
    m_layer = MLayer([16, 32, 64])
    m = m_layer(h)
    print("m.shape =", m.shape)

    # define and call FR-Layer
    fr_layer = FRLayer([16, 32, 64])
    fr = fr_layer(data)
    print("fr.shape =", fr.shape)

    # define and call A-Layer
    a_layer = ALayer()
    a = a_layer([m, fr])
    print("a.shape =", a.shape)

    # define and call CR-Layer
    cr_layer = CRLayer([64, 128, 256])
    cr = cr_layer(a)
    print("cr.shape =", cr.shape)

    # define and call RS-Block
    rs_block = RSBlock(
        input_block=True,
        H_h_list=h_list,
        H_K=K,
        M_filters_list=[16, 32, 64],
        M_activation="relu",
        FR_filters_list=[16, 32, 64],
        FR_activation="relu",
        A_aggregation="avg",
        CR_filters_list=[64, 128],
        CR_activation="relu",
        name="RS0",
    )
    rs_block_output = rs_block(data)
    print("rs_block_output[0].shape =", rs_block_output[0].shape)
    print("rs_block_output[1].shape =", rs_block_output[1].shape)

    # define and call second RS-Block
    rs_block2 = RSBlock(
        input_block=False,
        H_h_list=h_list,
        H_K=K + 4,
        M_filters_list=[32, 64, 128],
        M_activation="relu",
        FR_filters_list=[24, 48, 128],
        FR_activation="relu",
        A_aggregation="max",
        CR_filters_list=[128, 256],
        CR_activation="relu",
        name="RS1",
    )
    rs_block2_output = rs_block2(rs_block_output)
    print("rs_block2_output[0].shape =", rs_block2_output[0].shape)
    print("rs_block2_output[1].shape =", rs_block2_output[1].shape)

    # define and call GA-Block
    concatenated_input = tf.concat(
        [rs_block2_output[0], rs_block2_output[1]], axis=2
    )
    
    print("concatenated_input.shape =", concatenated_input.shape)

    ga_block = GABlockResRNN(
        dense_units_list=[64, 32],
        rnn_name="GRU",
        dense_activation="relu",
        dense_dropout=0.2,
        use_bidirectional=True,
    )

    ga_block_output = ga_block(concatenated_input)
    print("   ga_block_output.shape =", ga_block_output.shape)

    # end of main
    print("DONE")

# %% END-OF-FILE
