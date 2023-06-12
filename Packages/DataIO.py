# %% Imports
import gc
from typing import List

import numpy as np
import tensorflow as tf

try:
    from . import Utils as _utils
except:
    import Utils as _utils


# %% Define generator with data augmentation
def point_picker_generator(
    point_picker_paths: List[str],
    n_features: int = 6,
    azimuth_rotation: bool = False,
    time_tweak_ns: float = 0.0,
    time_translation_ns: float = 0.0,
    position_tweak_sigma: float = 0.0,
    position_translation_sigma: float = 0.0,
    shuffle_batch: bool = False,
):
    # hard coded resolution
    outer_xy_resolution = 125.0 / 2
    long_z_resolution = 17.0 / 2

    # shuffle batch, ie. shuffle the point_picker_paths
    if shuffle_batch:
        np.random.shuffle(point_picker_paths)
    for path in point_picker_paths:
        # read data file
        data_file = np.load(path)

        data_x = data_file["x"].astype("float32")
        data_y = data_file["y"].astype("float32")

        # read data
        features = data_x[:, :, :n_features]
        labels = data_y[:, :2]

        # Data Augmentation: rotate
        if azimuth_rotation:
            # random number generation
            az_tweak = np.random.uniform(0, 2 * np.pi)

            # rotate
            new_x = features[:, :, 3] * np.cos(az_tweak) - features[:, :, 4] * np.sin(
                az_tweak
            )
            new_y = features[:, :, 3] * np.sin(az_tweak) + features[:, :, 4] * np.cos(
                az_tweak
            )

            # apply
            features[:, :, 3] = new_x
            features[:, :, 4] = new_y

            # rotate label
            labels[:, 0] += az_tweak
            labels[:, 0] %= 2 * np.pi

        # Data Augmentation: time tweak
        if time_tweak_ns > 0:
            time_tweak = np.random.normal(0, time_tweak_ns, size=features.shape[:2])
            features[:, :, 0] += time_tweak

        # Data Augmentation: time translation
        # shift the time (0-th feature) for each events
        if time_translation_ns > 0:
            time_translation = np.random.normal(
                0, time_translation_ns, size=features.shape[0]
            )
            features[:, :, 0] += time_translation[:, np.newaxis]

        # Data Augmentation: position tweak
        if position_tweak_sigma > 0:
            # read error
            r_err = data_x[:, :, 6]
            z_err = data_x[:, :, 7]

            # random number generation
            x_tweak = np.random.normal(0, position_tweak_sigma, size=features.shape[:2])
            y_tweak = np.random.normal(0, position_tweak_sigma, size=features.shape[:2])
            z_tweak = np.random.normal(0, position_tweak_sigma, size=features.shape[:2])

            # apply error
            x_tweak *= r_err
            y_tweak *= r_err
            z_tweak *= z_err

            # apply tweak
            features[:, :, 3] += x_tweak
            features[:, :, 4] += y_tweak
            features[:, :, 5] += z_tweak

        # Data Augmentation: position translation
        # shift the position (3-5-th feature) for each events
        if position_translation_sigma > 0:
            # random number generation
            x_translation = np.random.normal(
                0, position_translation_sigma, size=features.shape[0]
            )
            y_translation = np.random.normal(
                0, position_translation_sigma, size=features.shape[0]
            )
            z_translation = np.random.normal(
                0, position_translation_sigma, size=features.shape[0]
            )

            # apply error
            x_translation *= outer_xy_resolution
            y_translation *= outer_xy_resolution
            z_translation *= long_z_resolution

            # apply translation
            features[:, :, 3] += x_translation[:, np.newaxis]
            features[:, :, 4] += y_translation[:, np.newaxis]
            features[:, :, 5] += z_translation[:, np.newaxis]

        # normalize
        features[:, :, 0] -= features[:, :, 0].min(axis=1, keepdims=True)
        features[:, :, 0] /= 1000  # time
        features[:, :, 1] /= 300  # charge
        features[:, :, 3:] /= 600  # space

        features[:, :, :][features[:, :, 1] == 0.] = 0.  # for masking

        # consume data
        for feature, label in zip(features, labels):
            yield (feature, label)

        # release memory
        if azimuth_rotation:
            del az_tweak, new_x, new_y
        if time_tweak_ns > 0:
            del time_tweak
        if time_translation_ns > 0:
            del time_translation
        if position_tweak_sigma > 0:
            del r_err, z_err, x_tweak, y_tweak, z_tweak
        if position_translation_sigma > 0:
            del x_translation, y_translation, z_translation
        del features, labels, data_x, data_y
        data_file.close()
        _ = gc.collect()


# %% Define data generator for low NA pulses data
def point_picker_generator_low_na(
    point_picker_paths: List[str],
    n_features: int = 6,
    max_na_pulses: int = 128,
    shuffle_batch: bool = False,
):
    # shuffle batch, ie. shuffle the point_picker_paths
    if shuffle_batch:
        np.random.shuffle(point_picker_paths)

    for path in point_picker_paths:
        # read data file
        data_file = np.load(path)

        data_x = data_file["x"].astype("float32")
        data_y = data_file["y"].astype("float32")

        # read data
        features = data_x[:, :, :n_features]
        labels = data_y[:, :2]

        # non-aux pulse count mask
        na_pulses = (data_x[:, :, 2] == 0).sum(axis=1)
        na_pulses_mask = na_pulses < max_na_pulses

        # apply mask
        features = features[na_pulses_mask]
        labels = labels[na_pulses_mask]

        # normalize
        features[:, :, 0] -= features[:, :, 0].min(axis=1, keepdims=True)
        features[:, :, 0] /= 1000  # time
        features[:, :, 1] /= 300  # charge
        features[:, :, 3:] /= 600  # space

        features[:, :, :][features[:, :, 1] == 0.] = 0.  # for masking

        # consume data
        for feature, label in zip(features, labels):
            yield (feature, label)

        # release memory
        del features, labels, data_x, data_y
        data_file.close()
        _ = gc.collect()

# %% Define data generator for lite version.
def point_picker_generator_lite(
    point_picker_paths: List[str],
    max_pulses: int = 128,
    shuffle_batch: bool = False,
):
    # shuffle batch, ie. shuffle the point_picker_paths
    if shuffle_batch:
        np.random.shuffle(point_picker_paths)

    for path in point_picker_paths:
        # read data file
        data_file = np.load(path)

        data_x = data_file["x"].astype("float32")
        data_y = data_file["y"].astype("float32")

        # pulse count mask
        pulses = (data_x[:, :, 2] != -1).sum(axis=1)
        pulses_mask = pulses < max_pulses
        
        # apply mask
        data_x = data_x[pulses_mask, :max_pulses, :]
        data_y = data_y[pulses_mask]

        # make is_deep core variable
        is_deep = (data_x[:, :, 7] < 4.).astype("float")

        # read data, append is_deep to features
        features = data_x[:, :, :6]
        features = np.concatenate((features, is_deep[:, :, np.newaxis]), axis=2)
        labels = data_y[:, :2]

        # normalize
        features[:, :, 0] -= features[:, :, 0].min(axis=1, keepdims=True)
        features[:, :, 0] /= 1000  # time
        features[:, :, 1] /= 300  # charge
        features[:, :, 3:] /= 600  # space

        features[:, :, :][features[:, :, 1] == 0.] = 0.  # for masking layer

        # consume data
        for feature, label in zip(features, labels):
            yield (feature, label)

        # release memory
        del features, labels, data_x, data_y
        data_file.close()
        _ = gc.collect()

# %% Main function is for testing
if __name__ == "__main__":
    # start of main
    print("\n\nDataIO.py is running as main\n")
    _utils.GpuMemoryManagement(verbose=True)

    # directory setting
    home_dir = "wowdao-self-supervised-learning-dataset"
    data_dir = home_dir + "/"
    point_picker_format = data_dir
    point_picker_format += "train/pointpicker_mpc128_n9_batch_{batch_id:d}.npz"

    batch_ids = np.arange(1, 2, 1, dtype=np.int32)
    batch_size = 32

    # data augmentation setting and args
    data_aug_az_rotation = True
    time_tweak_sigma = 2.0
    position_tweak_sigma = 1.0 / 3.0

    dataset_args = [
        [point_picker_format.format(batch_id=batch_id) for batch_id in batch_ids],
        6,
        data_aug_az_rotation,
        time_tweak_sigma,
        position_tweak_sigma,
    ]

    # intermediate verbose
    print("dataset_args:")
    for value in dataset_args:
        print(f"{value}")

    # define dataset
    print("defining dataset...")
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataset = tf.data.Dataset.from_generator(
        point_picker_generator,
        args=dataset_args,
        output_types=(tf.float32, tf.float32),
    )
    dataset = dataset.repeat().batch(batch_size).prefetch(AUTOTUNE)

    # Get one batch
    print("dataset.take(1):")
    for features, labels in dataset.take(1):
        print("features.shape: ", features.shape)
        print("  labels.shape: ", labels.shape)

        print(features)

    # point_picker_generator_lite
    print("\n\nTest point_picker_generator_lite")
    dataset_args = [
        [point_picker_format.format(batch_id=batch_id) for batch_id in batch_ids],
        80,
        False,
    ]

    # intermediate verbose
    print("dataset_args:")
    for value in dataset_args:
        print(f"{value}")

    # define dataset
    print("defining dataset...")
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    dataset = tf.data.Dataset.from_generator(
        point_picker_generator_lite,
        args=dataset_args,
        output_types=(tf.float32, tf.float32),
    )

    dataset = dataset.repeat().batch(batch_size).prefetch(AUTOTUNE)

    # Get one batch
    print("dataset.take(1):")
    for features, labels in dataset.take(1):
        print("features.shape: ", features.shape)
        print("  labels.shape: ", labels.shape)

        print(features)
    
    # end of main
    print("DONE")

# %% END-OF-FILE
