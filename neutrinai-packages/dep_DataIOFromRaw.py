# %% Imports
import gc
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List

import numpy as np
import tensorflow as tf

try:
    from . import Utils as _utils
except:
    import Utils as _utils

# TODO: ROLL BACK TO ORIGINAL


# %% Expand sensor geometry. append resolution
def expand_sensor_geometry(
    sensor_geometry_df: pd.DataFrame,
):
    # set sensor geometry
    # counts
    doms_per_string = 60
    string_num = 86

    # index
    outer_long_strings = np.concatenate(
        [np.arange(0, 25), np.arange(27, 34), np.arange(37, 44), np.arange(46, 78)]
    )
    inner_long_strings = np.array([25, 26, 34, 35, 36, 44, 45])
    inner_short_strings = np.array([78, 79, 80, 81, 82, 83, 84, 85])

    # known specs
    outer_xy_resolution = 125.0 / 2
    inner_xy_resolution = 70.0 / 2
    long_z_resolution = 17.0 / 2
    short_z_resolution = 7.0 / 2

    # evaluate error
    sensor_x = sensor_geometry_df.x
    sensor_y = sensor_geometry_df.y
    sensor_z = sensor_geometry_df.z
    sensor_r_err = np.ones(doms_per_string * string_num)
    sensor_z_err = np.ones(doms_per_string * string_num)

    # r-error
    for string_id in outer_long_strings:
        sensor_r_err[
            string_id * doms_per_string : (string_id + 1) * doms_per_string
        ] = outer_xy_resolution
    for string_id in np.concatenate([inner_long_strings, inner_short_strings]):
        sensor_r_err[
            string_id * doms_per_string : (string_id + 1) * doms_per_string
        ] = inner_xy_resolution

    # z-error
    for string_id in outer_long_strings:
        sensor_z_err[
            string_id * doms_per_string : (string_id + 1) * doms_per_string
        ] = long_z_resolution
    for string_id in np.concatenate([inner_long_strings, inner_short_strings]):
        for dom_id in range(doms_per_string):
            z = sensor_z[string_id * doms_per_string + dom_id]
            if (z < -156.0) or (z > 95.5 and z < 191.5):
                sensor_z_err[string_id * doms_per_string + dom_id] = short_z_resolution
            else:
                sensor_z_err[string_id * doms_per_string + dom_id] = long_z_resolution

    # register
    sensor_geometry_df["r_err"] = sensor_r_err
    sensor_geometry_df["z_err"] = sensor_z_err


# %% Define data generator. TODO: Implement data augmentation
def data_generator_from_npz(
    batch_ids: np.ndarray,
    max_pulse_count: int,
    return_label: bool,
    meta_data_path: str = "wowdao-self-supervised-learning-dataset/train_meta.parquet",
    data_path_header: str = "wowdao-self-supervised-learning-dataset/train/batch_",
    sensor_geometry_path: str = "wowdao-self-supervised-learning-dataset/sensor_geometry.csv",
    shuffle: bool = True,
    verbose: bool = False,
):
    # type conversion for tf
    if type(meta_data_path) == bytes:
        meta_data_path = str(meta_data_path, "utf-8")
    if type(data_path_header) == bytes:
        data_path_header = str(data_path_header, "utf-8")

    if verbose:
        print(meta_data_path)
        print(data_path_header)

    # set sensor geometry
    sensor_geometry_df = pd.read_csv(sensor_geometry_path)

    if "r_err" not in sensor_geometry_df.columns:
        expand_sensor_geometry(sensor_geometry_df)

    # detector constants
    c_const = 0.299792458  # speed of light [m/ns]

    x_min = sensor_geometry_df.x.min()
    x_max = sensor_geometry_df.x.max()
    y_min = sensor_geometry_df.y.min()
    y_max = sensor_geometry_df.y.max()
    z_min = sensor_geometry_df.z.min()
    z_max = sensor_geometry_df.z.max()

    detector_length = np.sqrt(
        (x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2
    )
    t_valid_length = detector_length / c_const

    if verbose:
        print("t_valid_length: ", t_valid_length)

    # read meta data
    if verbose:
        print("Reading meta data")
    meta_data_df = pd.read_parquet(meta_data_path)

    # read one batch
    if shuffle:
        if verbose:
            print("shuffle batch ids")
        batch_ids = np.random.permutation(batch_ids)
    for batch_id in batch_ids:
        # batch meta data
        if verbose:
            print("read batch " + str(batch_id) + " file")
        meta_batch = meta_data_df[meta_data_df.batch_id == batch_id]
        event_ids = meta_batch["event_id"].to_numpy()

        # batch data
        data_batch = np.load(data_path_header + f"{batch_id:d}.npz")
        batch_sensor_id = data_batch["sensor_id"]
        batch_time = data_batch["time"]
        batch_charge = data_batch["charge"]
        batch_auxiliary = data_batch["auxiliary"]

        pulse_index = np.append(
            meta_batch["first_pulse_index"].to_numpy(), [len(batch_sensor_id)]
        )

        # read each event
        if return_label:
            azimuth = meta_batch["azimuth"].to_numpy()
            zenith = meta_batch["zenith"].to_numpy()
            event_iter = zip(
                event_ids, pulse_index[:-1], pulse_index[1:], azimuth, zenith
            )
        else:
            event_iter = zip(event_ids, pulse_index[:-1], pulse_index[1:])
        if shuffle:
            event_iter = np.random.permutation(list(event_iter))

        for event_info in event_iter:
            # unpack event info
            if return_label:
                event_id, first_idx, last_idx, event_azimuth, event_zenith = event_info
                event_id, first_idx, last_idx = (
                    int(event_id),
                    int(first_idx),
                    int(last_idx),
                )
            else:
                event_id, first_idx, last_idx = event_info
                event_id, first_idx, last_idx = (
                    int(event_id),
                    int(first_idx),
                    int(last_idx),
                )

            # get event data
            event_time = batch_time[first_idx:last_idx].astype("float")
            event_charge = batch_charge[first_idx:last_idx].astype("float")
            event_auxiliary = batch_auxiliary[first_idx:last_idx].astype("float")
            event_x = sensor_geometry_df.x[
                batch_sensor_id[first_idx:last_idx]
            ].values.astype("float")
            event_y = sensor_geometry_df.y[
                batch_sensor_id[first_idx:last_idx]
            ].values.astype("float")
            event_z = sensor_geometry_df.z[
                batch_sensor_id[first_idx:last_idx]
            ].values.astype("float")

            # point picker
            if len(event_x) > max_pulse_count:
                # pack event data for sorting
                dtype = [
                    ("time", "float"),
                    ("charge", "float"),
                    ("auxiliary", "float"),
                    ("x", "float"),
                    ("y", "float"),
                    ("z", "float"),
                    ("rank", "short"),
                ]
                event_features = np.zeros(last_idx - first_idx, dtype)
                event_features["time"] = event_time
                event_features["charge"] = event_charge
                event_features["auxiliary"] = event_auxiliary
                event_features["x"] = event_x
                event_features["y"] = event_y
                event_features["z"] = event_z

                # find valid time window
                t_peak = event_time[event_charge.argmax()]
                t_valid_min = t_peak - t_valid_length
                t_valid_max = t_peak + t_valid_length

                # rank
                t_valid = (event_time > t_valid_min) * (event_time < t_valid_max)
                event_features["rank"] = 2 * (1 - event_auxiliary) + t_valid

                # sort by rank and charge
                event_features = np.sort(event_features, order=["rank", "charge"])

                # pick-up from backward
                event_features = event_features[-max_pulse_count:]

                # resort by time
                event_features = np.sort(event_features, order="time")

                # unpack
                event_time = event_features["time"]
                event_charge = event_features["charge"]
                event_auxiliary = event_features["auxiliary"]
                event_x = event_features["x"]
                event_y = event_features["y"]
                event_z = event_features["z"]

            # assign and normalize
            pulse_count = min(len(event_x), max_pulse_count)
            features = np.zeros((max_pulse_count, 6), dtype="float")
            features[:pulse_count, 0] = (event_time - event_time.min()) / 1000.0
            features[:pulse_count, 1] = event_charge / 300.0
            features[:pulse_count, 2] = event_auxiliary * 1.0
            features[:pulse_count, 3] = event_x / 600.0
            features[:pulse_count, 4] = event_y / 600.0
            features[:pulse_count, 5] = event_z / 600.0

            # yield event
            if return_label:
                label = np.array([event_azimuth, event_zenith], dtype="float")
                yield features, label
            else:
                yield features

        # memory management
        del (
            data_batch,
            pulse_index,
            batch_auxiliary,
            batch_charge,
            batch_time,
            batch_sensor_id,
            event_ids,
            meta_batch,
        )
        _ = gc.collect()


# %% Main function is for testing
if __name__ == "__main__":
    # start of main
    print("\n\nDataIO.py is running as main\n")
    _utils.GpuMemoryManagement(verbose=True)

    # directory setting
    home_dir = "wowdao-self-supervised-learning-dataset"
    data_dir = home_dir + "/"

    batch_ids = np.arange(90, 92, 1, dtype=np.int32)
    batch_size = 32

    max_pulse_count = 96
    return_label = False

    # test data generator
    # gen = data_generator_from_npz(batch_ids, 96, verbose=True, return_label=False)
    # print(next(gen))

    # tf dataset
    dataset_args = [batch_ids, max_pulse_count, return_label]
    output_shape = (max_pulse_count, 6)
    output_dtype = tf.float32
    if return_label:
        output_shape = (output_shape, (2, ))
        output_dtype = (output_dtype, tf.float32)
    print("dataset_args:")
    for value in dataset_args:
        print(f"{value}")

    # define dataset
    print("defining dataset...")
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataset = tf.data.Dataset.from_generator(
        data_generator_from_npz,
        args=dataset_args,
        output_types=output_dtype,
        output_shapes=output_shape,
    )
    dataset = dataset.repeat().batch(batch_size).prefetch(AUTOTUNE)

    # Get one batch
    print("dataset.take(1):")
    for data in dataset.take(1):
        if return_label:
            features, labels = data
            print("features.shape: ", features.shape)
            print("  labels.shape: ", labels.shape)
        else:
            features = data
            print("features.shape: ", features.shape)

    # end of main
    print("DONE")

# %% END-OF-FILE
