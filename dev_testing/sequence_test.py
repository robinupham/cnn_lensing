"""
Testing loading training data using a Sequence rather than loading it all into memory at once.
"""

import os

import numpy as np
import tensorflow.keras as keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class TrainingGenerator(keras.utils.Sequence):
    """
    Generator to load training data in batches.
    """

    def __init__(self, file_mask, x_arr_id, y_arr_id, n_files, shuffle=True, augment=True):
        self.file_mask = file_mask
        self.x_arr_id = x_arr_id
        self.y_arr_id = y_arr_id
        self.n_files = n_files
        self.shuffle = shuffle
        self.augment = augment

    def __len__(self):
        return self.n_files

    @staticmethod
    def augment_data(xy, shuffle=True):
        """
        Augment data by applying 7 unique combinations of rotations and reflections.
        Expecting data in shape (n_samples, 2, npix, npix, 1).
        Will return data in shape (8*n_samples, 2, npix, npix, 1).
        """

        n_samp = xy.shape[0]
        npix = xy.shape[2]
        shape = (n_samp, 2, npix, npix, 1)
        assert xy.shape == shape

        # Rotate 90
        xy_rotate90 = np.rot90(xy, 1, axes=(3, 2))
        assert xy_rotate90.shape == shape

        # Rotate 180
        xy_rotate180 = np.rot90(xy, 2, axes=(3, 2))
        assert xy_rotate180.shape == shape

        # Rotate 270
        xy_rotate270 = np.rot90(xy, 3, axes=(3, 2))
        assert xy_rotate270.shape == shape

        # Flip vert
        xy_flipvert = np.flip(xy, axis=2)
        assert xy_flipvert.shape == shape

        # Flip horiz
        xy_fliphoriz = np.flip(xy, axis=3)
        assert xy_fliphoriz.shape == shape

        # Rotate 90 + flip vert
        xy_rotate90_flipvert = np.flip(xy_rotate90, axis=2)
        assert xy_rotate90_flipvert.shape == shape

        # Rotate 90 + flip horiz
        xy_rotate90_fliphoriz = np.flip(xy_rotate90, axis=3)
        assert xy_rotate90_fliphoriz.shape == shape

        # Concatenate and shuffle
        xy_all = (xy, xy_rotate90, xy_rotate180, xy_rotate270, xy_flipvert, xy_fliphoriz, xy_rotate90_flipvert,
                  xy_rotate90_fliphoriz)
        xy = np.concatenate(xy_all, axis=0)
        if shuffle:
            np.random.default_rng().shuffle(xy, axis=0)

        return xy

    def __getitem__(self, idx):

        # Load data and stack into a single array of shape (n_samp, n_pix, n_pix, n_channel)
        train_path = self.file_mask.format(idx=idx)
        with np.load(train_path) as data:
            x = data[self.x_arr_id]
            y = data[self.y_arr_id]
        xy = np.stack((x, y), axis=1)

        # Augment and shuffle
        if self.augment:
            xy = self.__class__.augment_data(xy, shuffle=self.shuffle)

        # Re-split into x and y, do some checks and return tuple
        x = xy[:, 0, ...]
        y = xy[:, 1, ...]
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))
        assert np.amin(x) > 0
        assert np.amax(x) < 1
        assert np.amin(y) > 0
        assert np.amax(y) < 1
        return x, y


def load_validation_data(data_path, x_arr_id, y_arr_id, shuffle=True):
    """
    Loads validation data from a single npz file where x and y are in data[x_arr_id] and data[y_arr_id].

    Shuffles if requested, but doesn't augment.
    """

    # Load data and stack into a single array of shape (n_samp, n_pix, n_pix, n_channel)
    with np.load(data_path) as data:
        x = data[x_arr_id]
        y = data[y_arr_id]
    xy = np.stack((x, y), axis=1)
    print('Loaded validation data, length', len(xy))

    # Shuffle
    if shuffle:
        np.random.default_rng().shuffle(xy, axis=0)

    # Split into x and y
    x = xy[:, 0, ...]
    y = xy[:, 1, ...]

    # Do some final checks and return
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    assert np.amin(x) > 0
    assert np.amax(x) < 1
    assert np.amin(y) > 0
    assert np.amax(y) < 1
    return x, y


def main():
    """
    Train a given model depending on the command line argument.
    """

    # Input parameters
    train_mask = '/path/to/train_{idx}.npz'
    n_training_files = 5
    val_path = '/path/to/val.npz'
    x_arr_id = 'lensed'
    y_arr_id = 'kappa'

    # Training parameters
    epochs = 100

    # Prepare training and validation data
    train_xy = TrainingGenerator(train_mask, x_arr_id, y_arr_id, n_training_files)
    val_xy = load_validation_data(val_path, x_arr_id, y_arr_id, shuffle=True)

    # Define a simple model
    conv_args = {
        'activation': 'relu',
        'kernel_initializer': 'Orthogonal',
        'padding': 'same',
    }
    model = keras.Sequential([
        keras.Input(shape=(50, 50, 1)),
        keras.layers.Conv2D(8, 3, **conv_args),
        keras.layers.Conv2D(1, 3, **conv_args)
    ])

    # Train
    opt = keras.optimizers.Adam(clipnorm=1)
    model.compile(optimizer=opt, loss='mean_squared_error')
    model.fit(train_xy, epochs=epochs, verbose=1, validation_data=val_xy)
