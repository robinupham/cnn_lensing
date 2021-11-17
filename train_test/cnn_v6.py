"""
CNN script for v6 of the lensing estimation testing
- like v5 but with a wider field of view but keeping the high resolution.

Incorporating Sequence approach to loading training data.
"""

import os
import pathlib
import sys
import time

# pylint: disable=wrong-import-position
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
import pymaster as nmt


class TrainingGenerator(keras.utils.Sequence):
    """
    Generator to load training data in batches.
    """

    def __init__(self, file_mask, x_arr_id, y_arr_id, n_files, batches_per_file=1, shuffle=True, augment=True):
        self.file_mask = file_mask
        self.x_arr_id = x_arr_id
        self.y_arr_id = y_arr_id
        self.batches_per_file = batches_per_file
        self.n_batches = n_files * batches_per_file
        self.shuffle = shuffle
        self.augment = augment
        self.file_xy = None
        self.batch_size = None


    def __len__(self):
        return self.n_batches


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
        assert xy.shape == shape, xy.shape
        xy_out = np.concatenate((xy, np.full((7 * n_samp, 2, npix, npix, 1), np.nan)), axis=0)
        xy = xy_out[:n_samp, ...]

        # Rotate 90
        xy_out[n_samp:(2 * n_samp), ...] = np.rot90(xy, 1, axes=(3, 2))

        # Rotate 180
        xy_out[(2 * n_samp):(3 * n_samp), ...] = np.rot90(xy, 2, axes=(3, 2))

        # Rotate 270
        xy_out[(3 * n_samp):(4 * n_samp), ...] = np.rot90(xy, 3, axes=(3, 2))

        # Flip vert
        xy_out[(4 * n_samp):(5 * n_samp), ...] = np.flip(xy, axis=2)

        # Flip horiz
        xy_out[(5 * n_samp):(6 * n_samp), ...] = np.flip(xy, axis=3)

        # Rotate 90 + flip vert
        xy_out[(6 * n_samp):(7 * n_samp), ...] = np.flip(xy_out[n_samp:(2 * n_samp), ...], axis=2)

        # Rotate 90 + flip horiz
        xy_out[(7 * n_samp):(8 * n_samp), ...] = np.flip(xy_out[n_samp:(2 * n_samp), ...], axis=3)

        # Shuffle
        if shuffle:
            np.random.default_rng().shuffle(xy_out, axis=0)

        return xy_out


    def __getitem__(self, idx):

        idx_within_file = idx % self.batches_per_file

        # If first batch in file, load data and stack into a single array of shape (n_samp, n_pix, n_pix, n_channel)
        if idx_within_file == 0:
            train_path = self.file_mask.format(idx=idx)
            with np.load(train_path) as data:
                self.file_xy = np.stack((data[self.x_arr_id], data[self.y_arr_id]), axis=1)
            self.file_xy = self.file_xy[..., np.newaxis]
            self.batch_size = int(np.ceil(self.file_xy.shape[0] / self.batches_per_file))

        # Select slice for this batch
        batch_start_idx = idx_within_file * self.batch_size
        batch_stop_idx = batch_start_idx + self.batch_size
        batch_xy = self.file_xy[batch_start_idx:batch_stop_idx, ...]

        # Augment and shuffle
        if self.augment:
            batch_xy = self.__class__.augment_data(batch_xy, shuffle=self.shuffle)

        # Re-split into x and y, do some checks and return
        x = batch_xy[:, 0, ...]
        y = batch_xy[:, 1, ...]
        assert np.all(np.isfinite(x)), x
        assert np.all(np.isfinite(y)), y
        assert np.amin(x) > 0, np.amin(x)
        assert np.amax(x) < 1, np.amax(x)
        assert np.amin(y) > 0, np.amin(y)
        assert np.amax(y) < 1, np.amax(y)

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
    xy = xy[..., np.newaxis]
    print('Loaded validation data, length', len(xy))

    # Shuffle
    if shuffle:
        np.random.default_rng().shuffle(xy, axis=0)

    # Split into x and y
    x = xy[:, 0, ...]
    y = xy[:, 1, ...]

    # Do some final checks and return
    assert np.all(np.isfinite(x)), x
    assert np.all(np.isfinite(y)), y
    assert np.amin(x) > 0, np.amin(x)
    assert np.amax(x) < 1, np.amax(x)
    assert np.amin(y) > 0, np.amin(y)
    assert np.amax(y) < 1, np.amax(y)
    return x, y


def train_model(model, train_xy, val_xy, epochs, checkpoint_path):
    """
    Train a given model with training and validation data for a given number of epochs, saving checkpoints.
    """

    # Checkpoint to save weights every time val loss is improved upon
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                                                 mode='min', save_weights_only=False, save_freq='epoch')

    class BatchLossLogger(keras.callbacks.Callback):
        """Custom callback to print out training loss per batch."""
        def __init__(self):
            super(BatchLossLogger, self).__init__()
            self.epoch = -1
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch + 1
        def on_train_batch_end(self, batch, logs=None):
            print(f'Epoch {self.epoch} batch {batch}: training loss = {logs["loss"]:.6e} ({time.strftime("%c")})')

    # Save losses to file per epoch
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    train_loss_log = f'{checkpoint_path}/train_loss.txt'
    val_loss_log = f'{checkpoint_path}/val_loss.txt'
    with open(train_loss_log, 'w') as f:
        print('# train_loss', file=f)
    with open(val_loss_log, 'w') as f:
        print('# val_loss', file=f)
    class EpochLossLogger(keras.callbacks.Callback):
        """Custom callback to save losses to file per epoch."""
        def on_epoch_end(self, epoch, logs=None):
            with open(train_loss_log, 'a') as f:
                print(f'{logs["loss"]:.10e}', file=f)
            with open(val_loss_log, 'a') as f:
                print(f'{logs["val_loss"]:.10e}', file=f)

    # Train
    opt = keras.optimizers.Adam(lr=1e-4, clipnorm=1)
    callbacks = [BatchLossLogger(), checkpoint, EpochLossLogger(), keras.callbacks.TerminateOnNaN()]
    model.compile(optimizer=opt, loss='mean_squared_error')
    history = model.fit(train_xy, epochs=epochs, verbose=2, validation_data=val_xy, callbacks=callbacks)

    # Print best val loss and which epoch it was
    best_epoch_idx = np.argmin(history.history['val_loss'])
    print(f'Done. Best epoch was {best_epoch_idx + 1}',
          f'with val_loss = {history.history["val_loss"][best_epoch_idx]:.10f}.')


def determine_model_and_train():
    """
    Train a given model depending on the command line argument.
    """

    print(f'Beginning {__file__} function main at {time.strftime("%c")}')
    model_name = sys.argv[1]
    print(f'Requested model name is: {model_name}')

    # Input parameters
    train_mask = 'xyz/v6/train_10000.npz'
    n_training_files = 1
    batches_per_file = 10000 // 10
    val_path = 'xyz/v6/val_200.npz'
    x_arr_id = 'lensed_t_maps'
    y_arr_id = 'k_maps'

    # Training parameters
    epochs = 100
    checkpoint_dir = f'xyz/v6/{epochs}epoch_10k/'

    # Define the models
    keras.backend.clear_session()
    models = {
        '12x32': {
            'n_main_layers': 12,
            'nodes_per_layer': 32
        },
        '24x16': {
            'n_main_layers': 24,
            'nodes_per_layer': 16
        },
        '48x8': {
            'n_main_layers': 48,
            'nodes_per_layer': 8
        }
    }
    model = models[model_name]
    n_main_layers = model['n_main_layers']
    nodes_per_layer = model['nodes_per_layer']
    model = keras.Sequential(name=model_name)
    conv_args = {
        'activation': 'relu',
        'kernel_initializer': 'Orthogonal',
        'padding': 'same',
    }
    model.add(keras.layers.InputLayer(input_shape=(100, 100, 1)))
    _ = [model.add(keras.layers.Conv2D(nodes_per_layer, 3, **conv_args)) for _ in range(n_main_layers)]
    model.add(keras.layers.Conv2D(1, 3, **conv_args))
    model.summary()

    # Prepare training and validation data
    train_xy = TrainingGenerator(train_mask, x_arr_id, y_arr_id, n_training_files, batches_per_file)
    val_xy = load_validation_data(val_path, x_arr_id, y_arr_id, shuffle=True)

    # Run the model
    print(f'\nStarting training at {time.strftime("%c")}\n')
    train_model(model, train_xy, val_xy, epochs, checkpoint_dir + model_name + '_best')
    print(f'\nFinished training at {time.strftime("%c")}\n')

    print(f'Done at {time.strftime("%c")}')


def test(model):
    """
    Test trained model on unseen test set.
    """

    # Params for the plot
    lx_deg = 10
    npix = 100
    n_lbin = 30

    # Load test data
    test_path = 'xyz/v6/test_3.npz'
    x_arr_id = 'lensed_t_maps'
    y_arr_id = 'k_maps'
    test_x, test_y = load_validation_data(test_path, x_arr_id, y_arr_id, shuffle=False)

    # Make prediction
    test_prediction = model.predict(test_x, verbose=1)

    # Plot
    _, ax = plt.subplots(nrows=3, ncols=3, figsize=2*plt.figaspect(1), sharex=True, sharey=True)
    plt.subplots_adjust(left=.05, bottom=0.05, right=.95, top=.95, wspace=.05, hspace=.05)
    imshow_args = {'vmin': 0, 'vmax': 1, 'extent': [0, lx_deg, 0, lx_deg], 'interpolation': 'none'}
    for row, x, y_est, y_truth in zip(ax, test_x, test_prediction, test_y):
        row[0].imshow(np.squeeze(x), **imshow_args)
        row[1].imshow(np.squeeze(y_est), **imshow_args)
        row[2].imshow(np.squeeze(y_truth), **imshow_args)
    _ = [a.set_major_formatter("${x:.0f}^\\circ$") for a in (ax[0, 0].xaxis, ax[0, 0].yaxis)]
    _ = [col.set_title(title) for col, title in zip(ax[0], ['Input', 'Output', 'Truth'])]
    plt.show()

    # Attempt to measure power spectra
    pixel_size_deg = lx_deg / npix
    k_lmin_obs = int(1.5 * 180 / lx_deg)
    k_lmax_obs = int(2/3 * 180 / pixel_size_deg)
    lx_rad = np.radians(lx_deg)
    mask = np.ones((npix, npix))
    bin_edges = np.linspace(k_lmin_obs, k_lmax_obs, n_lbin + 1)
    bins = nmt.bins.NmtBinFlat(bin_edges[:-1], bin_edges[1:])
    ell_obs = bins.get_effective_ells()
    cl_fac_obs = ell_obs * (ell_obs + 1) / (2 * np.pi)
    _, ax = plt.subplots(nrows=3, figsize=2*plt.figaspect(1), sharex=True)
    for a, y_est, y_truth in zip(ax, test_prediction, test_y):
        field_est = nmt.field.NmtFieldFlat(lx_rad, lx_rad, mask, [np.squeeze(y_est)])
        field_truth = nmt.field.NmtFieldFlat(lx_rad, lx_rad, mask, [np.squeeze(y_truth)])
        obs_cl_est = np.squeeze(nmt.workspaces.compute_coupled_cell_flat(field_est, field_est, bins))
        obs_cl_truth = np.squeeze(nmt.workspaces.compute_coupled_cell_flat(field_truth, field_truth, bins))
        a.step(bin_edges, np.pad(cl_fac_obs * obs_cl_est, (0, 1), mode='edge'), where='post',
               label='Measured from CNN estimate')
        a.step(bin_edges, np.pad(cl_fac_obs * obs_cl_truth, (0, 1), mode='edge'), where='post',
               label='Measured from true map')
        a.legend(frameon=False, loc='upper left')
        a.set_ylabel(r'$C_\ell \times \ell (\ell + 1) / 2 \pi$')
    plt.xlabel(r'$\ell$')
    plt.show()


def test_from_file():
    """
    Load model from file and test it.
    """

    model_path = 'xyz'
    model = keras.models.load_model(model_path)
    test(model)
