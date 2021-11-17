"""
CNN script for v7 of the lensing estimation testing.
"""

import os
import pathlib
import sys
import time

# pylint: disable=wrong-import-position
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt
import tensorflow.keras as keras


def augment_data(xy):
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
    assert np.all(np.isfinite(xy_rotate90))
    assert xy_rotate90.shape == shape

    # Rotate 180
    xy_rotate180 = np.rot90(xy, 2, axes=(3, 2))
    assert np.all(np.isfinite(xy_rotate180))
    assert xy_rotate180.shape == shape

    # Rotate 270
    xy_rotate270 = np.rot90(xy, 3, axes=(3, 2))
    assert np.all(np.isfinite(xy_rotate270))
    assert xy_rotate270.shape == shape

    # Flip vert
    xy_flipvert = np.flip(xy, axis=2)
    assert np.all(np.isfinite(xy_flipvert))
    assert xy_flipvert.shape == shape

    # Flip horiz
    xy_fliphoriz = np.flip(xy, axis=3)
    assert np.all(np.isfinite(xy_fliphoriz))
    assert xy_fliphoriz.shape == shape

    # Rotate 90 + flip vert
    xy_rotate90_flipvert = np.flip(xy_rotate90, axis=2)
    assert np.all(np.isfinite(xy_rotate90_flipvert))
    assert xy_rotate90_flipvert.shape == shape

    # Rotate 90 + flip horiz
    xy_rotate90_fliphoriz = np.flip(xy_rotate90, axis=3)
    assert np.all(np.isfinite(xy_rotate90_fliphoriz))
    assert xy_rotate90_flipvert.shape == shape

    # Do a check for no duplication
    xy_all = (xy, xy_rotate90, xy_rotate180, xy_rotate270, xy_flipvert, xy_fliphoriz, xy_rotate90_flipvert,
              xy_rotate90_fliphoriz)
    for xy1_idx, xy1 in enumerate(xy_all):
        for xy2 in xy_all[:xy1_idx]:
            assert not np.allclose(xy1, xy2)

    # Concatenate and return
    xy = np.concatenate(xy_all, axis=0)
    return xy


def load_npz(npz_path, x_arr_id, y_arr_id, shuffle=True, augment=True, lim=None):
    """
    Loads training (or validation or test) data from a single npz file where x and y are in data[x_arr_id]
    and data[y_arr_id].

    Will shuffle and augment everything consistently, keeping the x-y correspondence, so they must be
    consistently ordered to begin with.

    lim is applied before augmentation, which multiplies the remaining data by 8.
    """

    # Load the data (up to lim if supplied)
    # and stack into a single array of shape (n_samp, n_pix, n_pix, n_channel)
    with np.load(npz_path) as data:
        x = data[x_arr_id][:lim]
        y = data[y_arr_id][:lim]
    xy = np.stack((x, y), axis=1)

    # Add the channel axis if missing
    n_samp = xy.shape[0]
    npix = xy.shape[2]
    shape = (n_samp, 2, npix, npix, 1)
    if xy.shape != shape:
        assert xy.shape == shape[:-1]
        xy = xy[..., np.newaxis]
    assert xy.shape == shape, xy.shape

    print('Number of samples pre-augmentation:', n_samp)

    # Augment
    if augment:
        print('Augmenting')
        xy = augment_data(xy)

    print('Number of samples post-augmentation:', xy.shape[0])

    # Shuffle
    if shuffle:
        print('Shuffling')
        np.random.default_rng().shuffle(xy, axis=0)

    # Do some final checks and return x, y
    x = xy[:, 0, ...]
    y = xy[:, 1, ...]
    n_samp = xy.shape[0]
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    assert np.amin(x) > 0, np.amin(x)
    assert np.amax(x) < 1, np.amax(x)
    assert np.amin(y) > 0, np.amin(y)
    assert np.amax(y) < 1, np.amax(y)
    assert x.shape == (n_samp, npix, npix, 1)
    assert y.shape == (n_samp, npix, npix, 1)
    return x, y


def train_model(model, train_x, train_y, val_xy, epochs, checkpoint_path):
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
    opt = keras.optimizers.Adam(lr=1e-3, clipnorm=1)
    callbacks = [BatchLossLogger(), checkpoint, EpochLossLogger(), keras.callbacks.TerminateOnNaN()]
    model.compile(optimizer=opt, loss='mean_squared_error')
    history = model.fit(train_x, train_y, batch_size=128, epochs=epochs, verbose=2, validation_data=val_xy,
                        callbacks=callbacks)

    # Print best val loss and which epoch it was
    best_epoch_idx = np.argmin(history.history['val_loss'])
    print(f'Done. Best epoch was {best_epoch_idx + 1}',
          f'with val_loss = {history.history["val_loss"][best_epoch_idx]:.10f}.')


def determine_model_and_train(model_name=None):
    """
    Train a given model depending on the command line argument.
    """

    print(f'Beginning {__file__} function main at {time.strftime("%c")}')
    if model_name is None:
        model_name = sys.argv[1]
    print(f'Requested model name is: {model_name}')

    # Input parameters
    train_path = 'xyz/v7/train_20k.npz'
    val_path = 'xyz/v7/val_200.npz'
    x_arr_id = 'lensed_t_maps'
    y_arr_id = 'k_maps'
    npix = 50

    # Training parameters
    epochs = 100
    checkpoint_dir = f'xyz/v7/{epochs}epoch_20k/'

    # Define the models
    keras.backend.clear_session()
    models = {
        '6x64-9': {
            'n_main_layers': 6,
            'nodes_per_layer': 64,
            'final_kernel_size': 9
        },
        '6x64-3': {
            'n_main_layers': 6,
            'nodes_per_layer': 64,
            'final_kernel_size': 3
        },
        '8x64-9': {
            'n_main_layers': 8,
            'nodes_per_layer': 64,
            'final_kernel_size': 9
        },
        '8x64-3': {
            'n_main_layers': 8,
            'nodes_per_layer': 64,
            'final_kernel_size': 3
        },
        '10x64-9': {
            'n_main_layers': 10,
            'nodes_per_layer': 64,
            'final_kernel_size': 9
        },
        '10x64-3': {
            'n_main_layers': 10,
            'nodes_per_layer': 64,
            'final_kernel_size': 3
        }
    }
    model = models[model_name]
    n_main_layers = model['n_main_layers']
    nodes_per_layer = model['nodes_per_layer']
    final_kernel_size = model['final_kernel_size']
    model = keras.Sequential(name=model_name)
    conv_args = {
        'activation': 'relu',
        'kernel_initializer': 'Orthogonal',
        'padding': 'same',
    }
    model.add(keras.layers.InputLayer(input_shape=(npix, npix, 1)))
    _ = [model.add(keras.layers.Conv2D(nodes_per_layer, 3, **conv_args)) for _ in range(n_main_layers)]
    model.add(keras.layers.Conv2D(1, final_kernel_size, **conv_args))
    model.summary()

    # Prepare training and validation data
    train_x, train_y = load_npz(train_path, x_arr_id, y_arr_id, shuffle=True, augment=True)
    val_xy = load_npz(val_path, x_arr_id, y_arr_id, shuffle=True, augment=False)

    # Run the model
    print(f'\nStarting training at {time.strftime("%c")}\n')
    train_model(model, train_x, train_y, val_xy, epochs, checkpoint_dir + model_name + '_best')
    print(f'\nFinished training at {time.strftime("%c")}\n')

    print(f'Done at {time.strftime("%c")}')


def loop_models():
    """Loop over models"""

    models = ['6x64-9', '6x64-3', '8x64-9', '8x64-3', '10x64-9', '10x64-3']
    for model in models:
        determine_model_and_train(model_name=model)


def test(model):
    """
    Test trained model on unseen test set.
    """

    # Params for the plot
    lx_deg = 22.9
    npix = 50
    n_lbin = 20

    # Load test data
    test_path = 'xyz/v7/test_3.npz'
    x_arr_id = 'lensed_t_maps'
    y_arr_id = 'k_maps'
    test_x, test_y = load_npz(test_path, x_arr_id, y_arr_id, shuffle=False, augment=False)

    # Make prediction
    test_prediction = model.predict(test_x, verbose=1)

    # Plot
    _, ax = plt.subplots(nrows=3, ncols=4, figsize=2*plt.figaspect(1.25), sharex=True, sharey=True)
    plt.subplots_adjust(left=.05, bottom=0.05, right=.95, top=.95, wspace=.05, hspace=.05)
    imshow_args = {'vmin': 0, 'vmax': 1, 'extent': [0, lx_deg, 0, lx_deg], 'interpolation': 'none'}
    for row, x, y_est, y_truth in zip(ax, test_x, test_prediction, test_y):
        row[0].imshow(np.squeeze(x), **imshow_args)
        row[1].imshow(np.squeeze(y_est), **imshow_args)
        row[2].imshow(np.squeeze(y_truth), **imshow_args)
        row[3].imshow(np.squeeze(y_est) - np.squeeze(y_truth), **imshow_args)
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
