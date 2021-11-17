"""
CNN script for v1 of the lensing estimation testing.
"""

import os

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import pymaster as nmt
import tensorflow.keras as keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def prepare_data(x_mask, y_mask, val_frac=0.2, scale=1, shuffle=True):
    """
    Returns x, y and validation_data.
    Everything is scaled consistently and shuffled consistently.

    Requires that x and y are consistently numbered continuously fom 0.
    x_mask and y_mask should be "/path/to/prefix_{i}.extension"
    """

    # Load all x and y images together
    x_imgs = []
    y_imgs = []
    i = 0
    while True:
        x_path = x_mask.format(i=i)
        y_path = y_mask.format(i=i)
        try:
            x_img = keras.preprocessing.image.load_img(x_path, color_mode='grayscale')
            y_img = keras.preprocessing.image.load_img(y_path, color_mode='grayscale')
        except FileNotFoundError:
            break
        x_imgs.append(x_img)
        y_imgs.append(y_img)
        i += 1
        print(f'Loaded {i} images', end='\r')
    print()

    # Convert to a single numpy array
    x_imgs = list(map(keras.preprocessing.image.img_to_array, x_imgs))
    y_imgs = list(map(keras.preprocessing.image.img_to_array, y_imgs))
    xy = np.stack((x_imgs, y_imgs), axis=1)

    # Apply scaling
    print('Scaling')
    xy *= scale
    assert np.amin(xy) >= 0
    assert np.amax(xy) <= 1

    # Shuffle
    if shuffle:
        print('Shuffling')
        np.random.default_rng().shuffle(xy, axis=0)

    # Split into training and validation
    print('Splitting')
    n_val = int(val_frac * xy.shape[0])
    xy_val = xy[:n_val, ...]
    xy_train = xy[n_val:, ...]
    print('Training size:', xy_train.shape[0], 'Validation size:', xy_val.shape[0])

    # Return x_train, y_train, (x_val, y_val)
    x_train = xy_train[:, 0, ...]
    y_train = xy_train[:, 1, ...]
    x_val = xy_val[:, 0, ...]
    y_val = xy_val[:, 1, ...]
    return x_train, y_train, (x_val, y_val)


def main():
    """Main function"""

    # Input parameters
    train_dir = 'xyz/'
    x_mask = train_dir + 'lensed/lensed/lensed_{i}.png'
    y_mask = train_dir + 'kappa/kappa/kappa_{i}.png'
    val_frac = 0.2
    scale = 1./255

    # Based on the super-resolution model at https://keras.io/examples/vision/super_resolution_sub_pixel/
    conv_args = {
        'activation': 'relu',
        'kernel_initializer': 'Orthogonal',
        'padding': 'same',
    }
    model = keras.Sequential([
        keras.Input(shape=(50, 50, 1)),
        keras.layers.Conv2D(64, 3, **conv_args),
        keras.layers.Conv2D(64, 3, **conv_args),
        keras.layers.Conv2D(32, 3, **conv_args),
        keras.layers.Conv2D(1, 9, **conv_args)
    ])
    model.summary()

    # Checkpoint to save weights every time val loss is improved upon
    save_path = 'xyz'
    checkpoint = keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True,
                                                 mode='min', save_weights_only=False, save_freq='epoch')

    # Load training and validation data
    train_x, train_y, val_xy = prepare_data(x_mask, y_mask, val_frac, scale, shuffle=True)

    # Train
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(train_x, train_y, epochs=5, verbose=1, validation_data=val_xy, callbacks=[checkpoint])

    # Evaluate on unseen test set
    test(model)


def examine_power_spectra(y_est, y_truth):
    """
    Examine power spectra using NaMaster.
    """

    # Parameters
    nside_equiv = 128
    npix = 50
    lmin = 2
    lmax = 50
    nbin = 10

    # Squeeze out the final axis and add an initial axis if not present
    y_est = np.squeeze(y_est)
    y_truth = np.squeeze(y_truth)
    if y_est.ndim == 2:
        y_est = y_est[np.newaxis, ...]
        y_truth = y_truth[np.newaxis, ...]
    assert y_est.shape == y_truth.shape
    assert y_est.ndim == 3
    assert y_est.shape[1:] == (npix, npix)

    # Prepare things needed by NaMaster
    lx = npix * np.sqrt(hp.nside2pixarea(nside_equiv))
    mask = np.ones((npix, npix))
    bin_edges = np.linspace(lmin, lmax, nbin +1)
    l0 = bin_edges[:-1]
    lf = bin_edges[1:]
    bins = nmt.bins.NmtBinFlat(l0, lf)
    ell = bins.get_effective_ells()

    for est, truth in zip(y_est, y_truth):

        # Create NaMaster fields
        field_tru = nmt.field.NmtFieldFlat(lx, lx, mask, [truth])
        field_est = nmt.field.NmtFieldFlat(lx, lx, mask, [est])

        cl_tru = np.squeeze(nmt.workspaces.compute_coupled_cell_flat(field_tru, field_tru, bins))
        cl_est = np.squeeze(nmt.workspaces.compute_coupled_cell_flat(field_est, field_est, bins))

        plt.figure()
        plt.plot(ell, cl_tru)
        plt.plot(ell, cl_est)
        plt.show()


def test(model, power_spectra=False):
    """
    Test trained model on unseen test set.
    """

    lx_rad = 0.39973699529159706
    lx_deg = np.degrees(lx_rad)

    test_x_mask = 'xyz/v1/test/lensed/lensed_{i}.png'
    test_y_mask = 'xyz/v1/test/kappa/kappa_{i}.png'
    test_x, test_y, _ = prepare_data(test_x_mask, test_y_mask, 0, 1./255, shuffle=False)
    test_prediction = model.predict(test_x, verbose=1)
    _, ax = plt.subplots(nrows=3, ncols=3, figsize=2*plt.figaspect(1), sharex=True, sharey=True)
    plt.subplots_adjust(left=.05, bottom=0.05, right=.95, top=.95, wspace=.05, hspace=.05)
    imshow_args = {'vmin': 0, 'vmax': 1, 'extent': [0, lx_deg, 0, lx_deg], 'interpolation': 'none'}
    for row, x, y_est, y_truth in zip(ax, test_x, test_prediction, test_y):
        row[0].imshow(np.squeeze(x), **imshow_args)
        row[1].imshow(np.squeeze(y_est), **imshow_args)
        row[2].imshow(np.squeeze(y_truth), **imshow_args)
    _ = [a.set_major_formatter('${x:.0f}^\\circ$') for a in (ax[0, 0].xaxis, ax[0, 0].yaxis)]
    _ = [col.set_title(title) for col, title in zip(ax[0], ['Input', 'Output', 'Truth'])]
    plt.show()

    # Also examine power spectra using NaMaster (doesn't work!)
    if power_spectra:
        examine_power_spectra(test_prediction, test_y)


def test_from_file():
    """
    Load model from file and test it.
    """

    model_path = 'xyz'
    model = keras.models.load_model(model_path)
    test(model)


def test_from_sep_files():
    """
    Load model and weights separately and test.
    """

    model_path = 'xyz'
    weights_path = 'xyz.hdf5'
    model = keras.models.load_model(model_path)
    model.load_weights(weights_path)
    test(model)
