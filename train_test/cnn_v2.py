"""
CNN script for v2 of the lensing estimation testing.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def augment_data(xy, reshuffle=True):
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

    # Concatenate and shuffle
    xy = np.concatenate(xy_all, axis=0)
    if reshuffle:
        print('Reshuffling')
        np.random.default_rng().shuffle(xy, axis=0)

    return xy


def prepare_data(x_mask, y_mask, val_frac=0.2, scale=1, shuffle=True, lim=None, augment=True):
    """
    Returns x, y and validation_data.
    Everything is scaled consistently and shuffled consistently.

    Requires that x and y are consistently numbered continuously fom 0.
    x_mask and y_mask should be "/path/to/prefix_{i}.extension"

    val_frac is the fraction of the training data theat will be set aside for validation prior to augmentation
    (which multiplies the size of the remaining training data by 8).
    """

    # Load all x and y images together (up to lim if supplied)
    x_imgs = []
    y_imgs = []
    i = 0
    while lim is None or i < lim:
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

    if i == 0:
        raise ValueError('No training images found')

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

    # augment the training only,
    # so as not to introduce possible biases into the validation
    if augment:
        print('Augmenting training set')
        xy_train = augment_data(xy_train, reshuffle=shuffle)
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
    train_dir = 'xyz/input_png/v2/'
    x_mask = train_dir + 'lensed/lensed_{i}.png'
    y_mask = train_dir + 'kappa/kappa_{i}.png'
    val_frac = 0.2
    scale = 1./255

    # Training parameters
    epochs = 50
    checkpoint_dir = 'xyz/'

    # Current best model
    conv_args = {
        'activation': 'relu',
        'kernel_initializer': 'Orthogonal',
        'padding': 'same',
    }
    model = keras.Sequential([
        keras.Input(shape=(50, 50, 1)),
        keras.layers.Conv2D(64, 3, **conv_args),
        keras.layers.Conv2D(64, 3, **conv_args),
        keras.layers.Conv2D(64, 3, **conv_args),
        keras.layers.Conv2D(64, 3, **conv_args),
        keras.layers.Conv2D(1, 9, **conv_args)
    ])
    model.summary()

    # Checkpoint to save weights every time val loss is improved upon
    save_path = checkpoint_dir + f'64_3_64_3_64_3_64_3_1_9_best_{epochs}epochs'
    checkpoint = keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True,
                                                 mode='min', save_weights_only=False, save_freq='epoch')

    # Load training and validation data
    train_x, train_y, val_xy = prepare_data(x_mask, y_mask, val_frac, scale, shuffle=True)

    # Train
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(train_x, train_y, epochs=epochs, verbose=2, validation_data=val_xy, callbacks=[checkpoint])

    plot_history(history, epochs)

    # Evaluate on unseen test set
    test(model)


def plot_history(history, epochs):
    """
    Plot history for a given number of epochs.
    """

    epoch = np.arange(1, epochs + 1)
    plt.plot(epoch, history.history['loss'], label='Training loss')
    plt.plot(epoch, history.history['val_loss'], label='Validation loss')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def test(model):
    """
    Test trained model on unseen test set.
    """

    lx_rad = 0.39973699529159706
    lx_deg = np.degrees(lx_rad)

    test_x_mask = 'xyz/v2/test/lensed/lensed_{i}.png'
    test_y_mask = 'xyz/v2/test/kappa/kappa_{i}.png'
    test_x, test_y, _ = prepare_data(test_x_mask, test_y_mask, 0, 1./255, shuffle=False, augment=False)
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


def test_from_file():
    """
    Load model from file and test it.
    """

    model_path = ('xyz')
    model = keras.models.load_model(model_path)
    test(model)
