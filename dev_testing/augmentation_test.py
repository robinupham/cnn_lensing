"""
Testing data augmentation composed of rotations and reflections.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import tensorflow.keras as keras


def with_numbers():
    """
    Test with a simple numbered array.

    Note that this uses scipy.ndimage.rotate for rotations, but I later switched to the simpler numpy.rot90.
    """

    x = np.array([[1, 2],
                  [3, 4]])

    # Rotate 90: 31
    #            42
    rotate90 = scipy.ndimage.rotate(x, -90, axes=(0, 1), reshape=False, order=1, mode='constant', cval=np.nan)
    print(rotate90)
    print()

    # Rotate 180: 43
    #             21
    rotate180 = scipy.ndimage.rotate(x, -180, axes=(0, 1), reshape=False, order=1, mode='constant', cval=np.nan)
    print(rotate180)
    print()

    # Rotate 270: 24
    #             13
    rotate270 = scipy.ndimage.rotate(x, -270, axes=(0, 1), reshape=False, order=1, mode='constant', cval=np.nan)
    print(rotate270)
    print()

    # Flip vert: 34
    #            12
    flipvert = np.flip(x, axis=0)
    print(flipvert)
    print()

    # Flip horiz: 21
    #             43
    fliphoriz = np.flip(x, axis=1)
    print(fliphoriz)
    print()

    # Rotate 90 + flip vert: 42
    #                        31
    rotate90_flipvert = np.flip(rotate90, axis=0)
    print(rotate90_flipvert)
    print()

    # Rotate 90 + flip horiz: 13
    #                         24
    rotate90_fliphoriz = np.flip(rotate90, axis=1)
    print(rotate90_fliphoriz)


def with_image():
    """
    Test with an image.
    """

    _, ax = plt.subplots(nrows=2, ncols=4)

    # Load test image
    test_img_path = 'augmentation_test_img.png'
    img = keras.preprocessing.image.load_img(test_img_path)
    img = keras.preprocessing.image.img_to_array(img)
    ax[0, 0].imshow(img.astype(int))
    ax[0, 0].set_title('Original')

    # Rotate 90
    rotate90 = np.rot90(img, 1, axes=(1, 0))
    ax[0, 1].imshow(rotate90.astype(int))
    ax[0, 1].set_title('Rotate 90')

    # Rotate 180
    rotate180 = np.rot90(img, 2, axes=(1, 0))
    ax[0, 2].imshow(rotate180.astype(int))
    ax[0, 2].set_title('Rotate 180')

    # Rotate 270
    rotate270 = np.rot90(img, 3, axes=(1, 0))
    ax[0, 3].imshow(rotate270.astype(int))
    ax[0, 3].set_title('Rotate 270')

    # Flip vert
    flipvert = np.flip(img, axis=0)
    assert np.all(flipvert >= 0)
    ax[1, 0].imshow(flipvert.astype(int))
    ax[1, 0].set_title('Flip vertical')

    # Flip horiz
    fliphoriz = np.flip(img, axis=1)
    assert np.all(fliphoriz >= 0)
    ax[1, 1].imshow(fliphoriz.astype(int))
    ax[1, 1].set_title('Flip horizontal')

    # Rotate 90 + flip vert
    rotate90_flipvert = np.flip(rotate90, axis=0)
    assert np.all(rotate90_flipvert >= 0)
    ax[1, 2].imshow(rotate90_flipvert.astype(int))
    ax[1, 2].set_title('Rotate 90\n+ flip vertical')

    # Rotate 90 + flip horiz
    rotate90_fliphoriz = np.flip(rotate90, axis=1)
    assert np.all(rotate90_fliphoriz >= 0)
    ax[1, 3].imshow(rotate90_fliphoriz.astype(int))
    ax[1, 3].set_title('Rotate 90\n+ flip horizontal')

    plt.setp(ax, xticks=[], yticks=[])

    plt.show()
