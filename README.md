# cnn_lensing: Weak lensing estimation using convolutional neural networks

## Introduction

This is an unfinished project I worked on during my PhD, which was eventually deprioritised due to time constraints.

The idea was to explore the possibility of machine learning (specifically convolutional neural networks, CNNs) for weak lensing estimation. The long-term goal was for this to be useful for radio weak lensing using the Square Kilometre Array, to be able to directly estimate shear from radio visibilities, therefore bypassing the imaging stage which can introduce biases. However, all the tests here were for the simpler case of CMB lensing.

I started with a very simple case and was gradually adding complexity, which is why for example there are eight very similar scripts in the `train_test` directory. A summary of the eight versions is below.

Initially I was using the [LensTools](https://lenstools.readthedocs.io/en/latest/) package by Andrea Petri for simulating lensing. I ran into some problems trying to install LensTools on a university machine (due to its dependencies), so later extracted the core functionality that I needed into `lenstools_light`, which only requires NumPy and SciPy.


## Summary of the eight versions

- v1: `npix` = 50, `nside_equiv` = 128, `T_lmax` = 383, `kappa_lmax` = 50, `kappa_exaggeration_factor` = 30, same unlensed T map realisation for every sample

- v2: as v1 but different unlensed T map realisation for each sample

- v3: as v2 but `kappa_exaggeration_factor` = 10

- v4: as v3 but `kappa_exaggeration_factor` = 1

- v5: as v4 but `nside_equiv` = 4096, `T_lmax` = 12287, `kappa_lmax` = 5000

- v6: `npix` = 100, `lx` = 10 deg, no exaggeration or `lmax`

- v7: `npix` = 50, `lx` = 22.9 deg, no exaggeration or `lmax`

- v8: `npix` = 50, `lx` = 22.9 deg, no exaggeration, `kappa_lmax` = 50, noise of rms 5 μK arcmin

### Explanation of terms used above

- `npix`: Number of pixels across each side (all images square)

- `lx`: Angular size of patch across each side

- `nside_equiv`: A way of specifying resolution by approximately equivalent `nside` in HEALPix. I later switched to specifying both `npix` and `lx` directly

- `T_lmax`: Maximum multipole used in input CMB TT power spectrum

- `kappa_lmax`: Maximum multipole used in input convergence power spectrum

- `kappa_exaggeration_factor`: Factor by which a convergence map is multiplied prior to the lensing simulation


## List of files

### `data_generation`: generation of training, validation and test data

- `generate_flat_maps.py`: Functions to generate flat-sky CMB T map or convergence map realisations using NaMaster. This was used for versions 1&ndash;5, after which there are specific per-version data generation scripts.

- `apply_lens.py`: Lens T maps with convergence maps using LensTools, with options for saving as png or npz. This was also used for versions 1&ndash;5 only.

- `v6_train_set.py`: Generate training, validation and test data from scratch for version 6.

- `v7_train_set.py`: Generate training, validation and test data from scratch for version 7.

- `v8_train_set.py`: Generate training, validation and test data from scratch for version 8.

### `train_test`: CNN model definitions, training and testing

- `cnn_v1.py`: Keras model definition, training and testing for version 1.

- `cnn_v[2-8].py`: As above for versions 2 to 8.

### `lenstools_light`: Simple CMB lensing code extracted from [LensTools](https://lenstools.readthedocs.io/en/latest/) by Andrea Petri

- `lenstools_light.py`: Functions to lens a square CMB T map using either a convergence or deflection potential map.

### `util`: miscellaneous utility functions

- `mem_usage.py`: Function to get the memory requirements of a Keras model, taken from https://stackoverflow.com/a/46216013.

- `plot_loss.py`: Plot training and validation loss for a selection of models.

### `dev_testing`: experimentation

- `augmentation_test.py`: Testing image augmentation techniques.

- `cmb_lensing.py`: Testing simulating CMB lensing 'manually', i.e. not using LensTools or `lenstools_light`.

- `lenstools_test.py`: Testing `lenstools_light` against the real LensTools.

- `phi_vs_kappa.py`: Testing for consistency between the `lenstools_light` functions accepting convergence and deflection potential maps.

- `resolution_test.py`: Investigating the effect of resolution in the lensing simulations.

- `sequence_test.py`: Developing loading training data using a Keras `Sequence` object.


## Dependencies

### Core dependencies

- numpy

- scipy

- [NaMaster](https://namaster.readthedocs.io/en/latest/installation.html)

- [TensorFlow 2](https://www.tensorflow.org/install)

### Non-essential dependencies

- [astropy](https://docs.astropy.org/en/stable/install.html) &ndash; only for units, which could be handled manually

- [CAMB](https://camb.readthedocs.io/en/latest/) &ndash; to generate TT and φφ power spectra, which could be obtained elsewhere

- [CCL](https://ccl.readthedocs.io/en/latest/) &ndash; to generate κκ power spectra, which could be obtained elsewhere (e.g. from CAMB φφ spectra)

- [healpy](https://healpy.readthedocs.io/en/latest/) &ndash; only for resolution specified as `npix_equiv` in earlier scripts, later replaced by directly specifying resolution in number of pixels and angular size of patch

- matplotlib &ndash; for plotting or saving training data as png

- [LensTools](https://lenstools.readthedocs.io/en/latest/) &ndash; can be replaced with equivalent calls to `lenstools_light`
