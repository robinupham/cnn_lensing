"""
Generate training, validation and test sets for v7 of the CNN lensing estimation project.

v8 is v4 plus noise, so the following spec:
    50 pixels
    22.9 deg field of view
    kappa lmax 50
    no T lmax (effectively)
    Gaussian noise with rms 5 μK arcmin
"""

import time

import astropy.units as units
import camb
import lenstools.image.convergence as lt
# import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
import pymaster as nmt


NPIX = 50
LX_DEG = 22.9
KAPPA_LMAX = 50
NOISE_RMS_MUK_ARCMIN = 5

N_REAL_TRAIN = 9800
N_REAL_VAL = 200
N_REAL_TEST = 3
AUTO_SCALE = False
T_SCALING = 515.0 # max abs value for normalisation, which only needs supplying if AUTO_SCALE = False
K_SCALING = 0.0387
SAVE_PATH = 'path_to_save/{set_name}_{n_real}.npz'


def create_train_val_test_sets():
    """
    Create training, validation and test sets.
    """

    # Choose T lmax to more than max out the available resolution
    pixel_size_deg = LX_DEG / NPIX
    t_lmax = int(2 * 180 / pixel_size_deg)

    # Calculate CMB temperature power spectrum in μK^2, with no prefactors
    camb_params = camb.CAMBparams()
    camb_params.set_cosmology(H0=67)
    camb_params.set_for_lmax(2 * t_lmax)
    res = camb.get_results(camb_params)
    t_cl_in = res.get_cmb_power_spectra(lmax=t_lmax, spectra=['unlensed_scalar'], CMB_unit='muK',
                                        raw_cl=True)['unlensed_scalar'][:, 0]

    # Calculate kappa Cls using CCL
    ccl_cosmo = ccl.core.CosmologyVanillaLCDM()
    z_source = 1100
    tracer = ccl.tracers.CMBLensingTracer(ccl_cosmo, z_source)
    ell_in = np.arange(KAPPA_LMAX + 1)
    k_cl_in = ccl.angular_cl(ccl_cosmo, tracer, tracer, ell_in)

    # Generate realisations
    rng = np.random.default_rng()
    n_real_tot = N_REAL_TRAIN + N_REAL_VAL + N_REAL_TEST
    lensed_t_maps = np.full((n_real_tot, NPIX, NPIX), np.nan)
    k_maps = lensed_t_maps.copy()
    start_time = time.time()
    for real_idx in range(n_real_tot):
        print(f'Generating realisation {real_idx + 1} / {n_real_tot}', end='\r')

        # Generate CMB and kappa realisations
        lx_rad = np.radians(LX_DEG)
        unlensed_t_data = np.squeeze(nmt.utils.synfast_flat(NPIX, NPIX, lx_rad, lx_rad, [t_cl_in], [0]))
        k_data = np.squeeze(nmt.utils.synfast_flat(NPIX, NPIX, lx_rad, lx_rad, [k_cl_in], [0]))

        # Initialise LensTools maps
        lx = LX_DEG * units.degree
        unlensed_t_map = lt.CMBTemperatureMap(unlensed_t_data, lx, space='real', unit=units.microKelvin)
        k_map = lt.ConvergenceMap(k_data, lx)

        # Lens
        lensed_t_map = unlensed_t_map.lens(k_map)
        lensed_t_data = lensed_t_map.data

        # # Plot
        # _, ax = plt.subplots(ncols=4, figsize=(2 * plt.figaspect(1 / 4)))
        # ax[0].imshow(unlensed_t_map.data)
        # ax[1].imshow(lensed_t_map.data)
        # ax[2].imshow(lensed_t_map.data - unlensed_t_map.data)
        # ax[3].imshow(k_map.data)
        # plt.show()

        # Add noise
        pixel_size_arcmin = 60 * pixel_size_deg
        noise_rms = NOISE_RMS_MUK_ARCMIN / pixel_size_arcmin
        noise = rng.normal(loc=0, scale=noise_rms, size=(NPIX, NPIX))
        # lensed_t_nonoise = lensed_t_data.copy() # for plotting below
        lensed_t_data += noise

        # # Plot lensing signal and noise
        # lensing_signal = lensed_t_nonoise - unlensed_t_data
        # total_resid = lensed_t_data - unlensed_t_data
        # fig, ax = plt.subplots(ncols=3, figsize=(2 * plt.figaspect(1 / 3)))
        # imshow_args = {'vmin': np.amin((lensing_signal, noise, total_resid)),
        #                'vmax': np.amax((lensing_signal, noise, total_resid))}
        # im = ax[0].imshow(lensing_signal, **imshow_args)
        # ax[1].imshow(noise, **imshow_args)
        # ax[2].imshow(total_resid, **imshow_args)
        # titles = ['lensed_nonoise - unlensed_nonoise', 'lensed_withnoise - lensed_nonoise',
        #           'lensed_withnoise - unlensed_nonoise']
        # _ = [a.set_title(title) for a, title in zip(ax, titles)]
        # cb = fig.colorbar(im, ax=ax)
        # cb.set_label(r'$\mu K$')
        # plt.show()
        # exit()

        # Store
        lensed_t_maps[real_idx] = lensed_t_data
        k_maps[real_idx] = k_data

    stop_time = time.time()
    print()
    print(f'Time taken for {n_real_tot} realisations: {(stop_time - start_time):.1f} s')
    assert np.all(np.isfinite(lensed_t_maps))
    assert np.all(np.isfinite(k_maps))

    # If scaling isn't supplied, determine automatically
    # The idea is that all data will end up between 0 and 1
    if AUTO_SCALE:
        t_scaling = 1.01 * np.amax(np.abs(lensed_t_maps))
        k_scaling = 1.01 * np.amax(np.abs(k_maps))
        # Round to 3sf for consistent replication
        t_scaling = float(f'{t_scaling:.2e}')
        k_scaling = float(f'{k_scaling:.2e}')
        print('Auto scaling')
    else:
        assert T_SCALING is not None and K_SCALING is not None
        t_scaling = T_SCALING
        k_scaling = K_SCALING
        print('Using preset scaling')
    print('t_scaling:', t_scaling)
    print('k_scaling:', k_scaling)

    # Apply scaling
    lensed_t_maps = lensed_t_maps / (2 * t_scaling) + 0.5
    k_maps = k_maps / (2 * k_scaling) + 0.5
    assert 0 < np.amin(lensed_t_maps) < np.amax(lensed_t_maps) < 1
    assert 0 < np.amin(k_maps) < np.amax(k_maps) < 1

    # Split into train, val and test
    lensed_t_maps_train = lensed_t_maps[:N_REAL_TRAIN]
    k_maps_train = k_maps[:N_REAL_TRAIN]
    lensed_t_maps_val = lensed_t_maps[N_REAL_TRAIN:(N_REAL_TRAIN + N_REAL_VAL)]
    k_maps_val = k_maps[N_REAL_TRAIN:(N_REAL_TRAIN + N_REAL_VAL)]
    lensed_t_maps_test = lensed_t_maps[(N_REAL_TRAIN + N_REAL_VAL):]
    k_maps_test = k_maps[(N_REAL_TRAIN + N_REAL_VAL):]

    # Save to file
    train_save_path = SAVE_PATH.format(set_name='train', n_real=N_REAL_TRAIN)
    header = (f'Training set output from {__file__} function create_train_val_test_sets with NPIX = {NPIX}, '
              f'LX_DEG = {LX_DEG}, N_REAL_TRAIN = {N_REAL_TRAIN}, t_scaling {t_scaling}, k_scaling {k_scaling} '
              f'at {time.strftime("%c")}')
    np.savez_compressed(train_save_path, lensed_t_maps=lensed_t_maps_train, k_maps=k_maps_train, t_scaling=t_scaling,
                        k_scaling=k_scaling, header=header)
    print('Saved ' + train_save_path)
    val_save_path = SAVE_PATH.format(set_name='val', n_real=N_REAL_VAL)
    header = (f'Validation set output from {__file__} function create_train_val_test_sets with NPIX = {NPIX}, '
              f'LX_DEG = {LX_DEG}, N_REAL_VAL = {N_REAL_VAL}, t_scaling {t_scaling}, k_scaling {k_scaling} '
              f'at {time.strftime("%c")}')
    np.savez_compressed(val_save_path, lensed_t_maps=lensed_t_maps_val, k_maps=k_maps_val, t_scaling=t_scaling,
                        k_scaling=k_scaling, header=header)
    print('Saved ' + val_save_path)
    test_save_path = SAVE_PATH.format(set_name='test', n_real=N_REAL_TEST)
    header = (f'Test set output from {__file__} function create_train_val_test_sets with NPIX = {NPIX}, '
              f'LX_DEG = {LX_DEG}, N_REAL_TEST = {N_REAL_TEST}, t_scaling {t_scaling}, k_scaling {k_scaling} '
              f'at {time.strftime("%c")}')
    np.savez_compressed(test_save_path, lensed_t_maps=lensed_t_maps_test, k_maps=k_maps_test, t_scaling=t_scaling,
                        k_scaling=k_scaling, header=header)
    print('Saved ' + test_save_path)
