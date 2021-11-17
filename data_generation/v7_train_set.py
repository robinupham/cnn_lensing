"""
Generate training, validation and test sets for v7 of the CNN lensing estimation project.
"""

import time

import astropy.units as units
import camb
import lenstools.image.convergence as lt
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
import pymaster as nmt


NPIX = 50
LX_DEG = 22.9

N_REAL_TRAIN = 20000
N_REAL_VAL = 200
N_REAL_TEST = 3
AUTO_SCALE = True
T_SCALING = None # max abs value for normalisation, which only needs supplying if AUTO_SCALE = False
K_SCALING = None
SAVE_PATH = 'path_to_save/{set_name}_{n_real}.npz'


def create_train_val_test_sets():
    """
    Create training, validation and test sets.
    """

    # Calculate CMB temperature power spectrum up to the max that CAMB can handle (l ~ 7000)
    t_lmax = 7000
    camb_params = camb.model.CAMBparams(max_l=2*t_lmax)
    camb_params.set_cosmology(H0=67)
    cmb_cls = camb.results.CAMBdata().get_cmb_power_spectra(params=camb_params, lmax=t_lmax,
                                                            spectra=['unlensed_scalar'], raw_cl=True)
    t_cl_in = cmb_cls['unlensed_scalar'][:, 0]

    # Calculate kappa Cls using CCL, up to an lmax chosen to more-than max out the available resolution
    pixel_size_deg = LX_DEG / NPIX
    k_lmax_in = 2 * 180 / pixel_size_deg
    ccl_cosmo = ccl.core.CosmologyVanillaLCDM()
    z_source = 1100
    tracer = ccl.tracers.CMBLensingTracer(ccl_cosmo, z_source)
    ell_in = np.arange(k_lmax_in + 1)
    k_cl_in = ccl.angular_cl(ccl_cosmo, tracer, tracer, ell_in)

    # Generate realisations
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
        lx = LX_DEG * units.degree # pylint: disable=no-member
        unlensed_t_map = lt.CMBTemperatureMap(unlensed_t_data, lx, space='real', unit=units.dimensionless_unscaled)
        k_map = lt.ConvergenceMap(k_data, lx)

        # Lens
        lensed_t_map = unlensed_t_map.lens(k_map)

        # # Plot
        # _, ax = plt.subplots(ncols=4, figsize=(2 * plt.figaspect(1 / 4)))
        # ax[0].imshow(unlensed_t_map.data)
        # ax[1].imshow(lensed_t_map.data)
        # ax[2].imshow(lensed_t_map.data - unlensed_t_map.data)
        # ax[3].imshow(k_map.data)
        # plt.show()

        # Store
        lensed_t_maps[real_idx] = lensed_t_map.data
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


def testing():
    """
    Experimenting with resolution.
    """

    # Input kappa lmax to deliberately more-than max out the available resolution
    pixel_size_deg = LX_DEG / NPIX
    k_lmax_in = 2 * 180 / pixel_size_deg

    # Calculate vanilla Lambda-CDM kappa Cls using CCL
    cosmo = ccl.core.CosmologyVanillaLCDM()
    z_source = 1100
    tracer = ccl.tracers.CMBLensingTracer(cosmo, z_source)
    ell_in = np.arange(k_lmax_in + 1)
    k_cl_in = ccl.angular_cl(cosmo, tracer, tracer, ell_in)

    # Generate a realisation
    lx_rad = np.radians(LX_DEG)
    k_map = np.squeeze(nmt.utils.synfast_flat(NPIX, NPIX, lx_rad, lx_rad, [k_cl_in], [0]))

    # Measure the power spectrum
    k_lmin_obs = int(1.5 * 180 / LX_DEG)
    k_lmax_obs = int(2/3 * 180 / pixel_size_deg)
    nbin = 20 # 50
    mask = np.ones((NPIX, NPIX))
    k_field = nmt.field.NmtFieldFlat(lx_rad, lx_rad, mask, [k_map])
    bin_edges = np.linspace(k_lmin_obs, k_lmax_obs, nbin + 1)
    bins = nmt.bins.NmtBinFlat(bin_edges[:-1], bin_edges[1:])
    k_cl_obs = np.squeeze(nmt.workspaces.compute_coupled_cell_flat(k_field, k_field, bins))

    # Plot input and observed power spectra
    ell_obs = bins.get_effective_ells()
    ell_in = ell_in[k_lmin_obs:(k_lmax_obs + 1)]
    k_cl_in = k_cl_in[k_lmin_obs:(k_lmax_obs + 1)]
    cl_fac_in = ell_in * (ell_in + 1) / (2 * np.pi)
    cl_fac_obs = ell_obs * (ell_obs + 1) / (2 * np.pi)
    plt.plot(ell_in, cl_fac_in * k_cl_in, label='Input')
    plt.step(bin_edges, np.pad(cl_fac_obs * k_cl_obs, (0, 1), mode='edge'), where='post', label='Obs')
    plt.legend()
    plt.show()
