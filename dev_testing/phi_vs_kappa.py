"""
Check that the lenstools_light functions for kappa and phi give the same results.
"""

import camb
import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt

import lenstools_light


def main():
    """
    Check that the lenstools_light functions for kappa and phi give the same results.

    Contains an infinite loop, so must be killed to stop.
    """

    # Parameters
    lmax = 7000
    pixel_size_arcmin = 1
    lx_deg = 20

    # Check and adjust lx if non-integer number of pixels
    npix = int(60 * lx_deg / pixel_size_arcmin)
    lx_deg = pixel_size_arcmin * npix / 60

    # Get TT and φφ power spectra with CAMB, assuming a vanilla ΛCDM model
    # Raw Cls mean no factors of l attached
    # TT Cls are in microkelvin squared, φφ is dimensionless
    print('Generating Cls')
    camb_params = camb.CAMBparams()
    camb_params.set_cosmology(H0=67)
    camb_params.set_for_lmax(2 * lmax, lens_potential_accuracy=1)
    res = camb.get_results(camb_params)
    cl_tt = res.get_cmb_power_spectra(lmax=lmax, spectra=['unlensed_scalar'], CMB_unit='muK',
                                      raw_cl=True)['unlensed_scalar'][:, 0]
    cl_pp = res.get_lens_potential_cls(lmax=lmax, CMB_unit='muK', raw_cl=True)[:, 0]

    # Calculate kappa Cls from phi Cls
    ell = np.arange(lmax + 1)
    kappa_fac = ((ell * (ell + 1)) ** 2) / 4.
    cl_kk = kappa_fac * cl_pp

    while True: # NOTE infinite loop with no break

        # Create T and equivalent phi and kappa realisations with NaMaster
        print('Generating maps')
        lx_rad = np.radians(lx_deg)
        unlensed_t_map = np.squeeze(nmt.utils.synfast_flat(npix, npix, lx_rad, lx_rad, [cl_tt], [0]))
        # Use the same random seed, although this doesn't actually give complete equivalence
        seed = int(np.random.default_rng().integers(1000))
        kappa_map = np.squeeze(nmt.utils.synfast_flat(npix, npix, lx_rad, lx_rad, [cl_kk], [0], seed=seed))
        phi_map = np.squeeze(nmt.utils.synfast_flat(npix, npix, lx_rad, lx_rad, [cl_pp], [0], seed=seed))

        # # Plot
        # _, ax = plt.subplots(ncols=2)
        # ax[0].imshow(kappa_map)
        # ax[1].imshow(phi_map)
        # plt.show()

        # Lens
        print('Lensing')
        lensed_by_kappa = lenstools_light.lens_t_map(unlensed_t_map, kappa_map, lx_deg)
        lensed_by_phi = lenstools_light.lens_t_with_phi(unlensed_t_map, phi_map, lx_deg)

        # Plot residuals
        lensed_by_kappa_resid = lensed_by_kappa - unlensed_t_map
        lensed_by_phi_resid = lensed_by_phi - unlensed_t_map
        _, ax = plt.subplots(ncols=2)
        vmin = np.amin((lensed_by_kappa_resid, lensed_by_phi_resid))
        vmax = np.amax((lensed_by_kappa_resid, lensed_by_phi_resid))
        ax[0].imshow(lensed_by_kappa_resid, vmin=vmin, vmax=vmax)
        ax[1].imshow(lensed_by_phi_resid, vmin=vmin, vmax=vmax)
        plt.show()
