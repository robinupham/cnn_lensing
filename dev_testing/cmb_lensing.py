"""
Testing simulating CMB lensing 'manually', i.e. not using LensTools.

Turned out to not work very well.
"""

import warnings

import camb
import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt
import scipy.interpolate as interp


def apply_lens(unlensed_t_map, phi_map):
    """
    Simple CMB lensing. Lens unlensed_t_map with deflection map phi_map.
    """

    warnings.warn('I think this doesn\'t work correctly, don\'t trust the results')

    # Generate coordinate grid
    nx, ny = np.indices(unlensed_t_map.shape)

    # Calculate grad phi
    dphi_dx, dphi_dy = np.gradient(phi_map, edge_order=2)

    # Apply lens equation using interpolation
    grid_x = np.arange(unlensed_t_map.shape[0])
    grid_y = np.arange(unlensed_t_map.shape[1])
    lensed_t_map = interp.RectBivariateSpline(grid_x, grid_y, unlensed_t_map).ev(nx + dphi_dx, ny + dphi_dy)

    return lensed_t_map


def main():
    """Main function"""

    # Parameters
    npix = 50
    lx_deg = 22.9
    phi_lmax = 50

    # Choose T lmax to more than max out the available resolution
    pixel_size_deg = lx_deg / npix
    t_lmax = int(2 * 180 / pixel_size_deg)

    # Get TT and φφ power spectra with CAMB, assuming a vanilla ΛCDM model
    # Raw Cls mean no factors of l attached
    # TT Cls are in microkelvin squared, φφ is dimensionless
    camb_params = camb.CAMBparams()
    camb_params.set_cosmology(H0=67)
    camb_params.set_for_lmax(2 * max(t_lmax, phi_lmax), lens_potential_accuracy=1)
    res = camb.get_results(camb_params)
    cl_tt = res.get_cmb_power_spectra(lmax=t_lmax, spectra=['unlensed_scalar'], CMB_unit='muK',
                                      raw_cl=True)['unlensed_scalar'][:, 0]
    cl_pp = res.get_lens_potential_cls(lmax=phi_lmax, CMB_unit='muK', raw_cl=True)[:, 0]

    # # Plot φφ power spectrum
    # ell = np.arange(phi_lmax + 1)
    # cl_fac = ell * (ell + 1) / (2 * np.pi)
    # plt.plot(ell, cl_fac * cl_pp)
    # plt.show()

    # Create T and φ realisations with NaMaster
    lx_rad = np.radians(lx_deg)
    unlensed_t_map = np.squeeze(nmt.utils.synfast_flat(npix, npix, lx_rad, lx_rad, [cl_tt], [0]))
    phi_map = np.squeeze(nmt.utils.synfast_flat(npix, npix, lx_rad, lx_rad, [cl_pp], [0]))

    # Lens
    lensed_t_map = apply_lens(unlensed_t_map, phi_map)

    # Plot lensed vs unlensed
    _, ax = plt.subplots(ncols=4, figsize=plt.figaspect(1 / 4))
    imshow_args = {'extent': [0, lx_deg, 0, lx_deg], 'interpolation': 'none'}
    ax[0].imshow(unlensed_t_map, **imshow_args)
    ax[1].imshow(lensed_t_map, **imshow_args)
    ax[2].imshow(lensed_t_map - unlensed_t_map, **imshow_args)
    ax[3].imshow(phi_map, **imshow_args)
    _ = [ai.set_major_formatter("${x:.0f}^\\circ$") for a in ax for ai in (a.xaxis, a.yaxis)]
    _ = [col.set_title(title) for col, title in zip(ax, ['Unlensed', 'Lensed', r'Unlensed $-$ lensed', 'Deflection'])]
    plt.show()
