"""
Testing lenstools_light against the real lenstools.
"""

import astropy.units as units
import camb
import lenstools.image.convergence as lt
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
import pymaster as nmt

import lenstools_light


def main():
    """
    Test lenstools_light against the real lenstools.
    """

    npix = 50
    lx_deg = 22.9

    # Calculate CMB temperature power spectrum
    t_lmax = 7000
    camb_params = camb.model.CAMBparams(max_l=2*t_lmax)
    camb_params.set_cosmology(H0=67)
    cmb_cls = camb.results.CAMBdata().get_cmb_power_spectra(params=camb_params, lmax=t_lmax,
                                                            spectra=['unlensed_scalar'], raw_cl=True)
    t_cl_in = cmb_cls['unlensed_scalar'][:, 0]

    # Calculate kappa Cls using CCL, up to an lmax chosen to more-than max out the available resolution
    # pixel_size_deg = lx_deg / npix
    k_lmax_in = 50 # 2 * 180 / pixel_size_deg # NOTE using kappa lmax 50
    ccl_cosmo = ccl.core.CosmologyVanillaLCDM()
    z_source = 1100
    tracer = ccl.tracers.CMBLensingTracer(ccl_cosmo, z_source)
    ell_in = np.arange(k_lmax_in + 1)
    k_cl_in = ccl.angular_cl(ccl_cosmo, tracer, tracer, ell_in)

    # Generate CMB and kappa realisations with NaMaster
    lx_rad = np.radians(lx_deg)
    unlensed_t_data = np.squeeze(nmt.utils.synfast_flat(npix, npix, lx_rad, lx_rad, [t_cl_in], [0]))
    k_data = np.squeeze(nmt.utils.synfast_flat(npix, npix, lx_rad, lx_rad, [k_cl_in], [0]))

    # Lens with lenstools_light first
    result_light = lenstools_light.lens_t_map(unlensed_t_data, k_data, lx_deg)

    # Initialise LensTools maps
    lx = lx_deg * units.degree # pylint: disable=no-member
    unlensed_t_map = lt.CMBTemperatureMap(unlensed_t_data, lx, space='real', unit=units.dimensionless_unscaled)
    k_map = lt.ConvergenceMap(k_data, lx)

    # Lens with LensTools
    lensed_t_map = unlensed_t_map.lens(k_map)
    result_lenstools = lensed_t_map.data

    # Compare
    print('Match:', np.allclose(result_lenstools, result_light))

    # Plot lensed vs unlensed
    _, ax = plt.subplots(ncols=4, figsize=plt.figaspect(1 / 4))
    imshow_args = {'extent': [0, lx_deg, 0, lx_deg], 'interpolation': 'none'}
    ax[0].imshow(unlensed_t_data, **imshow_args)
    ax[1].imshow(result_light, **imshow_args)
    ax[2].imshow(result_light - unlensed_t_data, **imshow_args)
    ax[3].imshow(k_data, **imshow_args)
    _ = [ai.set_major_formatter("${x:.0f}^\\circ$") for a in ax for ai in (a.xaxis, a.yaxis)]
    _ = [col.set_title(title) for col, title in zip(ax, ['Unlensed', 'Lensed', r'Unlensed $-$ lensed', 'Convergence'])]
    plt.show()
