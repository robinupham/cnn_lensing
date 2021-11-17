"""
Investigating the effect of resolution on CMB lensing simulations.
"""

import camb
import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt

import lenstools_light


def downgrade_map(input_map, downgrade_factor):
    """
    Downgrade a map by the given factor, by averaging.
    """

    # Do checks and calculate output size
    assert input_map.ndim == 2
    assert int(downgrade_factor) == downgrade_factor
    nx_in, ny_in = input_map.shape
    assert nx_in % downgrade_factor == 0
    assert ny_in % downgrade_factor == 0
    nx_out = nx_in // downgrade_factor
    ny_out = ny_in // downgrade_factor

    # Reshape the input and then average over the binned axes
    input_map_reshaped = input_map.reshape((nx_out, downgrade_factor, ny_out, downgrade_factor))
    output_map = np.mean(input_map_reshaped, axis=(1, 3))
    assert output_map.shape == (nx_out, ny_out)

    return output_map


def main():
    """Main function"""

    # Parameters
    lmax = 7000
    pixel_size_arcmin = 0.25
    lores_pixel_size_arcmin = 1
    lx_deg = 20

    # Check and adjust lx if non-integer number of pixels
    downgrade_factor = lores_pixel_size_arcmin / pixel_size_arcmin
    assert int(downgrade_factor) == downgrade_factor, 'lores pixel size must be divisible by hi-res pixel size'
    downgrade_factor = int(downgrade_factor)
    npix_hires = int(60 * lx_deg / pixel_size_arcmin)
    lx_deg = pixel_size_arcmin * npix_hires / 60

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

    # Create T and kappa realisations with NaMaster
    print('Generating hi-res maps')
    lx_rad = np.radians(lx_deg)
    unlensed_t_map = np.squeeze(nmt.utils.synfast_flat(npix_hires, npix_hires, lx_rad, lx_rad, [cl_tt], [0]))
    kappa_map = np.squeeze(nmt.utils.synfast_flat(npix_hires, npix_hires, lx_rad, lx_rad, [cl_kk], [0]))

    # Lens
    print('Lensing')
    lensed_t_map = lenstools_light.lens_t_map(unlensed_t_map, kappa_map, lx_deg)

    # # Plot
    # _, ax = plt.subplots(ncols=4, figsize=plt.figaspect(1 / 4))
    # imshow_args = {'extent': [0, lx_deg, 0, lx_deg], 'interpolation': 'none'}
    # ax[0].imshow(unlensed_t_map, **imshow_args)
    # ax[1].imshow(lensed_t_map, **imshow_args)
    # ax[2].imshow(lensed_t_map - unlensed_t_map, **imshow_args)
    # ax[3].imshow(kappa_map, **imshow_args)
    # _ = [col.set_title(title) for col, title in zip(ax, ['Unlensed', 'Lensed', r'Unlensed $-$ lensed', 'Deflection'])]
    # plt.show()
    # exit()

    # Downgrade the maps to low res
    print('Downgrading')
    unlensed_t_lores = downgrade_map(unlensed_t_map, downgrade_factor)
    kappa_lores = downgrade_map(kappa_map, downgrade_factor)
    lensed_t_downgraded = downgrade_map(lensed_t_map, downgrade_factor)

    # Lens again
    print('Lensing again')
    lensed_t_lores = lenstools_light.lens_t_map(unlensed_t_lores, kappa_lores, lx_deg)

    # Compare
    _, ax = plt.subplots(ncols=4, nrows=2, figsize=1.4*plt.figaspect(2 / 4), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    imshow_args = {'extent': [0, lx_deg, 0, lx_deg], 'interpolation': 'none'}
    top_vmin = np.amin([np.amin(map_) for map_ in (lensed_t_map, lensed_t_downgraded, lensed_t_lores)])
    top_vmax = np.amax([np.amax(map_) for map_ in (lensed_t_map, lensed_t_downgraded, lensed_t_lores)])
    ax[0, 0].imshow(lensed_t_map, vmin=top_vmin, vmax=top_vmax, **imshow_args)
    ax[0, 1].imshow(lensed_t_downgraded, vmin=top_vmin, vmax=top_vmax, **imshow_args)
    ax[0, 2].imshow(lensed_t_lores, vmin=top_vmin, vmax=top_vmax, **imshow_args)
    ax[0, 3].imshow(lensed_t_lores - lensed_t_downgraded, **imshow_args)
    resid_hires = lensed_t_map - unlensed_t_map
    resid_downgraded = lensed_t_downgraded - unlensed_t_lores
    resid_lores = lensed_t_lores - unlensed_t_lores
    bottom_vmin = np.amin([np.amin(map_) for map_ in (resid_hires, resid_downgraded, resid_lores)])
    bottom_vmax = np.amax([np.amax(map_) for map_ in (resid_hires, resid_downgraded, resid_lores)])
    ax[1, 0].imshow(resid_hires, vmin=bottom_vmin, vmax=bottom_vmax, **imshow_args)
    ax[1, 1].imshow(resid_downgraded, vmin=bottom_vmin, vmax=bottom_vmax, **imshow_args)
    ax[1, 2].imshow(resid_lores, vmin=bottom_vmin, vmax=bottom_vmax, **imshow_args)
    ax[1, 3].imshow(resid_lores - resid_downgraded, **imshow_args)
    col_titles = [f'Hi res (pixel = {pixel_size_arcmin} arcmin)', 'Hi res downgraded',
                  f'Low res (pixel = {lores_pixel_size_arcmin} arcmin)', r'Low res $-$ hi res downgraded']
    row_titles = ['Lensed', r'Lensed $-$ unlensed']
    _ = [a.set_major_formatter("${x:.0f}^\\circ$") for a in (ax[0, 0].xaxis, ax[0, 0].yaxis)]
    _ = [col.set_title(title) for col, title in zip(ax[0, :], col_titles)]
    _ = [row[0].annotate(title, xy=(-0.2, 0.5), xycoords='axes fraction', ha='right', va='center',
                         size=plt.rcParams['axes.titlesize']) for row, title in zip(ax, row_titles)]
    plt.show()
