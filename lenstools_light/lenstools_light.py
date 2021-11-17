"""
Contains code for CMB lensing extracted from LensTools by Andrea Petri (https://lenstools.readthedocs.io/).
"""

# import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp


LENS_LMAX = 3500 # Hardcoded value in lenstools


def make_lensed_map_flat_sky(unlensed_t_map, phi_fft, npix, pixel_size_rad, psi=0.0):
    """
    perform the remapping operation of lensing in the flat-sky approximation.

    (optional) psi = angle to rotate the deflection field by, in radians
                     (e.g. psi=pi/2 results in phi being treated as a curl potential).
    """

    # Deflection field
    lx, ly = np.meshgrid(np.fft.fftfreq(npix, pixel_size_rad)[0:(npix // 2 + 1)] * 2. * np.pi,
                         np.fft.fftfreq(npix, pixel_size_rad) * 2. * np.pi)
    x, y = np.meshgrid(np.arange(0, npix) * pixel_size_rad, np.arange(0, npix) * pixel_size_rad)
    gpx = np.fft.irfft2(phi_fft * lx * -1.j * np.sqrt((npix * npix) / (pixel_size_rad * pixel_size_rad)))
    gpy = np.fft.irfft2(phi_fft * ly * -1.j * np.sqrt((npix * npix) / (pixel_size_rad * pixel_size_rad)))

    # Apply rotation
    if psi != 0.0:
        gp = (gpx + 1.j * gpy) * np.exp(1.j * psi)
        gpx = gp.real
        gpy = gp.imag

    # # Study size of deflections
    # gpx_arcmin = 60 * np.degrees(gpx)
    # gpy_arcmin = 60 * np.degrees(gpy)
    # gp_arcmin = np.concatenate((np.ravel(gpx_arcmin), np.ravel(gpy_arcmin)))
    # rms_deflection = np.std(gp_arcmin)
    # plt.hist(gp_arcmin, bins=50)
    # plt.xlabel('Deflection angle (arcmin)')
    # plt.ylabel('Count')
    # plt.annotate(f'rms = {rms_deflection:.2f} arcmin', (0.9, 0.9), xycoords='axes fraction', ha='right')
    # plt.show()

    # Interpolate
    lxs = (x + gpx).flatten()
    del x, gpx
    lys = (y + gpy).flatten()
    del y, gpy
    interp_x = np.arange(0, npix) * pixel_size_rad
    lensed_t_map = interp.RectBivariateSpline(interp_x, interp_x, unlensed_t_map).ev(lys, lxs).reshape([npix, npix])

    return lensed_t_map


def lens_t_map(unlensed_t_map, kappa_map, lx_deg):
    """
    Lens CMB temperature map unlensed_t_map with kappa_map and return the lensed T map.

    Only works for square maps.
    """

    # Determine number of pixels and pixel size, and do some consistency checks
    npix = unlensed_t_map.shape[0]
    assert unlensed_t_map.shape == (npix, npix)
    assert unlensed_t_map.shape == kappa_map.shape
    lx_rad = lx_deg * np.pi / 180
    pixel_size_rad = lx_rad / npix

    # Form ell and ell^2 grids
    fft_freq = np.fft.fftfreq(npix)
    ell_x, ell_y = np.meshgrid(fft_freq, fft_freq, indexing='ij')
    ell2 = (ell_x ** 2 + ell_y ** 2) * ((2.0 * np.pi * npix / lx_rad) ** 2)
    ell2[0, 0] = 1.0
    ell = np.sqrt(ell2)

    # Compute lensing potential
    tfac = lx_rad / (npix ** 2)
    kappa_fft = np.fft.rfft2(kappa_map) * tfac
    phi_fft = kappa_fft * 2.0 / ell2[:, :(npix // 2 + 1)]

    # Zero out high multipoles
    phi_fft[ell[:, :(npix // 2 + 1)] > LENS_LMAX] = 0.

    # Lens
    lensed_t_map = make_lensed_map_flat_sky(unlensed_t_map, phi_fft, npix, pixel_size_rad)

    return lensed_t_map


def lens_t_with_phi(unlensed_t_map, phi_map, lx_deg):
    """
    Lens CMB temperature map unlensed_t_map with phi_map and return the lensed T map.

    Only works for square maps.
    """

    # Determine number of pixels and pixel size, and do some consistency checks
    npix = unlensed_t_map.shape[0]
    assert unlensed_t_map.shape == (npix, npix)
    assert unlensed_t_map.shape == phi_map.shape
    lx_rad = lx_deg * np.pi / 180
    pixel_size_rad = lx_rad / npix

    # Form ell and ell^2 grids
    fft_freq = np.fft.fftfreq(npix)
    ell_x, ell_y = np.meshgrid(fft_freq, fft_freq, indexing='ij')
    ell2 = (ell_x ** 2 + ell_y ** 2) * ((2.0 * np.pi * npix / lx_rad) ** 2)
    ell2[0, 0] = 1.0
    ell = np.sqrt(ell2)

    # Compute lensing potential in Fourier space
    tfac = lx_rad / (npix ** 2)
    phi_fft = np.fft.rfft2(phi_map) * tfac

    # Zero out high multipoles
    phi_fft[ell[:, :(npix // 2 + 1)] > LENS_LMAX] = 0.

    # Lens
    lensed_t_map = make_lensed_map_flat_sky(unlensed_t_map, phi_fft, npix, pixel_size_rad)

    return lensed_t_map
