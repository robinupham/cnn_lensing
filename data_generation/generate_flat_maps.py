"""
Functions to generate flat-sky CMB T map or kappa map realisations using NaMaster.

This was used for versions 1-5, after which there are specific per-version data generation scripts.
"""

import os.path
import time

import camb
import healpy as hp
# import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
import pymaster as nmt


NPIX = 50
NSIDE_EQUIV = 8192 # resolution specified as 'nside equivalent', which is converted to NaMaster lx parameter

T_LMAX = 7000
K_LMAX = 3 * NSIDE_EQUIV - 1

N_IMG = 10
SAVE_DIR = 'directory to save output'


def generate_unlensed_tmaps():
    """
    Generate unlensed CMB temperature maps and save to disk.
    """

    # Generate Cls using CAMB
    params = camb.model.CAMBparams(max_l=2*T_LMAX)
    params.set_cosmology(H0=67)
    cmb_cls = camb.results.CAMBdata().get_cmb_power_spectra(params=params, lmax=T_LMAX, spectra=['unlensed_scalar'],
                                                            raw_cl=True)
    cl = cmb_cls['unlensed_scalar'][:, 0]

    # Calculate NaMaster lx (=ly) parameter based on the nside equivalent
    lx = NPIX * np.sqrt(hp.nside2pixarea(NSIDE_EQUIV))
    print(f'lx = {np.degrees(lx)} deg')

    # Generate maps
    tmaps = np.full((N_IMG, NPIX, NPIX), np.nan)
    for i in range(N_IMG):
        print(f'Generating realisation {i + 1} / {N_IMG}', end='\r')
        tmaps[i] = nmt.utils.synfast_flat(NPIX, NPIX, lx, lx, [cl], [0])
    assert np.all(np.isfinite(tmaps))
    print()

    # # Plot
    # for tmap in tmaps:
    #     plt.imshow(np.squeeze(tmap))
    #     plt.show()

    # Save to disk
    save_path = os.path.join(SAVE_DIR + f'tmaps_{N_IMG}.npz')
    header = (f'Output from {__file__} function generate_unlensed_tmaps for input NPIX = {NPIX}, '
              f'NSIDE_EQUIV = {NSIDE_EQUIV}, T_LMAX = {T_LMAX}, N_IMG = {N_IMG} at {time.strftime("%c")}')
    np.savez_compressed(save_path, tmaps=tmaps, header=header)
    print('Saved ' + save_path)


def generate_kappa_maps():
    """
    Generate CMB lensing convergence maps and save to disk.
    """

    # Calculate vanilla Lambda-CDM Cls using CCL
    cosmo = ccl.core.CosmologyVanillaLCDM()
    z_source = 1100
    tracer = ccl.tracers.CMBLensingTracer(cosmo, z_source)
    ell = np.arange(K_LMAX + 1)
    cl = ccl.angular_cl(cosmo, tracer, tracer, ell)

    # Calculate NaMaster lx (=ly) parameter based on the nside equivalent
    lx = NPIX * np.sqrt(hp.nside2pixarea(NSIDE_EQUIV))
    print(f'lx = {np.degrees(lx)} deg')
    print(f'pixel size = {np.degrees(lx / NPIX) * 60} arcmin')

    # Generate maps
    kapmaps = np.full((N_IMG, NPIX, NPIX), np.nan)
    for i in range(N_IMG):
        print(f'Generating realisation {i + 1} / {N_IMG}', end='\r')
        kapmaps[i] = nmt.utils.synfast_flat(NPIX, NPIX, lx, lx, [cl], [0])
    assert np.all(np.isfinite(kapmaps))

    # Save to disk
    save_path = os.path.join(SAVE_DIR, f'kappa_maps_{N_IMG}.npz')
    header = (f'Output from {__file__} function generate_kappa_maps for input NPIX = {NPIX}, '
              f'NSIDE_EQUIV = {NSIDE_EQUIV}, K_LMAX = {K_LMAX}, cosmology = CCL Vanilla LCDM, z_source = {z_source}, '
              f'N_IMG = {N_IMG} at {time.strftime("%c")}')
    np.savez_compressed(save_path, kapmaps=kapmaps, header=header)
    print('Saved ' + save_path)
