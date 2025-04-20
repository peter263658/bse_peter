#!/usr/bin/env python3
# coding: utf-8

"""
Modified Binaural Short-Time Objective Intelligibility (MBSTOI) implementation.

Adapted from the original implementation described in:
A. H. Andersen, J. M. de Haan, Z.-H. Tan, and J. Jensen, "Refinement
and validation of the binaural short time objective intelligibility
measure for spatially diverse conditions," Speech Communication,
vol. 102, pp. 1-13, Sep. 2018.

This implementation removes dependencies on external modules like clarity_core.
"""

import numpy as np
import math
import logging
from scipy.signal import resample

# Import MBSTOI components or define them here if not available
import sys
import os

# Create a config object to replace the clarity_core.CONFIG dependency
class Config:
    def __init__(self):
        self.fs = 16000  # Default sampling rate

CONFIG = Config()

def thirdoct(fs, nfft, num_bands, min_freq):
    """Returns the 1/3 octave band matrix and its center frequencies
    """
    f = np.linspace(0, fs, nfft + 1)
    f = f[: int(nfft / 2) + 1]
    k = np.array(range(num_bands)).astype(float)
    cf = np.power(2.0 ** (1.0 / 3), k) * min_freq
    freq_low = min_freq * np.power(2.0, (2 * k - 1) / 6)
    freq_high = min_freq * np.power(2.0, (2 * k + 1) / 6)
    obm = np.zeros((num_bands, len(f)))  # a verifier
    fids = np.zeros((num_bands, 2))

    for i in range(len(cf)):
        # Match 1/3 oct band freq with fft frequency bin
        f_bin = np.argmin(np.square(f - freq_low[i]))
        freq_low[i] = f[f_bin]
        fl_ii = f_bin
        f_bin = np.argmin(np.square(f - freq_high[i]))
        freq_high[i] = f[f_bin]
        fh_ii = f_bin
        # Assign to the octave band matrix
        obm[i, fl_ii:fh_ii] = 1
        fids[i, :] = [fl_ii + 1, fh_ii]

    cf = cf[np.newaxis, :]

    return obm, cf, fids, freq_low, freq_high

def stft(x, win_size, fft_size):
    """Short-time Fourier transform"""
    hop = int(win_size / 2)
    frames = list(range(0, len(x) - win_size, hop))
    stft_out = np.zeros((len(frames), fft_size), dtype=np.complex128)

    w = np.hanning(win_size + 2)[1:-1]
    x = x.flatten()

    for i in range(len(frames)):
        ii = list(range(frames[i], (frames[i] + win_size), 1))
        stft_out[i, :] = np.fft.fft(x[ii] * w, n=fft_size, axis=0)

    return stft_out

def remove_silent_frames(xl, xr, yl, yr, dyn_range, framelen, hop):
    """Remove silent frames of x and y based on x"""
    EPS = np.finfo("float").eps
    dyn_range = int(dyn_range)
    hop = int(hop)

    # Compute Mask
    w = np.hanning(framelen + 2)[1:-1]

    xl_frames = np.array(
        [w * xl[i : i + framelen] for i in range(0, len(xl) - framelen, hop)]
    )
    xr_frames = np.array(
        [w * xr[i : i + framelen] for i in range(0, len(xr) - framelen, hop)]
    )
    yl_frames = np.array(
        [w * yl[i : i + framelen] for i in range(0, len(yl) - framelen, hop)]
    )
    yr_frames = np.array(
        [w * yr[i : i + framelen] for i in range(0, len(yr) - framelen, hop)]
    )

    # Compute energies in dB
    xl_energies = 20 * np.log10(np.linalg.norm(xl_frames, axis=1) + EPS)
    xr_energies = 20 * np.log10(np.linalg.norm(xr_frames, axis=1) + EPS)

    # Find boolean mask of energies lower than dynamic_range dB
    # with respect to maximum clean speech energy frame
    maskxl = (np.max(xl_energies) - dyn_range - xl_energies) < 0
    maskxr = (np.max(xr_energies) - dyn_range - xr_energies) < 0

    mask = (maskxl == True) | (maskxr == True)

    # Remove silent frames by masking
    xl_frames = xl_frames[mask]
    xr_frames = xr_frames[mask]
    yl_frames = yl_frames[mask]
    yr_frames = yr_frames[mask]

    # init zero arrays to hold x, y with silent frames removed
    n_sil = (len(xl_frames) - 1) * hop + framelen
    xl_sil = np.zeros(n_sil)
    xr_sil = np.zeros(n_sil)
    yl_sil = np.zeros(n_sil)
    yr_sil = np.zeros(n_sil)

    for i in range(xl_frames.shape[0]):
        xl_sil[range(i * hop, i * hop + framelen)] += xl_frames[i, :]
        xr_sil[range(i * hop, i * hop + framelen)] += xr_frames[i, :]
        yl_sil[range(i * hop, i * hop + framelen)] += yl_frames[i, :]
        yr_sil[range(i * hop, i * hop + framelen)] += yr_frames[i, :]

    return xl_sil, xr_sil, yl_sil, yr_sil

def firstpartfunc(L1, L2, R1, R2, ntaus, gammas, epsexp):
    result = (
        np.ones((ntaus, 1))
        * (
            (
                10 ** (2 * gammas) * np.sum(L1 * L2)
                + 10 ** (-2 * gammas) * np.sum(R1 * R2)
            )
            * epsexp
        )
        + np.sum(L1 * R2)
        + np.sum(R1 * L2)
    )
    return result

def secondpartfunc(L1, L2, rho1, rho2, tauexp, epsdelexp, gammas):
    result = (
        2
        * (
            np.transpose(
                np.dot(L1, np.real(np.transpose(rho1) * tauexp))
                + np.dot(L2, np.real(np.transpose(rho2) * tauexp))
            )
            * 10 ** gammas
        )
        * epsdelexp
    )
    return result

def thirdpartfunc(R1, R2, rho1, rho2, tauexp, epsdelexp, gammas):
    result = (
        2
        * np.transpose(
            np.dot(
                R1,
                np.real(np.dot(np.transpose(rho1), tauexp)),
            )
            + np.dot(R2, np.real(np.transpose(rho2) * tauexp))
        )
        * 10 ** -gammas
        * epsdelexp
    )
    return result

def fourthpartfunc(rho1, rho2, tauexp2, ngammas, deltexp):
    result = (
        2
        * np.transpose(
            np.real(np.dot(rho1, np.conj(np.transpose(rho2))))
            + deltexp * np.real(np.dot(rho1, np.transpose(rho2) * tauexp2))
        )
        * np.ones((1, ngammas))
    )
    return result

def ec(
    xl_hat,
    xr_hat,
    yl_hat,
    yr_hat,
    J,
    N,
    fids,
    cf,
    taus,
    ntaus,
    gammas,
    ngammas,
    d,
    p_ec_max,
    sigma_epsilon,
    sigma_delta,
):
    """Run the equalisation-cancellation (EC) stage of the MBSTOI metric."""
    taus = np.expand_dims(taus, axis=0)
    sigma_delta = np.expand_dims(sigma_delta, axis=0)
    sigma_epsilon = np.expand_dims(sigma_epsilon, axis=0)
    gammas = np.expand_dims(gammas, axis=0)
    epsexp = np.exp(2 * np.log(10) ** 2 * sigma_epsilon ** 2)

    for i in range(J):  # per frequency band
        tauexp = np.exp(-1j * cf[i] * taus)
        tauexp2 = np.exp(-1j * 2 * cf[i] * taus)
        deltexp = np.exp(-2 * cf[i] ** 2 * sigma_delta ** 2)
        epsdelexp = np.exp(
            0.5
            * (
                np.ones((ntaus, 1))
                * (
                    np.log(10) ** 2 * sigma_epsilon ** 2
                    - cf[i] ** 2 * np.transpose(sigma_delta) ** 2
                )
                * np.ones((1, ngammas))
            )
        )

        for jj in range(np.shape(d)[1]):  # per frame
            seg_xl = xl_hat[int(fids[i, 0] - 1) : int(fids[i, 1]), jj : (jj + N)]
            seg_xr = xr_hat[int(fids[i, 0] - 1) : int(fids[i, 1]), jj : (jj + N)]
            seg_yl = yl_hat[int(fids[i, 0] - 1) : int(fids[i, 1]), jj : (jj + N)]
            seg_yr = yr_hat[int(fids[i, 0] - 1) : int(fids[i, 1]), jj : (jj + N)]

            # All normalised by subtracting mean
            Lx = np.sum(np.conj(seg_xl) * seg_xl, axis=0)
            Lx = np.expand_dims(Lx, axis=0)
            Lx = Lx - np.mean(Lx)
            Rx = np.sum(np.conj(seg_xr) * seg_xr, axis=0)
            Rx = np.expand_dims(Rx, axis=0)
            Rx = Rx - np.mean(Rx)
            rhox = np.sum(np.conj(seg_xr) * seg_xl, axis=0)
            rhox = np.expand_dims(rhox, axis=0)
            rhox = rhox - np.mean(rhox)
            Ly = np.sum(np.conj(seg_yl) * seg_yl, axis=0)
            Ly = np.expand_dims(Ly, axis=0)
            Ly = Ly - np.mean(Ly)
            Ry = np.sum(np.conj(seg_yr) * seg_yr, axis=0)
            Ry = np.expand_dims(Ry, axis=0)
            Ry = Ry - np.mean(Ry)
            rhoy = np.sum(np.conj(seg_yr) * seg_yl, axis=0)
            rhoy = np.expand_dims(rhoy, axis=0)
            rhoy = rhoy - np.mean(rhoy)

            # Evaluate parts of intermediate correlation - EC stage exhaustive search over ITD/ILD values
            # These correspond to equations 7 and 8 in Andersen et al. 2018
            # Calculate Exy
            firstpart = firstpartfunc(Lx, Ly, Rx, Ry, ntaus, gammas, epsexp)
            secondpart = secondpartfunc(Lx, Ly, rhoy, rhox, tauexp, epsdelexp, gammas)
            thirdpart = thirdpartfunc(Rx, Ry, rhoy, rhox, tauexp, epsdelexp, gammas)
            fourthpart = fourthpartfunc(rhox, rhoy, tauexp2, ngammas, deltexp)
            exy = np.real(firstpart - secondpart - thirdpart + fourthpart)

            # Calculate Exx
            firstpart = firstpartfunc(Lx, Lx, Rx, Rx, ntaus, gammas, epsexp)
            secondpart = secondpartfunc(Lx, Lx, rhox, rhox, tauexp, epsdelexp, gammas)
            thirdpart = thirdpartfunc(Rx, Rx, rhox, rhox, tauexp, epsdelexp, gammas)
            fourthpart = fourthpartfunc(rhox, rhox, tauexp2, ngammas, deltexp)
            exx = np.real(firstpart - secondpart - thirdpart + fourthpart)

            # Calculate Eyy
            firstpart = firstpartfunc(Ly, Ly, Ry, Ry, ntaus, gammas, epsexp)
            secondpart = secondpartfunc(Ly, Ly, rhoy, rhoy, tauexp, epsdelexp, gammas)
            thirdpart = thirdpartfunc(Ry, Ry, rhoy, rhoy, tauexp, epsdelexp, gammas)
            fourthpart = fourthpartfunc(rhoy, rhoy, tauexp2, ngammas, deltexp)
            eyy = np.real(firstpart - secondpart - thirdpart + fourthpart)

            # Ensure that intermediate correlation will be sensible and compute it
            # If all minimum values are less than 1e-40, set d[i,jj] to -1
            if np.min(abs(exx * eyy), axis=0).all() < 1e-40:
                d[i, jj] = -1
                continue
            else:
                p = np.divide(exx, eyy)
                tmp = p.max(axis=0)
                idx1 = p.argmax(axis=0)

                # Return overall maximum and index
                p_ec_max[i, jj] = tmp.max()
                idx2 = tmp.argmax()
                d[i, jj] = np.divide(
                    exy[idx1[idx2], idx2],
                    np.sqrt(exx[idx1[idx2], idx2] * eyy[idx1[idx2], idx2]),
                )

    return d, p_ec_max

def mbstoi(xl, xr, yl, yr, fs_signal=16000, gridcoarseness=1):
    """A Python implementation of the Modified Binaural Short-Time
    Objective Intelligibility (MBSTOI) measure as described in:
    A. H. Andersen, J. M. de Haan, Z.-H. Tan, and J. Jensen, "Refinement
    and validation of the binaural short time objective intelligibility
    measure for spatially diverse conditions," Speech Communication,
    vol. 102, pp. 1-13, Sep. 2018. A. H. Andersen, 10/12-2018

    Args:
        xl (ndarray): clean speech signal from left ear
        xr (ndarray): clean speech signal from right ear.
        yl (ndarray): noisy/processed speech signal from left ear.
        yr (ndarray): noisy/processed speech signal from right ear.
        fs_signal (int): sampling rate of the input signals
        gridcoarseness (integer): grid coarseness as denominator of ntaus and ngammas (default: 1)

    Returns
        float: MBSTOI index d
    """
    # Basic STOI parameters
    fs = 10000  # Sample rate of proposed intelligibility measure in Hz
    N_frame = 256  # Window support in samples
    K = 512  # FFT size in samples
    J = 15  # Number of one-third octave bands
    mn = 150  # Centre frequency of first 1/3 octave band in Hz
    N = 30  # Number of frames for intermediate intelligibility measure (length analysis window)
    dyn_range = 40  # Speech dynamic range in dB

    # Values to define EC grid
    tau_min = -0.001  # Minimum interaural delay compensation in seconds. B: -0.01.
    tau_max = 0.001  # Maximum interaural delay compensation in seconds. B: 0.01.
    ntaus = math.ceil(100 / gridcoarseness)  # Number of tau values to try out
    gamma_min = -20  # Minimum interaural level compensation in dB
    gamma_max = 20  # Maximum interaural level compensation in dB
    ngammas = math.ceil(40 / gridcoarseness)  # Number of gamma values to try out

    # Constants for jitter
    # ITD compensation standard deviation in seconds. Equation 6 Andersen et al. 2018 Refinement
    sigma_delta_0 = 65e-6
    # ILD compensation standard deviation.  Equation 5 Andersen et al. 2018
    sigma_epsilon_0 = 1.5
    # Constant for level shift deviation in dB. Equation 5 Andersen et al. 2018
    alpha_0_db = 13
    # Constant for time shift deviation in seconds. Equation 6 Andersen et al. 2018
    tau_0 = 1.6e-3
    # Constant for level shift deviation. Power for calculation of sigma delta gamma in equation 5 Andersen et al. 2018.
    p = 1.6

    # Prepare signals, ensuring that inputs are column vectors
    xl = np.asarray(xl).flatten()
    xr = np.asarray(xr).flatten()
    yl = np.asarray(yl).flatten()
    yr = np.asarray(yr).flatten()

    # Resample signals to 10 kHz if needed
    if fs_signal != fs:
        logging.debug(f"Resampling signals with sr={fs} for MBSTOI calculation.")
        l = len(xl)
        xl = resample(xl, int(l * (fs / fs_signal) + 1))
        xr = resample(xr, int(l * (fs / fs_signal) + 1))
        yl = resample(yl, int(l * (fs / fs_signal) + 1))
        yr = resample(yr, int(l * (fs / fs_signal) + 1))

    # Remove silent frames
    xl_sil, xr_sil, yl_sil, yr_sil = remove_silent_frames(
        xl, xr, yl, yr, dyn_range, N_frame, N_frame / 2
    )
    
    # Use the resampled signals without silent frames
    xl = xl_sil
    xr = xr_sil
    yl = yl_sil
    yr = yr_sil

    # Handle case when signals are zeros or very different in energy
    if (len(xl) == 0 or len(xr) == 0 or len(yl) == 0 or len(yr) == 0 or
        abs(np.log10(np.linalg.norm(xl) / (np.linalg.norm(yl) + 1e-10))) > 5.0 or
        abs(np.log10(np.linalg.norm(xr) / (np.linalg.norm(yr) + 1e-10))) > 5.0):
        return 0

    # STDFT and filtering
    # Get 1/3 octave band matrix
    H, cf, fids, freq_low, freq_high = thirdoct(fs, K, J, mn)
    cf = 2 * math.pi * cf  # This is now the angular frequency in radians per sec

    # Apply short time DFT to signals and transpose
    xl_hat = stft(xl, N_frame, K).transpose()
    xr_hat = stft(xr, N_frame, K).transpose()
    yl_hat = stft(yl, N_frame, K).transpose()
    yr_hat = stft(yr, N_frame, K).transpose()

    # Take single sided spectrum of signals
    idx = int(K / 2 + 1)
    xl_hat = xl_hat[0:idx, :]
    xr_hat = xr_hat[0:idx, :]
    yl_hat = yl_hat[0:idx, :]
    yr_hat = yr_hat[0:idx, :]

    # Compute intermediate correlation via EC search
    # Here intermediate correlation coefficients are evaluated for a discrete set of
    # gamma and tau values (a "grid") and the highest value is chosen.
    d = np.zeros((J, np.shape(xl_hat)[1] - N + 1))
    p_ec_max = np.zeros((J, np.shape(xl_hat)[1] - N + 1))

    # Interaural compensation time and level values
    taus = np.linspace(tau_min, tau_max, ntaus)
    gammas = np.linspace(gamma_min, gamma_max, ngammas)

    # Jitter incorporated below - Equations 5 and 6 in Andersen et al. 2018
    sigma_epsilon = (
        np.sqrt(2) * sigma_epsilon_0 * (1 + (abs(gammas) / alpha_0_db) ** p) / 20
    )
    gammas = gammas / 20
    sigma_delta = np.sqrt(2) * sigma_delta_0 * (1 + (abs(taus) / tau_0))

    # Run EC stage
    d, p_ec_max = ec(
        xl_hat,
        xr_hat,
        yl_hat,
        yr_hat,
        J,
        N,
        fids,
        cf.flatten(),
        taus,
        ntaus,
        gammas,
        ngammas,
        d,
        p_ec_max,
        sigma_epsilon,
        sigma_delta,
    )

    # Compute the better ear STOI
    # Arrays for the 1/3 octave envelope
    Xl = np.zeros((J, np.shape(xl_hat)[1]))
    Xr = np.zeros((J, np.shape(xl_hat)[1]))
    Yl = np.zeros((J, np.shape(xl_hat)[1]))
    Yr = np.zeros((J, np.shape(xl_hat)[1]))

    # Apply 1/3 octave bands as described in Eq.(1) of the STOI article
    for k in range(np.shape(xl_hat)[1]):
        Xl[:, k] = np.dot(H, abs(xl_hat[:, k]) ** 2)
        Xr[:, k] = np.dot(H, abs(xr_hat[:, k]) ** 2)
        Yl[:, k] = np.dot(H, abs(yl_hat[:, k]) ** 2)
        Yr[:, k] = np.dot(H, abs(yr_hat[:, k]) ** 2)

    # Arrays for better-ear correlations
    dl_interm = np.zeros((J, len(range(N, len(xl_hat[1]) + 1))))
    dr_interm = np.zeros((J, len(range(N, len(xl_hat[1]) + 1))))
    pl = np.zeros((J, len(range(N, len(xl_hat[1]) + 1))))
    pr = np.zeros((J, len(range(N, len(xl_hat[1]) + 1))))

    # Compute temporary better-ear correlations
    for m in range(N, np.shape(xl_hat)[1]):
        Xl_seg = Xl[:, (m - N) : m]
        Xr_seg = Xr[:, (m - N) : m]
        Yl_seg = Yl[:, (m - N) : m]
        Yr_seg = Yr[:, (m - N) : m]

        for n in range(J):
            xln = Xl_seg[n, :] - np.sum(Xl_seg[n, :]) / N
            xrn = Xr_seg[n, :] - np.sum(Xr_seg[n, :]) / N
            yln = Yl_seg[n, :] - np.sum(Yl_seg[n, :]) / N
            yrn = Yr_seg[n, :] - np.sum(Yr_seg[n, :]) / N
            pl[n, m - N] = np.sum(xln * xln) / np.sum(yln * yln)
            pr[n, m - N] = np.sum(xrn * xrn) / np.sum(yrn * yrn)
            dl_interm[n, m - N] = np.sum(xln * yln) / (
                np.linalg.norm(xln) * np.linalg.norm(yln)
            )
            dr_interm[n, m - N] = np.sum(xrn * yrn) / (
                np.linalg.norm(xrn) * np.linalg.norm(yrn)
            )

    # Get the better ear intermediate coefficients
    idx = np.isfinite(dl_interm)
    dl_interm[~idx] = 0
    idx = np.isfinite(dr_interm)
    dr_interm[~idx] = 0
    p_be_max = np.maximum(pl, pr)
    dbe_interm = np.zeros((np.shape(dl_interm)))

    idx = pl > pr
    dbe_interm[idx] = dl_interm[idx]
    dbe_interm[~idx] = dr_interm[~idx]

    # Compute STOI measure
    # Whenever a single ear provides a higher correlation than the corresponding EC
    # processed alternative, the better-ear correlation is used.
    idx = p_be_max > p_ec_max
    d[idx] = dbe_interm[idx]
    sii = np.mean(d)

    return sii

# Simple test case
if __name__ == "__main__":
    import argparse
    import soundfile as sf
    
    parser = argparse.ArgumentParser(description="Calculate MBSTOI for binaural audio files")
    parser.add_argument("--clean_left", type=str, help="Path to clean left channel audio")
    parser.add_argument("--clean_right", type=str, help="Path to clean right channel audio")
    parser.add_argument("--proc_left", type=str, help="Path to processed left channel audio")
    parser.add_argument("--proc_right", type=str, help="Path to processed right channel audio")
    parser.add_argument("--fs", type=int, default=16000, help="Sampling rate")
    
    args = parser.parse_args()
    
    if args.clean_left and args.clean_right and args.proc_left and args.proc_right:
        # Load audio files
        clean_left, sr = sf.read(args.clean_left)
        clean_right, _ = sf.read(args.clean_right)
        proc_left, _ = sf.read(args.proc_left)
        proc_right, _ = sf.read(args.proc_right)
        
        # Calculate MBSTOI
        mbstoi_score = mbstoi(clean_left, clean_right, proc_left, proc_right, fs_signal=args.fs)
        print(f"MBSTOI score: {mbstoi_score:.4f}")
    else:
        # Generate test signals if no files provided
        print("No audio files provided. Running with test signals...")
        
        # Create simple test signals
        fs_signal = 16000
        t = np.linspace(0, 1, fs_signal)
        clean_left = np.sin(2 * np.pi * 440 * t)
        clean_right = np.sin(2 * np.pi * 440 * t + 0.1)  # Slight phase shift
        
        # Create noisy versions
        np.random.seed(42)
        noise_left = np.random.normal(0, 0.1, len(clean_left))
        noise_right = np.random.normal(0, 0.1, len(clean_right))
        proc_left = clean_left + noise_left
        proc_right = clean_right + noise_right
        
        # Calculate MBSTOI
        mbstoi_score = mbstoi(clean_left, clean_right, proc_left, proc_right, fs_signal=fs_signal)
        print(f"MBSTOI score for test signals: {mbstoi_score:.4f}")