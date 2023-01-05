import matplotlib.pyplot as plt
import numpy as np
from numba import njit, cuda
from numpy.fft import fft, fftfreq, fftshift

from optic.dsp import pnorm
from optic.modulation import GrayMapping

import cupy as cp
import yappi

yappi.set_clock_type("wall")


def cpr(Ei, symbTx=[], paramCPR=[]):
    """
    Carrier phase recovery function (CPR)

    Parameters
    ----------
    Ei : complex-valued ndarray
        received constellation symbols.
    symbTx :complex-valued ndarray, optional
        Transmitted symbol sequence. The default is [].
    paramCPR : core.param object, optional
        configuration parameters. The default is [].

        BPS params:

        paramCPR.alg: CPR algorithm to be used ['bps' or 'ddpll']
        paramCPR.M: constellation order. The default is 4.
        paramCPR.N: length of BPS the moving average window. The default is 35.    
        paramCPR.B: number of BPS test phases. The default is 64.

        DDPLL params:

        paramCPR.tau1: DDPLL loop filter param. 1. The default is 1/2*pi*10e6.
        paramCPR.tau2: DDPLL loop filter param. 2. The default is 1/2*pi*10e6.
        paramCPR.Kv: DDPLL loop filter gain. The default is 0.1.
        paramCPR.Ts: symbol period. The default is 1/32e9.
        paramCPR.pilotInd: indexes of pilot-symbol locations.

    Raises
    ------
    ValueError
        Error is generated if the CPR algorithm is not correctly
        passed.

    Returns
    -------
    Eo : complex-valued ndarray
        Phase-compensated signal.
    θ : real-valued ndarray
        Time-varying estimated phase-shifts.

    """
    # check input parameters
    alg = getattr(paramCPR, "alg", "bps")
    M = getattr(paramCPR, "M", 4)
    constType = getattr(paramCPR, 'constType', 'qam')
    B = getattr(paramCPR, "B", 64)
    N = getattr(paramCPR, "N", 35)
    Kv = getattr(paramCPR, "Kv", 0.1)
    tau1 = getattr(paramCPR, "tau1", 1 / (2 * np.pi * 10e6))
    tau2 = getattr(paramCPR, "tau2", 1 / (2 * np.pi * 10e6))
    Ts = getattr(paramCPR, "Ts", 1 / 32e9)
    pilotInd = getattr(paramCPR, "pilotInd", np.array([len(Ei) + 1]))

    try:
        Ei.shape[1]
    except IndexError:
        Ei = Ei.reshape(len(Ei), 1)

    # constellation parameters
    constSymb = GrayMapping(M, constType)
    constSymb = pnorm(constSymb)

    # 4th power frequency offset estimation/compensation
    Ei, _ = fourthPowerFOE(Ei, 1/Ts)
    Ei = pnorm(Ei)

    if alg == "ddpll":
        θ = ddpll(Ei, Ts, Kv, tau1, tau2, constSymb, symbTx, pilotInd)
    elif alg == "bps":
        θ = bps(Ei, N // 2, constSymb, B)
    elif alg == "bpsGPU":
        θ = bpsGPU(Ei, N // 2, constSymb, B)
    else:
        raise ValueError("CPR algorithm incorrectly specified.")
    θ = np.unwrap(4 * θ, axis=0) / 4

    Eo = Ei * np.exp(1j * θ)

    if Eo.shape[1] == 1:
        Eo = Eo[:]
        θ = θ[:]
    return Eo, θ


# @njit
def bps(Ei, N, constSymb, B):
    """
    Blind phase search (BPS) algorithm

    Parameters
    ----------
    Ei : complex-valued ndarray
        Received constellation symbols.
    N : int
        Half of the 2*N+1 average window.
    constSymb : complex-valued ndarray
        Complex-valued constellation.
    B : int
        number of test phases.

    Returns
    -------
    θ : real-valued ndarray
        Time-varying estimated phase-shifts.

    """
    nModes = Ei.shape[1]

    ϕ_test = np.arange(0, B) * (np.pi / 2) / B  # test phases

    θ = np.zeros(Ei.shape, dtype="float")

    zeroPad = np.zeros((N, nModes), dtype="complex")
    x = np.concatenate(
        (zeroPad, Ei, zeroPad)
    )  # pad start and end of the signal with zeros

    L = x.shape[0]

    for n in range(nModes):

        dist = np.zeros((B, constSymb.shape[0]), dtype="float")
        dmin = np.zeros((B, 2 * N + 1), dtype="float")

        for k in range(L):
            for indPhase, ϕ in enumerate(ϕ_test):
                # For each test phase, compute the distances to all constellation symbols
                dist[indPhase, :] = np.abs(
                    x[k, n] * np.exp(1j * ϕ) - constSymb) ** 2

                # Save the minimum distance for the test phase
                dmin[indPhase, -1] = np.min(dist[indPhase, :])
            if k >= 2 * N:
                sumDmin = np.sum(dmin, axis=1)
                indRot = np.argmin(sumDmin)
                θ[k - 2 * N, n] = ϕ_test[indRot]
            dmin = np.roll(dmin, -1)
    return θ


@njit
def gen_test_phases(num_rotations: int) -> np.ndarray:
    return np.arange(num_rotations) * (np.pi / 2) / num_rotations


# @njit
def min_dist_to_constellation_symbols(symbols, const_symb):
    new_shape = (len(const_symb), len(symbols))
    resized_symbols = np.resize(symbols, new_shape).transpose()

    dist_to_each_const_symbol = np.abs(resized_symbols - const_symb) ** 2
    return dist_to_each_const_symbol.min(axis=1)


# @njit
def get_optimum_phase_angle(ϕ_test, dmin):
    """
    The optimum phase angle is determined by searching the minimum sum of
    distance values.
    """
    sumDmin = np.sum(dmin, axis=1)
    optimum_phase_index = np.argmin(sumDmin)
    return ϕ_test[optimum_phase_index]


# @njit
def bpsGPU2(Ei, N, constSymb, B, prec=cp.complex128):
    nModes = Ei.shape[1]

    ϕ_test = gen_test_phases(B)

    θ = np.zeros(Ei.shape, dtype="float")

    zeroPad = np.zeros((N, nModes), dtype="complex")
    x = np.concatenate(
        (zeroPad, Ei, zeroPad)
    )  # pad start and end of the signal with zeros

    num_symbols_after_padding = x.shape[0]

    # start = cuda.grid(1)

    for mode_index in range(nModes):

        # Minimum distance for each test phase
        dmin = np.zeros((B, 2 * N + 1), dtype="float")

        for symbol_index in range(num_symbols_after_padding):
            symbol = x[symbol_index, mode_index]
            symbol_rotations = symbol * np.exp(1j * ϕ_test)

            dmin = min_dist_to_constellation_symbols(
                symbol_rotations, constSymb)
            # for indPhase, ϕ in enumerate(ϕ_test):
            #     rotated_symbol = symbol * np.exp(1j * ϕ)
            #     dmin[indPhase, -1] = min_dist_to_constellation_symbols(
            #         rotated_symbol, constSymb)
            if symbol_index >= 2 * N:
                θ[symbol_index - 2 * N,
                    mode_index] = get_optimum_phase_angle(ϕ_test, dmin)
            dmin = np.roll(dmin, -1)


def bpsDistances(x, ϕ_test, constSymb):
    x_gpu = cp.asarray(x)
    ϕ_test_gpu = cp.asarray(ϕ_test)
    constSymb_gpu = cp.asarray(constSymb)

    x_expanded = x_gpu[:, :, cp.newaxis]
    ϕ_expanded = cp.exp(1j * ϕ_test_gpu)[None, None, :]
    rotated_x = x_expanded * ϕ_expanded
    constSymb_expanded = constSymb_gpu[None, None, None, :]
    dist = cp.absolute(cp.subtract(
        rotated_x[:, :, :, None], constSymb_expanded)) ** 2

    return cp.asnumpy(dist)


def bpsDistancesNumpy(x, ϕ_test, constSymb):
    x_expanded = x[:, :, np.newaxis]
    ϕ_expanded = np.exp(1j * ϕ_test)[None, None, :]
    rotated_x = x_expanded * ϕ_expanded
    constSymb_expanded = constSymb[None, None, None, :]
    dist = np.abs(np.subtract(
        rotated_x[:, :, :, None], constSymb_expanded)) ** 2

    return dist


def bpsGPU(Ei, N, constSymb, B):
    yappi.start()
    # Ei = Ei[:5]
    nModes = Ei.shape[1]

    ϕ_test = np.arange(0, B) * (np.pi / 2) / B  # test phases

    θ = np.zeros(Ei.shape, dtype="float")

    zeroPad = np.zeros((N, nModes), dtype="complex")
    x = np.concatenate(
        (zeroPad, Ei, zeroPad)
    )  # pad start and end of the signal with zeros

    L = x.shape[0]
    dmin = np.zeros((B, 2 * N + 1), dtype="float")

    # x.shape = (100084,2)
    # ϕ_test.shape = (64,)
    # rotated_x.shape = (100084, 2, 64) -> The element rotated_x[500][0] has the 64 rotations of input x[500][0]
    # dist.shape = (100084, 2, 64, 16)

    dist_cupy = bpsDistances(x, ϕ_test, constSymb)
    # dist = bpsDistancesNumpy(x, ϕ_test, constSymb)
    # assert np.array_equiv(dist.round(decimals=5), dist_cupy.round(decimals=5))

    yappi.get_func_stats().print_all()
    yappi.get_func_stats().save("ystats1.ys")

    import sys
    sys.exit(1)

    dmin = np.roll(dmin, -1, axis=-1)
    sumDmin = np.sum(dmin, axis=1)
    indRot = np.argmin(sumDmin, axis=-1)
    θ = ϕ_test[indRot]

    return θ


def bpsGPU3(Ei, N, constSymb, B):
    Ei = Ei[:100]
    nModes = Ei.shape[1]

    ϕ_test = np.arange(0, B) * (np.pi / 2) / B  # test phases

    zeroPad = np.zeros((N, nModes), dtype="complex")
    x = np.concatenate(
        (zeroPad, Ei, zeroPad)
    )  # pad start and end of the signal with zeros

    θ = np.zeros(x.shape, dtype="float")
    dmin = np.zeros((B, 2 * N + 1), dtype="float")

    x_expanded = x[:, :, np.newaxis]
    ϕ_expanded = np.exp(1j * ϕ_test)[None, None, :]
    rotated_x = x_expanded * ϕ_expanded
    constSymb_expanded = constSymb[None, None, None, :]
    dist = np.abs(np.subtract(
        rotated_x[:, :, :, None], constSymb_expanded)) ** 2
    dist = np.squeeze(dist)  # shape: (100084, 2, 64)
    dmin[:, -1] = np.min(dist, axis=-1)
    dmin = np.roll(dmin, -1, axis=-1)
    sumDmin = np.sum(dmin, axis=1)
    indRot = np.argmin(sumDmin, axis=-1)
    θ[N:-N] = ϕ_test[indRot]  # shape: (100084,)

    return θ


@njit
def ddpll(Ei, Ts, Kv, tau1, tau2, constSymb, symbTx, pilotInd):
    """
    Decision-directed Phase-locked Loop (DDPLL) algorithm

    Parameters
    ----------
    Ei : complex-valued ndarray
        Received constellation symbols.
    Ts : float scalar
        Symbol period.
    Kv : float scalar
        Loop filter gain.
    tau1 : float scalar
        Loop filter parameter 1.
    tau2 : float scalar
        Loop filter parameter 2.
    constSymb : complex-valued ndarray
        Complex-valued ideal constellation symbols.
    symbTx : complex-valued ndarray
        Transmitted symbol sequence.
    pilotInd : int ndarray
        Indexes of pilot-symbol locations.

    Returns
    -------
    θ : real-valued ndarray
        Time-varying estimated phase-shifts.

    References
    -------
    [1] H. Meyer, Digital Communication Receivers: Synchronization, Channel 
    estimation, and Signal Processing, Wiley 1998. Section 5.8 and 5.9.    

    """
    nModes = Ei.shape[1]

    θ = np.zeros(Ei.shape)

    # Loop filter coefficients
    a1b = np.array(
        [
            1,
            Ts / (2 * tau1) * (1 - 1 / np.tan(Ts / (2 * tau2))),
            Ts / (2 * tau1) * (1 + 1 / np.tan(Ts / (2 * tau2))),
        ]
    )

    u = np.zeros(3)  # [u_f, u_d1, u_d]

    for n in range(nModes):

        u[2] = 0  # Output of phase detector (residual phase error)
        u[0] = 0  # Output of loop filter

        for k in range(len(Ei)):
            u[1] = u[2]

            # Remove estimate of phase error from input symbol
            Eo = Ei[k, n] * np.exp(1j * θ[k, n])

            # Slicer (perform hard decision on symbol)
            if k in pilotInd:
                # phase estimation with pilot symbol
                # Generate phase error signal (also called x_n (Meyer))
                u[2] = np.imag(Eo * np.conj(symbTx[k, n]))
            else:
                # find closest constellation symbol
                decided = np.argmin(np.abs(Eo - constSymb))
                # Generate phase error signal (also called x_n (Meyer))
                u[2] = np.imag(Eo * np.conj(constSymb[decided]))
            # Pass phase error signal in Loop Filter (also called e_n (Meyer))
            u[0] = np.sum(a1b * u)

            # Estimate the phase error for the next symbol
            θ[k + 1, n] = θ[k, n] - Kv * u[0]
    return θ


def fourthPowerFOE(Ei, Fs, plotSpec=False):
    """
    4th power frequency offset estimator (FOE).

    Parameters
    ----------
    Ei : np.array
        Input signal.
    Fs : real scalar
        Sampling frequency.
    plotSpec : bolean, optional
        Plot spectrum. The default is False.

    Returns
    -------
    Real scalar
        Estimated frequency offset.

    """
    Nfft = Ei.shape[0]

    f = Fs * fftfreq(Nfft)
    f = fftshift(f)

    nModes = Ei.shape[1]
    Eo = Ei.copy()
    t = np.arange(0, Eo.shape[0])*1/Fs

    for n in range(nModes):
        f4 = 10 * np.log10(np.abs(fftshift(fft(Ei[:, n] ** 4))))
        indFO = np.argmax(f4)
        fo = f[indFO] / 4
        Eo[:, n] = Ei[:, n] * np.exp(-1j * 2 * np.pi * fo * t)

    if plotSpec:
        plt.figure()
        plt.plot(f, f4, label="$|FFT(s[k]^4)|[dB]$")
        plt.plot(f[indFO], f4[indFO], "x", label="$4f_o$")
        plt.legend()
        plt.xlim(min(f), max(f))
        plt.grid()
    return Eo, f[indFO] / 4
