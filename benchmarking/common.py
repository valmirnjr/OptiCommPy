import numpy as np
import optic.core as opti_core


def get_paramCPR(alg: str = "bps"):
    paramCPR = opti_core.parameters()
    paramCPR.alg = alg
    paramCPR.M = 16
    paramCPR.constType = "qam"
    paramCPR.N = 85
    paramCPR.B = 64

    return paramCPR


def getSigRx():
    return np.load("benchmarking/sigRx.npy")


def get_expected_BPS_results():
    y_CPR = np.load("benchmarking/expected_y_CPR.npy")
    θ = np.load("benchmarking/expected_θ.npy")
    return y_CPR, θ
