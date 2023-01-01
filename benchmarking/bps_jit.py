#!/usr/bin/python3

import sys
import numpy as np
from cupy import prof
import cupyx.profiler as profiler
import optic.core as opti_core
import optic.carrierRecovery as carrierRecovery


def get_paramCPR():
    paramCPR = opti_core.parameters()
    paramCPR.alg = "bps"
    paramCPR.M = 16
    paramCPR.constType = "qam"
    paramCPR.N = 85
    paramCPR.B = 64

    return paramCPR


def getSigRx():
    return np.load("benchmarking/sigRx.npy")


def main():
    loops = int(sys.argv[1])

    sigRx = getSigRx()
    paramCPR = get_paramCPR()

    # with profiler.time_range("bps_jit", color_id=0):
    with prof.time_range("bps_jit", color_id=0):
        y_CPR, θ = carrierRecovery.cpr(sigRx, paramCPR=paramCPR)

    for _ in range(loops):
        # with profiler.time_range("bps_jit loop", color_id=0):
        with prof.time_range("bps_jit loop", color_id=0):
            y_CPR, θ = carrierRecovery.cpr(sigRx, paramCPR=paramCPR)
    # print(y_CPR)
    # print(θ)


if __name__ == "__main__":
    sys.exit(main())
