#!/usr/bin/python

import sys
from cupy import prof
import cupyx.profiler as profiler
import numpy as np

import benchmarking.common as common
import optic.carrierRecovery as carrierRecovery


def main():
    alg = "bps" if len(sys.argv) <= 1 else sys.argv[1]
    loops = 0 if len(sys.argv) <= 2 else int(sys.argv[2])

    sigRx = common.getSigRx()
    paramCPR = common.get_paramCPR(alg)

    y_CPR, θ = carrierRecovery.cpr(sigRx, paramCPR=paramCPR)

    # results = profiler.benchmark(
    #     carrierRecovery.cpr,
    #     (sigRx,),
    #     {
    #         "paramCPR": paramCPR
    #     },
    #     n_repeat=loops,
    #     n_warmup=1,
    #     name=alg
    # )
    # print(results)

    # expected_y_CPR, expected_θ = common.get_expected_BPS_results()

    # np.testing.assert_allclose(θ, expected_θ)
    # np.testing.assert_allclose(y_CPR, expected_y_CPR)


if __name__ == "__main__":
    sys.exit(main())
