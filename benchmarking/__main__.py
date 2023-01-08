import os
import sys
import cupyx.profiler as profiler
import cupy as cp
import numpy as np

import benchmarking.common as common
import optic.carrierRecovery as carrierRecovery


def get_bps_fn(alg_name: str):
    name_to_fn = {
        "bps": carrierRecovery.bps,
        "bpsGPU": carrierRecovery.bpsGPU,
    }
    return name_to_fn[alg_name]


def assert_bps_result(θ):
    _, expected_θ = common.get_expected_BPS_results()
    np.testing.assert_allclose(θ, expected_θ)


def main():
    alg = "bps" if len(sys.argv) <= 1 else sys.argv[1]
    loops = 0 if len(sys.argv) <= 2 else int(sys.argv[2])

    sigRx = common.getSigRx()
    paramCPR = common.get_paramCPR(alg)

    with profiler.profile():
        y_CPR, θ = carrierRecovery.cpr(sigRx, paramCPR=paramCPR)
    os.system("echo passed!")
    y_CPR, θ = carrierRecovery.cpr(sigRx, paramCPR=paramCPR)

    # print(cp.cuda.get_nvcc_path())
    # for _ in range(loops):
    #     os.system("echo loop = " + str(_))
    #     y_CPR, θ = carrierRecovery.cpr(sigRx, paramCPR=paramCPR)
    assert_bps_result(θ)


if __name__ == "__main__":
    sys.exit(main())
