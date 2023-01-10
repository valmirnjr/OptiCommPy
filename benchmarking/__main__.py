import os
import sys
import cupyx.profiler as profiler
import cupy as cp
import numpy as np

import benchmarking.common as common
import optic.carrierRecovery as carrierRecovery


def assert_bps_result(θ):
    try:
        _, expected_θ = common.get_expected_BPS_results()
        np.testing.assert_allclose(θ, expected_θ)
        print("Assertion of result passed!")
    except AssertionError as err:
        print(f"Assertion error: {err}")


def main():
    alg = "bps" if len(sys.argv) <= 1 else sys.argv[1]
    loops = 0 if len(sys.argv) <= 2 else int(sys.argv[2])

    print(f"Running carrier recovery algorithm '{alg}'.")

    sigRx = common.getSigRx()
    paramCPR = common.get_paramCPR(alg)

    y_CPR, θ = carrierRecovery.cpr(sigRx, paramCPR=paramCPR)

    for _ in range(loops):
        y_CPR, θ = carrierRecovery.cpr(sigRx, paramCPR=paramCPR)

    assert_bps_result(θ)


if __name__ == "__main__":
    sys.exit(main())
