import sys
import cupyx.profiler as profiler
import numpy as np

import benchmarking.common as common
import optic.carrierRecovery as carrierRecovery


def main():
    alg = "bps" if len(sys.argv) <= 1 else sys.argv[1]
    loops = 0 if len(sys.argv) <= 2 else int(sys.argv[2])

    sigRx = common.getSigRx()
    paramCPR = common.get_paramCPR(alg)

    with profiler.time_range(f"{alg}", color_id=0):
        y_CPR, θ = carrierRecovery.cpr(sigRx, paramCPR=paramCPR)

    for _ in range(loops):
        with profiler.time_range(f"{alg} loop", color_id=0):
            y_CPR, θ = carrierRecovery.cpr(sigRx, paramCPR=paramCPR)

    expected_y_CPR, expected_θ = common.get_expected_BPS_results()

    np.testing.assert_allclose(y_CPR, expected_y_CPR)
    np.testing.assert_allclose(θ, expected_θ)


sys.exit(main())
