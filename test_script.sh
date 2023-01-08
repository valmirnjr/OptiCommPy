#!/bin/bash
declare LOOPS=$1

if [ -z "$LOOPS" ];
then
	LOOPS=10
else
	LOOPS=${LOOPS}
fi

#Test BPS algorithm with numba (JIT)

# echo -e "**************************************************"
# echo -e "Test bps_jit.py ${LOOPS}"
# echo -e "Just-in-time compilation of BPS"
# echo -e "**************************************************"
# nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 -m benchmarking bps ${LOOPS}
# echo -e


#Test BPS algorithm with CUDA (JIT)

# echo -e "**************************************************"
# echo -e "Test bps_cuda_jit.py ${LOOPS}"
# echo -e "Just-in-time compilation of BPS with CUDA"
# echo -e "**************************************************"
# nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 -m benchmarking bpsGPU ${LOOPS}
# echo -e

#Test BPS algorithm with CUPY

echo -e "**************************************************"
echo -e "Test benchmark_bps.py ${LOOPS}"
echo -e "Just-in-time compilation of BPS with CUPY"
echo -e "**************************************************"
nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 -m benchmarking bpsGPU ${LOOPS}
echo -e

# Test Numba histogram w/ rounding

# echo -e "**************************************************"
# echo -e "Test numba_v1.py ${LOOPS}"
# echo -e "Numba custom histogram kernel, and forall rounding"
# echo -e "**************************************************"
# nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 numba_v1.py ${LOOPS}
# echo -e

# Test NumPy and CuPy built in implementation of histogram

# echo -e "**************************************************"
# echo -e "Test cupy_v1.py ${LOOPS}"
# echo -e "NumPy and CuPy built in implementation of histogram"
# echo -e "**************************************************"
# nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 cupy_v1.py ${LOOPS}
# echo -e


# Test Numba reduce generator for l2 norm calculation

# echo -e "**************************************************"
# echo -e "Test numba_v2.py ${LOOPS}"
# echo -e "Numba reduce generator for l2 norm"
# echo -e "**************************************************"
# nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 numba_v2.py ${LOOPS}
# echo -e

# Test CuPy reduction kernel for l2 norm calculation

# echo -e "**************************************************"
# echo -e "Test cupy_v2.py ${LOOPS}"
# echo -e "CuPy reduction kernel for l2 norm"
# echo -e "**************************************************"
# nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 cupy_v2.py ${LOOPS}
# echo -e

# Test CuPy and Numba for CuPy array as input to custom kernel

# echo -e "**************************************************"
# echo -e "Test cupy_and_numba_v1.py ${LOOPS}"
# echo -e "CuPy array as input to custom kernel"
# echo -e "**************************************************"
# nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 cupy_and_numba_v1.py ${LOOPS}
# echo -e
