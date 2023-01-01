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

echo -e "**************************************************"
echo -e "Test bps_cuda_jit.py ${LOOPS}"
echo -e "Just-in-time compilation of BPS with CUDA"
echo -e "**************************************************"
nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 -m benchmarking bpsGPU ${LOOPS}
echo -e

