#!/bin/bash
cd ..

LOCAL_RESULTS_PATH="results/ExpRewardNormTransfqmixMe"

for i in {1..3}
do
    COMMAND="python src/main.py alg-config=transf_qmix env-config=mpe/multi agent=n_agent_care local_results_path=${LOCAL_RESULTS_PATH} emb=32 heads=1 depth=3 ff_hidden_mult=1 mixer_emb=32 mixer_heads=1 buffer_cpu_only=False t_max=1000000"
    echo "Running command: ${COMMAND}"
    ${COMMAND}
done
