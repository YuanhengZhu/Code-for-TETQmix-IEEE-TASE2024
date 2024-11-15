#!/bin/bash
cd ..

LOCAL_RESULTS_PATH="results/ExpRewardNormTransfqmixTaskImp"

# 如果用户传入了一个数字参数，就使用该数字作为循环次数，否则默认循环3次
if [ -n "$1" ] && [ "$1" -eq "$1" ] 2>/dev/null; then
    LOOP_COUNT="$1"
else
    LOOP_COUNT=3
fi

for i in $(seq 1 "$LOOP_COUNT")
do
    COMMAND="python src/main.py alg-config=transf_qmix env-config=mpe/multi_with_task local_results_path=${LOCAL_RESULTS_PATH} run=run_imp emb=32 heads=1 depth=3 ff_hidden_mult=1 mixer_emb=32 mixer_heads=1 buffer_cpu_only=False t_max=1000000"
    echo "Running command: ${COMMAND}"
    ${COMMAND}
done
