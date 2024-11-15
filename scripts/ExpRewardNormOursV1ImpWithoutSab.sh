#!/bin/bash
cd ..

LOCAL_RESULTS_PATH="results/ExpRewardNormOursV1ImpWithoutSab"

# 如果用户传入了一个数字参数，就使用该数字作为循环次数，否则默认循环3次
if [ -n "$1" ] && [ "$1" -eq "$1" ] 2>/dev/null; then
    LOOP_COUNT="$1"
else
    LOOP_COUNT=3
fi

for i in $(seq 1 "$LOOP_COUNT")
do
    COMMAND="python src/main.py alg-config=transf_qmix_task_cross env-config=mpe/multi run=run_imp agent=n_transf_task_cross_v1_without_sab local_results_path=${LOCAL_RESULTS_PATH} buffer_cpu_only=False"
    echo "Running command: ${COMMAND}"
    ${COMMAND}
done
