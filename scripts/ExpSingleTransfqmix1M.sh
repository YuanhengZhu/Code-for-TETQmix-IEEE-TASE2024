#!/bin/bash
cd ..

LOCAL_RESULTS_PATH="results/ExpSingleTransfqmix1M"

# 如果用户传入了一个数字参数，就使用该数字作为循环次数，否则默认循环3次
if [ -n "$1" ] && [ "$1" -eq "$1" ] 2>/dev/null; then
    LOOP_COUNT="$1"
else
    LOOP_COUNT=3
fi

# envs=("spread" "form_shape" "push" "tag1" "tag2")
envs=("spread" "form_shape" "push")

for env in "${envs[@]}"
do
    for i in $(seq 1 "$LOOP_COUNT")
    do
        COMMAND="python src/main.py alg-config=transf_qmix env-config=mpe/${env} local_results_path=${LOCAL_RESULTS_PATH} buffer_cpu_only=False t_max=1000000"
        echo "Running command: ${COMMAND}"
        ${COMMAND}
    done
done
