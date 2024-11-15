#!/bin/bash
cd ..

LOCAL_RESULTS_PATH="results/ExpTransfqmixFormShape"

COMMAND="python src/main.py alg-config=transf_qmix env-config=mpe/form_shape local_results_path=${LOCAL_RESULTS_PATH} buffer_cpu_only=False t_max=200000"
echo "Running command: ${COMMAND}"
${COMMAND}

LOCAL_RESULTS_PATH="results/ExpFormShapeAndNew"

# 循环3次
for i in {1..3}
do
    COMMAND="python src/main.py alg-config=transf_qmix env-config=mpe/push_away local_results_path=${LOCAL_RESULTS_PATH} run=run_new checkpoint_path=results/ExpTransfqmixFormShape/models buffer_cpu_only=False t_max=200000"
    echo "Running command: ${COMMAND}"
    ${COMMAND}
done
