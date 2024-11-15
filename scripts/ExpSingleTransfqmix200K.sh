#!/bin/bash
cd ..

LOCAL_RESULTS_PATH="results/ExpSingleTransfqmix200K"

# ENVs=("mpe/spread" "mpe/form_shape" "mpe/push" "mpe/tag1" "mpe/tag2")

# for i in {1..3}
# do
#     for ENV in "${ENVs[@]}"
#     do
#         COMMAND="python src/main.py alg-config=transf_qmix env-config=${ENV} local_results_path=${LOCAL_RESULTS_PATH} emb=32 heads=1 depth=2 ff_hidden_mult=1 mixer_emb=32 mixer_heads=1 t_max=200000"
#         echo "Running command: ${COMMAND}"
#         ${COMMAND}
#     done
# done

for i in {1..3}
do
    COMMAND="python src/main.py alg-config=transf_qmix env-config=mpe/form_shape local_results_path=${LOCAL_RESULTS_PATH} emb=32 heads=1 depth=2 ff_hidden_mult=1 mixer_emb=32 mixer_heads=1 t_max=200000"
    echo "Running command: ${COMMAND}"
    ${COMMAND}
done