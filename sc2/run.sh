#!/bin/sh$
game="StarCraft2"
algo="mappo"
offline_map_name="3m"
offline_data_dir='/home/lhmeng/rlproj/offline_marl/expert_policy/on-policy/onpolicy/scripts/offline_datasets/bk/'
model_type="naive"
offline_eval_interval=1
seed_max=5

export PYTHONPATH="${PYTHONPATH}:/home/lhmeng/rlproj/offline_marl/framework/decision_transformer"
export PYTHONPATH="${PYTHONPATH}:/home/lhmeng/rlproj/offline_marl/framework/decision_transformer/sc2"

# python run_madt_sc2.py --seed 123 --context_length 3 --epochs 1000 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'StarCraft' --batch_size 128

for seed in `seq ${seed_max}`;
do
  let real_seed=123+$seed
  echo "real_seed is ${real_seed}"
  CUDA_VISIBLE_DEVICES=-1 python run_madt_sc2.py --seed $real_seed --offline_data_dir ${offline_data_dir} --offline_eval_interval ${offline_eval_interval} --offline_map_name ${offline_map_name} --context_length 1 --offline_epochs 500 --model_type ${model_type} --game ${game} --mini_batch_size 128
done
