#!/bin/bash -l

case $SLURM_ARRAY_TASK_ID in
    0)   seed=42 ;; 
    1)   seed=43 ;; 
    2)   seed=44 ;; 
    3)   seed=45 ;; 
    4)   seed=46 ;;

    5)   seed=47 ;; 
    6)   seed=48 ;; 
    7)   seed=49 ;; 
    8)   seed=50 ;; 
    9)   seed=51 ;; 
esac

srun python train_online_pretrain.py --env_name=antmaze-medium-diverse-v2 --eval_save_dir=./tmp/evaluation/mediumDiverse/ --per=True --per_type=OER --min_clip=1 --max_clip=50 --gumbel_max_clip=7 --temp=0.4 --seed=${seed} --max_steps=2000000