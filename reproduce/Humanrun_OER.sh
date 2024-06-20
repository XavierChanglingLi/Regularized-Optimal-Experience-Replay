#!/bin/bash -l

case $SLURM_ARRAY_TASK_ID in
    0)  seed=42 ;; 
    1)  seed=43 ;; 
    2)  seed=44 ;; 
    3)  seed=45 ;;
    4)  seed=46 ;;
    
    5)  seed=47 ;;
    6)  seed=48 ;; 
    7)  seed=49 ;; 
    8)  seed=50 ;; 
    9)  seed=51 ;; 
    
    10)  seed=52 ;;
    11)  seed=53 ;; 
    12)  seed=54 ;;
    13)  seed=55 ;;
    14)  seed=56 ;;
    
    15)  seed=57 ;;
    16)  seed=58 ;;
    17)  seed=59 ;;
    18)  seed=60 ;;
    19)  seed=61 ;;
esac

srun python train_online.py --env_name=humanoid-run --eval_save_dir=./tmp/evaluation/humanoidrun/ --per=True --per_type=OER --min_clip=1 --max_clip=100 --gumbel_max_clip=7 --temp=4 --seed=${seed}