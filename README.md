# Regularized Optimal Experience Replay (ROER)

Official code implementation for **[ROER: Regularized Optimal Experience Replay](https://arxiv.org/abs/2407.03995)** by [Changling Li](https://scholar.google.com/citations?user=jgJvfvMAAAAJ&hl=en), [Zhang-Wei Hong](https://williamd4112.github.io/), [Pulkit Agrawal](https://people.csail.mit.edu/pulkitag/), [Divyansh Garg](https://divyanshgarg.com/) and [Joni Pajarinen](https://scholar.google.com/citations?user=-2fJStwAAAAJ&hl=en).

This repo contains the code for our proposed regularized optimal experience replay with KL-divergence which corresponds to the objective of [extreme Q-learning](https://arxiv.org/abs/2301.02328) and other baselines for comparison including uniform experience replay, [prioritized experience replay](https://arxiv.org/abs/2007.06049) and [large batch experience replay](https://arxiv.org/abs/2110.01528). The code implementation is adapted from [JAXRL](https://github.com/ikostrikov/jaxrl?tab=readme-ov-file).


## How to run the code

### Install dependencies
Python >= 3.8

```bash
pip install --upgrade pip

pip install -r requirements.txt

pip install --upgrade "jax[cuda]=0.3.15" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

### Example training code for ROER

Gym Mujoco
```bash
python train_online.py --env_name=HalfCheetah-v2 --per=True --per_type=OER --min_clip=10 --max_clip=50 --gumbel_max_clip=7 --temp=4
```

DeepMind Control
```bash
python train_online.py --env_name=hopper-stand --per=True --per_type=OER --min_clip=1 --max_clip=100 --gumbel_max_clip=7 --temp=1
```

Pretrain with D4RL AntMaze
```bash
python train_online_pretrain.py --env_name=antmaze-umaze-diverse-v2 --per=True --per_type=OER --gumbel_max_clip=7 --temp=0.4 --max_clip=50 --min_clip=1 --max_steps=2000000
```

### Reproduction

For reproducing our experiments, please run the scripts in the [reproduce](reproduce) folder for the settings we use for each environment.

## Questions
Please feel free to email us if you have any questions. 

Changling Li ([lichan@student.ethz.ch](mailto:lichan@student.ethz.ch?subject=[GitHub]%ROER))
