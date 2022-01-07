#!/bin/bash
#python train.py --env-id CartPole-v0 --algo A2C --log-dir CartPole --num-envs 4 --num-steps 256 --max-steps 300000
#
#python train.py --env-id MetaDrive-Tut-Easy-v0 --algo A2C --log-dir MetaDriveEasy --num-envs 4 --num-steps 256 --max-steps 300000
#
#python train.py --env-id CartPole-v0 --algo PPO --log-dir CartPole --num-envs 4 --num-steps 256 --max-steps 300000
#
#python train.py --env-id MetaDrive-Tut-Easy-v0 --algo PPO --log-dir MetaDriveEasy --num-envs 4 --num-steps 256 --max-steps 500000

#python train.py --env-id MetaDrive-Tut-Hard-v0 --algo PPO --log-dir MetaDriveHard --num-envs 10 --num-steps 256 --max-steps 10000000 -lr 8e-5

python train.py --env-id MetaDrive-Tut-1Env-v0 --algo PPO --log-dir MetaDrive1Env --num-envs 10 --num-steps 1024 --max-steps 10000000 -lr 8e-5
python train.py --env-id MetaDrive-Tut-5Env-v0 --algo PPO --log-dir MetaDrive5Env --num-envs 10 --num-steps 1024 --max-steps 10000000- lr 8e-5
python train.py --env-id MetaDrive-Tut-10Env-v0 --algo PPO --log-dir MetaDrive10Env --num-envs 10 --num-steps 1024 --max-steps 10000000 -lr 8e-5
python train.py --env-id MetaDrive-Tut-20Env-v0 --algo PPO --log-dir MetaDrive20Env --num-envs 10 --num-steps 1024 --max-steps 10000000 -lr 8e-5
python train.py --env-id MetaDrive-Tut-50Env-v0 --algo PPO --log-dir MetaDrive50Env --num-envs 10 --num-steps 1024 --max-steps 10000000 -lr 8e-5
python train.py --env-id MetaDrive-Tut-100Env-v0 --algo PPO --log-dir MetaDrive100Env --num-envs 10 --num-steps 1024 --max-steps 10000000 -lr 8e-5



