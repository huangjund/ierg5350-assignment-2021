"""
Credit: https://github.com/sfujim/TD3
"""
import argparse
import copy
import os
import os.path as osp
import sys
from collections import deque
from pprint import pprint

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

current_dir = osp.join(osp.abspath(osp.dirname(__file__)))
sys.path.append(current_dir)
sys.path.append(osp.dirname(current_dir))
# print(current_dir)

from core.envs import make_envs
from core.utils import summary, save_progress

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class TD3Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(TD3Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TD3Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3Trainer:
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            lr=5e-5  # Small LR in TD3 is important to train in MetaDrive!
    ):
        self.actor = TD3Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = TD3Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # [TODO] Following the TODOs below to implement critic loss
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # [TODO] Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy,eval_env, seed, eval_episodes=10):
    eval_env.seed(seed + 100)

    avg_reward = 0.
    reward_recorder = deque(maxlen=eval_episodes)
    success_recorder = deque(maxlen=eval_episodes)

    for i in range(eval_episodes):
        state, done = eval_env.reset(), False
        ep_reward = 0
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, info = eval_env.step(action)
            ep_reward += reward
            if done:
                success = info["arrive_dest"]
                success_recorder.append(float(success))
                reward_recorder.append(float(ep_reward))
                print("Episode {}, reward: {}, success rate: {}".format(i + 1, ep_reward,
                                                                        np.mean(success_recorder)))

    eval_env.close()
    stats = dict(
        training_episode_reward=summary(reward_recorder, "episode_reward"),
        success_rate=summary(success_recorder, "success_rate"),
        env_name=eval_env,
        log_dir=log_dir
    )

    return stats

# def compare(policy,eval_env, seed=0, eval_episodes=50,load_dir="MetaDrive1Env/TD3/models"):

#     if args.load_dir:
#         policy.load(f"{args.load_dir}/default") # to accelerate training
#     stats = eval_policy(policy, env, seed, eval_episodes=episode_num)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="MetaDrive-Tut-Test-v0")
    parser.add_argument("--seed", default=0)
    parser.add_argument("-lr", default=5e-5, type=float)
    parser.add_argument(
        "--log-dir",
        default="data/",
        type=str,
        help="The path of directory that you want to store the data to. "
             "Default: ./data/"
    )
    parser.add_argument("--start_timesteps", default=1e4, type=int)  # Time steps initial random policy is used
    parser.add_argument("--save_freq", default=5e3, type=int)  # How often (time steps) we save model
    parser.add_argument("--log_freq", default=1000, type=int)  # How often (time steps) we save model
    parser.add_argument("--max_timesteps", default=1e5, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--load_dir", default="MetaDrive20Env/TD3/models")
    args = parser.parse_args()

    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(osp.join(log_dir, "results")):
        os.makedirs(osp.join(log_dir, "results"))

    if not os.path.exists(osp.join(log_dir, "models")):
        os.makedirs(osp.join(log_dir, "models"))

    # Setup some stats helpers
    success_recorder = deque(maxlen=100)
    reward_recorder = deque(maxlen=100)
    progress=[]
    total_steps = 0

    environments = make_envs(
        env_id=args.env_id,
        log_dir=log_dir,
        num_envs=1,
        asynchronous=False,
    )
    env = environments.envs[0]

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    kwargs["lr"] = args.lr
    policy = TD3Trainer(**kwargs)
    if args.load_dir:
        policy.load(f"{args.load_dir}/default") # to accelerate training

    discrete = False
    max_size = 1e-6
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # state, done = env.reset(), False
    # episode_reward = 0
    # episode_timesteps = 0
    episode_num = 50
    seed = 0
    stats = eval_policy(policy, env, seed, eval_episodes=episode_num)
    pprint(stats)

    # compare(policy,env,seed,eval_episodes=episode_num,load_dir)
