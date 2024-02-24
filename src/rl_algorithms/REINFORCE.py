from collections import deque
from typing import Callable, Any

import torch
import gymnasium
from torch import nn
import torch.nn.functional as F


class REINFORCEwSTM:
    episode_action_log_probs: list[torch.Tensor] = []
    episode_rewards: list[float] = []

    def __init__(
            self,
            env: gymnasium.Env,
            init_policy_and_optimizer: Callable[[], tuple[nn.Module, torch.optim.Optimizer]],
            select_action: Callable[[torch.tensor], tuple[Any, torch.Tensor]],
            gamma=0.99,
            on_episode_done: Callable[['REINFORCEwSTM', bool, float], None]
                = lambda _self, is_best_episode, best_total_reward: None,
            on_optimization_done: Callable[['REINFORCEwSTM', bool, float], None]
                = lambda _self, is_best_episode, best_total_reward: None,
    ):
        self.env = env
        self.init_policy_and_optimizer = init_policy_and_optimizer
        self.select_action = select_action
        self.gamma = gamma
        self.on_episode_done = on_episode_done
        self.on_optimization_done = on_optimization_done

        self.policy_network, self.optimizer = self.init_policy_and_optimizer()

    def optimize_using_episode(self):
        reinforce_objectives = []
        returns = deque()
        discounted_return = 0.0
        for r in self.episode_rewards[::-1]:
            discounted_return = r + self.gamma * discounted_return
            returns.appendleft(discounted_return)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 10**-6)

        for log_prob, discounted_return in zip(self.episode_action_log_probs, returns):
            reinforce_objectives.append(-log_prob * discounted_return)

        self.optimizer.zero_grad()

        reinforce_objective = torch.cat(reinforce_objectives).mean()
        reinforce_objective.backward()

        self.optimizer.step()

        return returns, reinforce_objective

    def find_optimal_policy(self, num_episodes: int):
        best_total_reward = 0

        for i_episode in range(num_episodes):
            state, _ = self.env.reset()
            info = {}

            max_timestep = 1000000
            timestep = 0
            for timestep in range(1, max_timestep):  # Don't infinite loop while learning
                action_pred = self.policy_network(torch.tensor(state).float())
                action, action_log_probs = self.select_action(action_pred)
                state, reward, done, truncated, info = self.env.step(action)

                self.episode_action_log_probs.append(action_log_probs)
                self.episode_rewards.append(float(reward))

                if done:
                    break

            if timestep == max_timestep - 1:
                info['termination_reason'] = 'timestep_limit_reached'

            episode_total_reward = sum(self.episode_rewards)

            is_best_episode = False
            if episode_total_reward >= best_total_reward:
                best_total_reward = episode_total_reward
                is_best_episode = True

            self.on_episode_done(
                self,
                is_best_episode,
                best_total_reward
            )

            self.optimize_using_episode()

            self.on_optimization_done(
                self,
                is_best_episode,
                best_total_reward
            )

            del self.episode_rewards[:]
            del self.episode_action_log_probs[:]
