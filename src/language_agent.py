import os
import imageio
from collections import OrderedDict

import torch
import gym
import numpy as np

from habitat_baselines.utils.common import batch_obs
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    get_active_obs_transforms,
)
from habitat_baselines.utils.common import get_num_actions

from scene_parser import SceneParser
from hierarchichal_policy import NewHierarchicalPolicy


class LanguageAgent:
    def __init__(self, env, conf, mappings, device):
        self.env = env
        self.conf = conf
        self.mappings = mappings
        self.device = device

        self.sim = env._sim
        self.scene_parser = SceneParser(self.sim)
        self.ppo_cfg = conf.habitat_baselines.rl.ppo

        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.__get_internal_obs_space()
        self.obs_transforms = get_active_obs_transforms(conf)

        self.policy = NewHierarchicalPolicy.from_config(
            conf, self.internal_observation_space, self.action_space
        )
        self.policy.to(device)

        self.frames = []
        self.reset_environment()

    def set_target(self, target):
        self.scene_parser.set_dynamic_target(target)

    def reset_environment(self):
        obs = self.env.reset()
        self.batch = self.__parse_obs(obs)

        self.sim = self.env._sim
        self.scene_parser = SceneParser(self.sim)
        self.reset_internals()

    def reset_internals(self):
        self.recurrent_hidden_states = torch.zeros(
            self.conf.habitat_baselines.num_environments,
            self.conf.habitat_baselines.rl.ddppo.num_recurrent_layers * 2,
            self.ppo_cfg.hidden_size,
            device=self.device,
        )
        self.prev_actions = torch.zeros(
            self.conf.habitat_baselines.num_environments,
            *(get_num_actions(self.action_space),),
            device=self.device,
            dtype=torch.float,
        )
        self.not_done_masks = torch.zeros(
            self.conf.habitat_baselines.num_environments,
            1,
            device=self.device,
            dtype=torch.bool,
        )

        self.policy.reset()
        self.frames = []

    def go(self):
        # Reward missing
        self.reset_internals()
        done = False
        while not done:
            _, actions, _, self.recurrent_hidden_states = self.policy.act(
                self.batch,
                self.recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
            )
            self.prev_actions.copy_(actions)

            actions = self.__parse_actions(actions, self.action_space)

            obs = self.env.step(actions)
            self.frames.append(obs["robot_third_rgb"])

            self.batch = self.__parse_obs(obs)
            done = actions["action_args"]["rearrange_stop"][0] > 0.99

            self.not_done_masks = torch.ones(
                self.conf.habitat_baselines.num_environments,
                1,
                device=self.device,
                dtype=torch.bool,
            )

        self.__make_video()

    def __get_internal_obs_space(self):
        inner_observation_space = {}
        for key, value in self.observation_space.items():
            if key in self.mappings:
                inner_observation_space[self.mappings[key]] = value
            else:
                inner_observation_space[key] = value
        self.internal_observation_space = gym.spaces.Dict(inner_observation_space)

    def __parse_obs(self, obs):
        new_obs = []
        for key in sorted(obs.keys()):
            if key in self.mappings:
                new_obs.append((self.mappings[key], obs[key]))
            else:
                new_obs.append((key, obs[key]))
        new_obs = [OrderedDict(new_obs)]
        batch = batch_obs(new_obs, device=self.device)
        return apply_obs_transforms_batch(batch, self.obs_transforms)

    def __parse_actions(self, action, action_space):
        cur = 0
        act = {"action": [], "action_args": {}}
        for action_name in action_space:
            act["action"].append(action_name)
            if action_name == "rearrange_stop":
                act["action_args"]["rearrange_stop"] = (
                    action[0][cur: cur + 1].cpu().detach().numpy()
                )
                cur += 1
            else:
                for arg in action_space[action_name]:
                    length = action_space[action_name][arg]._shape[0]
                    act["action_args"][arg] = (
                        action[0][cur: cur + length].cpu().detach().numpy()
                    )
                    act["action_args"][arg] = np.clip(
                        act["action_args"][arg],
                        action_space[action_name][arg].low,
                        action_space[action_name][arg].high,
                    )
                    cur += length

        act["action"] = tuple(act["action"])
        return act

    def __make_video(self):
        writer = imageio.get_writer(
            os.path.join("videos/", "test.mp4"),
            fps=30,
            quality=4,
        )
        for im in self.frames:
            writer.append_data(im)
        writer.close()
