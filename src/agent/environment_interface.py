import os
import imageio
from collections import OrderedDict

import torch
import gym
import numpy as np

# HABITAT
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    get_active_obs_transforms,
)
from habitat_baselines.utils.common import get_num_actions

# LOCAL
from scene.scene_parser import SceneParser
from sensors import SENSOR_MAPPINGS


class EnvironmentInterface:
    def __init__(self, env, conf, device):
        self.env = env
        self.conf = conf
        self.mappings = SENSOR_MAPPINGS
        self.device = device

        self.sim = env._sim
        self.scene_parser = SceneParser(self.sim)
        self.ppo_cfg = conf.habitat_baselines.rl.ppo

        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.__get_internal_obs_space()
        self.obs_transforms = get_active_obs_transforms(conf)

        self.frames = []
        self.reset_environment()

    def set_target(self, target):
        self.scene_parser.set_dynamic_target(target)

    def reset_environment(self):
        obs = self.env.reset()
        self.batch = self.__parse_observations(obs)

        self.sim = self.env._sim
        self.scene_parser = SceneParser(self.sim)
        self.reset_internals()
        self.video_name = "test.mp4"

        if self.frames != []:
            self.__make_video()
            
        self.frames = []

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

    def execute_low_level_skill(self, skill, deterministic=False):
        # Reward missing

        self.reset_internals()
        skill.on_enter([0], self.batch)
        finished, failed = False, False

        while not finished and not failed:

            actions, self.recurrent_hidden_states = skill.act(
                self.batch,
                self.recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                cur_batch_idx=[0],
                deterministic=deterministic,
            )
            self.prev_actions.copy_(actions)
            actions = self.__parse_actions(actions, self.action_space)

            obs = self.env.step(actions)
            self.frames.append(obs["robot_third_rgb"])
            self.batch = self.__parse_observations(obs)

            finished, failed = skill.should_terminate(
                self.batch,
                self.recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                batch_idx=[0]
            )

            self.not_done_masks = torch.ones(
                self.conf.habitat_baselines.num_environments,
                1,
                device=self.device,
                dtype=torch.bool,
            )
        
        self.__make_video()
        
        return finished, failed

    def __get_internal_obs_space(self):
        inner_observation_space = {}
        for key, value in self.observation_space.items():
            if key in self.mappings:
                inner_observation_space[self.mappings[key]] = value
            else:
                inner_observation_space[key] = value
        self.internal_observation_space = gym.spaces.Dict(inner_observation_space)

    def __parse_observations(self, obs):
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
        os.makedirs("videos", exist_ok=True)
        
        writer = imageio.get_writer(
            os.path.join("videos/", self.video_name),
            fps=30,
            quality=4,
        )
        for im in self.frames:
            writer.append_data(im)
        writer.close()
