# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import gym.spaces as spaces
import numpy as np
import torch

# Habitat
from habitat.core.spaces import ActionSpace
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.config.default import get_config
from habitat_baselines.utils.common import get_num_actions

# Local
from agent.motor_skills.skill import SkillPolicy



class NnSkillPolicy(SkillPolicy):
    """
    Defines a skill to be used in the TP+SRL baseline.
    """

    def __init__(
        self,
        actor_critic,
        config,
        action_space: spaces.Space,
        filtered_obs_space: spaces.Space,
        filtered_action_space: spaces.Space,
        batch_size,
        should_keep_hold_state: bool = False,
    ):
        """1
        :param action_space: The overall action space of the entire task, not task specific.
        """
        super().__init__(
            config, action_space, batch_size, should_keep_hold_state
        )
        self.actor_critic = actor_critic
        self._filtered_obs_space = filtered_obs_space
        self._filtered_action_space = filtered_action_space
        self._ac_start = 0
        self._ac_len = get_num_actions(filtered_action_space)
        self._did_want_done = torch.zeros(self._batch_size)

        for k, space in action_space.items():
            if k not in filtered_action_space.spaces.keys():
                self._ac_start += get_num_actions(space)
            else:
                break

        self._internal_log(
            f"Skill {self._config.skill_name}: action offset {self._ac_start}, action length {self._ac_len}"
        )

    def parameters(self):
        if self.actor_critic is not None:
            return self.actor_critic.parameters()
        else:
            return []

    @property
    def num_recurrent_layers(self):
        if self.actor_critic is not None:
            return self.actor_critic.net.num_recurrent_layers
        else:
            return 0

    def to(self, device):
        super().to(device)
        self._did_want_done = self._did_want_done.to(device)
        if self.actor_critic is not None:
            self.actor_critic.to(device)

    def on_enter(
        self,
        batch_idxs,
        observations
    ):
        super().on_enter(
            batch_idxs, observations
        )
        self._did_want_done *= 0.0

    def _get_filtered_obs(self, observations, cur_batch_idx) -> TensorDict:
        return TensorDict(
            {
                k: observations[k]
                for k in self._filtered_obs_space.spaces.keys()
            }
        )

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        filtered_obs = self._get_filtered_obs(observations, cur_batch_idx)

        filtered_prev_actions = prev_actions[
            :, self._ac_start : self._ac_start + self._ac_len
        ]

        _, action, _, rnn_hidden_states = self.actor_critic.act(
            filtered_obs,
            rnn_hidden_states,
            filtered_prev_actions,
            masks,
            deterministic,
        )
        full_action = torch.zeros(prev_actions.shape, device=masks.device)
        full_action[:, self._ac_start : self._ac_start + self._ac_len] = action

        self._did_want_done[cur_batch_idx] = full_action[
            cur_batch_idx, self._stop_action_idx
        ]
        return full_action, rnn_hidden_states


    @classmethod
    def from_config(
        cls, config, observation_space, action_space, batch_size, full_config
    ):
        # Load the wrap policy from file
        try:
            ckpt_dict = torch.load(
                config.load_ckpt_file, map_location="cpu"
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Could not load neural network weights for skill."
            ) from e

        policy_cfg = ckpt_dict["config"]
        policy = baseline_registry.get_policy(config.name)

        expected_obs_space = policy_cfg.habitat.gym.obs_keys

        excepted_action_space = policy_cfg.habitat.task.actions.keys()

        filtered_obs_space = spaces.Dict(
            {k: observation_space.spaces[k] for k in expected_obs_space}
        )

        baselines_logger.debug(
            f"Loaded obs space {filtered_obs_space} for skill {config.skill_name}",
        )

        baselines_logger.debug(
            "Expected obs space: " + str(expected_obs_space)
        )

        filtered_action_space = ActionSpace(
            OrderedDict(
                (k, action_space[k])
                for k in excepted_action_space
            )
        )

        if "arm_action" in filtered_action_space.spaces and (
            policy_cfg.habitat.task.actions.arm_action.grip_controller is None
        ):
            filtered_action_space["arm_action"] = spaces.Dict(
                {
                    k: v
                    for k, v in filtered_action_space["arm_action"].items()
                    if k != "grip_action"
                }
            )

        baselines_logger.debug(
            f"Loaded action space {filtered_action_space} for skill {config.skill_name}",
        )
        baselines_logger.debug(
            '=' * 80
        )

        actor_critic = policy.from_config(
            policy_cfg, filtered_obs_space, filtered_action_space
        )

        try:
            actor_critic.load_state_dict(
                {  # type: ignore
                    k[len("actor_critic.") :]: v
                    for k, v in ckpt_dict["state_dict"].items()
                }
            )

        except Exception as e:
            raise ValueError(
                f"Could not load checkpoint for skill {config.skill_name} from {config.load_ckpt_file}"
            ) from e

        #baselines_logger.debug(f'Initializing skill with action space {action_space} and observation space {filtered_obs_space} and filtered action space {filtered_action_space}')
        return cls(
            actor_critic,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
        )
