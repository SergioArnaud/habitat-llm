# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import gym.spaces as spaces
import torch

from habitat.tasks.rearrange.rearrange_sensors import IsHoldingSensor
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.common.logging import baselines_logger


class SkillPolicy(Policy):
    def __init__(
        self,
        config,
        action_space: spaces.Space,
        batch_size,
        should_keep_hold_state: bool = False,
    ):
        self._config = config
        self._batch_size = batch_size
        self.action_space = action_space
        self._should_keep_hold_state = should_keep_hold_state
        self._cur_skill_step = torch.zeros(self._batch_size)
        self.device = self._cur_skill_step.device

        # This is the index of the stop action in the action space
        self._stop_action_idx, _ = find_action_range(action_space, "rearrange_stop")

        # This is the index of the grip action in the action space
        _, self.grip_action_index = find_action_range(action_space, "arm_action") 
        self.grip_action_index -= 1
        baselines_logger.debug("grip_action_index " + str(self.grip_action_index))

    def to(self, device):
        self.device = device
        self._cur_skill_step = self._cur_skill_step.to(device)

    def _keep_holding_state(
        self, full_action: torch.Tensor, observations
    ) -> torch.Tensor:
        """
        Makes the action so it does not result in dropping or picking up an
        object. Used in navigation and other skills which are not supposed to
        interact through the gripper.
        """

        # Keep the same grip state as the previous action.
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)

        # If it is not holding (0) want to keep releasing -> output -1.
        # If it is holding (1) want to keep grasping -> output +1.
        full_action[:, self.grip_action_index] = is_holding + (is_holding - 1.0)
        return full_action

    def _internal_log(self, s, observations=None):
        baselines_logger.debug(
            f"Skill {self._config.skill_name} @ step {self._cur_skill_step}: {s}"
        )

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ) -> torch.BoolTensor:
        """
        :returns: A (batch_size,) size tensor where 1 indicates the skill wants to end and 0 if not.
        """
        return torch.zeros(observations.shape[0], dtype=torch.bool).to(self.device)

    def should_terminate(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
        """
        :returns: A (batch_size,) size tensor where 1 indicates the skill wants to end and 0 if not.
        """
        is_skill_done = self._is_skill_done(
            observations, rnn_hidden_states, prev_actions, masks, batch_idx
        )
        if is_skill_done.sum() > 0:
            self._internal_log(
                f"Requested skill termination {is_skill_done}",
                observations,
            )

        bad_terminate = torch.zeros(
            self._cur_skill_step.shape,
            device=self.device,
            dtype=torch.bool,
        )
        if self._config.max_skill_steps > 0:
            over_max_len = self._cur_skill_step > self._config.max_skill_steps
            if self._config.force_end_on_timeout:
                bad_terminate = over_max_len
            else:
                is_skill_done = is_skill_done | over_max_len

        if bad_terminate.sum() > 0:
            self._internal_log(
                f"Bad terminating due to timeout {self._cur_skill_step}, {bad_terminate}",
                observations,
            )

        return is_skill_done, bad_terminate

    def on_enter(
        self,
        batch_idxs: List[int],
        observations,
        rnn_hidden_states,
        prev_actions,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes in the data at the current `batch_idx`
        :returns: The new hidden state and prev_actions ONLY at the batch_idx.
        """

        self._cur_skill_step[batch_idxs] = 0
        return (
            rnn_hidden_states[batch_idxs] * 0.0,
            prev_actions[batch_idxs] * 0.0,
        )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        """
        :returns: Predicted action and next rnn hidden state.
        """
        self._cur_skill_step[cur_batch_idx] += 1
        action, hxs = self._internal_act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            cur_batch_idx,
            deterministic,
        )

        if self._should_keep_hold_state:
            action = self._keep_holding_state(action, observations)
        return action, hxs

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @classmethod
    def from_config(
        cls, config, observation_space, action_space, batch_size, full_config
    ):
        return cls(config, action_space, batch_size)