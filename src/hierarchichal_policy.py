# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym.spaces as spaces

from habitat.core.spaces import ActionSpace
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.logging import baselines_logger
from high_level import (  # noqa: F401.
    FixedHighLevelPolicy,
    HighLevelPolicy,
)
from skills import (  # noqa: F401.
    NavSkillPolicy,
    PickSkillPolicy,
    PlaceSkillPolicy,
    ResetArmSkill
)
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import Policy


class State:
    def __init__(self):
        self.observations = None
        self.prev_rnn_hidden_states = None
        self.prev_actions = None
        self.masks = None
        self.actions = None
        self.new_hidden_states = None
        self.bad_termination = False

    def set_state(self, observations, rnn_hidden_states, prev_actions, masks):
        self.observations = observations
        self.prev_rnn_hidden_states = rnn_hidden_states
        self.prev_actions = prev_actions
        self.masks = masks

    def set_actions(self, actions):
        self.actions = actions

    def set_new_hidden_states(self, new_hidden_states):
        self.new_hidden_states = new_hidden_states

    def set_bad_termination(self, terminate):
        self.bad_termination = terminate

    def get_state_dict(self):
        return {
            "observations": self.observations,
            "rnn_hidden_states": self.prev_rnn_hidden_states,
            "prev_actions": self.prev_actions,
            "masks": self.masks,
        }

    def reset_prev_state_information(self):
        self.prev_actions *= 0
        self.prev_rnn_hidden_states *= 0

    def is_new_episode(self):
        return bool(self.masks[0][0]) is False

# class SkillState():
#     def __init__(self, env):
#         return


@baseline_registry.register_policy
class NewHierarchicalPolicy(Policy):
    def __init__(
        self,
        config,
        full_config,
        observation_space: spaces.Space,
        action_space: ActionSpace,
        num_envs: int,
    ):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        self.full_config = full_config
        self._num_envs = num_envs

        # Initialize the available skills for the high level planner
        self.available_skills = {}
        self.initialize_skills()

        # The skill history is a list that contains the skills used for a given
        # environment and episode.
        self.skill_history = []

        # Initialize tha states for each environment
        self.states = State() 

        # high-level Policy
        high_level_cls = eval(
            config.hierarchical_policy.high_level_policy.name
        )
        self.high_level = high_level_cls()

        # Stop action index of the current action space
        # NOTE: (is it the same for low-level skills?)
        self._stop_action_idx, _ = find_action_range(
            action_space, "rearrange_stop"
        ) 

    def get_current_skill(self):
        """
        Returns the high-level skill that is currently acting in a given environment/
        """
        return self.skill_history[-1]

    def reset(self):
        self.skill_history = []
        self.high_level.reset()
    
    def set_current_skill(self, skill):
        self.skill_history.append(skill)

    def should_terminate(self, should_change_skill):
        '''
        Currently the following conditions terminate the episode:
            - Bad skill termination (usually timeout of the skill)
            - Finished plan ((we performed all the actions)
        '''
        state = self.states

        bad_termination = state.bad_termination

        finished_plan = self.high_level.is_in_last_action and should_change_skill

        return any([bad_termination, finished_plan])

    def should_change_skill(self):
        """
        The following conditions determine whether the high-level skill should change:
            - The current skill called the done action.
            - It's a new episode
            - The high-level planner decided to change the skill.
        """

        # If the episode was over, we should change the skill to the first one of the 
        # new episode.
        state = self.states
        if state.is_new_episode():
            self.reset()
            state.set_bad_termination(False)
            return True
            
        # Skill termination
        current_skill = self.get_current_skill()
        want_to_terminate, bad_termination = current_skill.should_terminate(
            **state.get_state_dict(), batch_idx=[0]
        )
        state.set_bad_termination(bad_termination)

        # High-Level decision
        # skill_history = self.skill_history[env]
        # high_planner_decision = self._high_level_policy.should_terminate(env, skill_history)
        high_level_decision = False

        if any([want_to_terminate, high_level_decision]):
            return True
        return False

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        not_done_masks,
        deterministic=False,
    ):  
        # Initialize state with new information and update skills if necessary
        # Set the state information
        state = self.states
        state.set_state(
            observations, rnn_hidden_states, prev_actions, not_done_masks
        )

        should_change_skill = self.should_change_skill()

        # We need to terminate the episode
        if self.should_terminate(should_change_skill):
            env_actions = prev_actions * 0
            env_actions[:, self._stop_action_idx] = 1
            return (None, env_actions, None, rnn_hidden_states)

        # Change the skill
        if should_change_skill:
            # Get the next skill
            next_skill = self.high_level.get_next_skill()
            next_skill = self.available_skills[next_skill]

            # Initialize skill
            state_dict = state.get_state_dict()
            state_dict.pop("masks")
            next_skill.on_enter([0], **state_dict)
            self.set_current_skill(next_skill)
            
            # Since we changed the skill, we need to reset the previous 
            # state information
            state.reset_prev_state_information()

        # Perform the actions

        # Get the current skill
        env_skill = self.get_current_skill()
        state, state_dict = self.states, self.states.get_state_dict()

        # Act
        env_actions, env_hidden_states = env_skill.act(
            **state_dict, cur_batch_idx=[0], deterministic=deterministic
        )

        # Update the state
        state.set_actions(env_actions)
        state.set_new_hidden_states(env_hidden_states)

        # Response
        actions = state.actions
        hidden_states = state.new_hidden_states
        return (None, actions, None, hidden_states)

    def initialize_skills(self):
        """
        Initialize a dictionary that associates the name of the skill (nav, pick,
        place, etc.) with the actual skill policy (NavSkillPolicy, ...)
        """
        for skill_name, skill_id in self.config.hierarchical_policy.use_skills.items():
            skill_config = self.config.hierarchical_policy.defined_skills[skill_id]
            cls = eval(skill_config.skill_name)
            baselines_logger.info("Initializing skill: {}".format(skill_name))
            skill_policy = cls.from_config(
                skill_config,
                self.observation_space,
                self.action_space,
                self._num_envs,
                self.full_config
            )
            self.available_skills[skill_name] = skill_policy

    def eval(self):
        pass

    @property
    def num_recurrent_layers(self):
        # Should be something else
        return list(self.available_skills.values())[0].num_recurrent_layers

    @property
    def should_load_agent_state(self):
        return False

    def parameters(self):
        # Should be the parameters of all the nn_skills
        return list(self.available_skills.values())[0].parameters()

    def to(self, device):
        for skill in self.available_skills.values():
            skill.to(device)

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        orig_action_space,
        **kwargs,
    ):
        return cls(
            config.habitat_baselines.rl.policy,
            config,
            observation_space,
            orig_action_space,
            config.habitat_baselines.num_environments,
        )
