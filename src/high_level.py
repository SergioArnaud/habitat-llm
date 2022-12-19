# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Tuple

import torch

from habitat_baselines.common.logging import baselines_logger


class HighLevelPolicy:
    def get_next_skill(self) -> Tuple[torch.Tensor, List[Any], torch.BoolTensor]:
        """
        :returns: A tuple containing the next skill index, a list of arguments
            for the skill, and if the high-level policy requests immediate
            termination.
        """
        raise NotImplementedError()


class FixedHighLevelPolicy(HighLevelPolicy):
    def __init__(self):
        self.skills = ["nav", "pick", "nav", "place"]
        self.skills = self.wrap_with_reset()
        self.cur = 0

    def wrap_with_reset(self):
        skills = []
        for skill in self.skills:
            skills.append("reset_arm")
            skills.append(skill)
        return skills

    def reset(self):
        self.cur = 0

    def get_next_skill(self):
        if self.is_in_last_action:
            raise Exception("No more skills to execute, should call is_in_last_action")

        baselines_logger.debug(f"Entering skill {self.skills[self.cur]}")

        to_ret = self.skills[self.cur]
        self.cur += 1
        return to_ret

    @property
    def is_in_last_action(self):
        return self.cur == len(self.skills)
