#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from datetime import datetime
import os
import subprocess
import pathlib

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
import numpy as np
import torch
from omegaconf import open_dict

from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)
from habitat.config.default_structured_configs import HabitatConfigPlugin
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.ddp_utils import rank0_only


def get_random_seed():
    seed = (
        os.getpid()
        + int(datetime.now().strftime("%S%f"))
        + int.from_bytes(os.urandom(2), "big")
    )
    print("Using a generated random seed {}".format(seed))
    return seed

def setup():
    """
        Setups the random seed and the wandb logger.
    """
    # Habitat environment variables
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"

    # Register habitat hydra plugins
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    register_hydra_plugin(HabitatConfigPlugin)

    with initialize(config_path="conf"):
        config = compose(config_name="config")

    # Set random seed
    seed = get_random_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    with open_dict(config):
        # Add the seed to the config
        config.habitat.seed = seed

        # Single-agent setup
        config.habitat.simulator.agents_order = list(config.habitat.simulator.agents.keys())

        # Add the wandb information to the habitat config
        config.habitat_baselines.wb.project_name = config.WANDB.project
        config.habitat_baselines.wb.run_name = config.WANDB.name
        config.habitat_baselines.wb.group = config.WANDB.group
        config.habitat_baselines.wb.entity = config.WANDB.entity

    # Create the symlink to the data folder
    subprocess.call(
        [
            "ln",
            "-s",
            "/Users/sergioarnaud/Documents/habitat-llm/data",
            "data",
        ]
    )
    return config

def main() -> None:
    config = setup()
    # Get the trainer
    #trainer_init = baseline_registry.get_trainer(config.habitat_baselines.trainer_name)

if __name__ == "__main__":

    # Call hydra main
    main()