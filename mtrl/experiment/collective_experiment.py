# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""`Experiment` class manages the lifecycle of a model."""

import json
import os
from copy import deepcopy
from typing import Any, List, Optional, Tuple
import re
import random

import hydra
import torch
from torch.utils.data import DataLoader
import numpy as np

from mtrl.env.types import EnvType
from mtrl.logger import Logger
from mtrl.utils import checkpointable
from mtrl.utils import config as config_utils
from mtrl.utils import utils, video
from mtrl.utils.types import ConfigType, EnvMetaDataType, EnvsDictType

import importlib
import copy


class Experiment(checkpointable.Checkpointable):
    def __init__(self, config: ConfigType, experiment_id: str = "0"):
        """Experiment Class to manage the lifecycle of a model.

        Args:
            config (ConfigType):
            experiment_id (str, optional): Defaults to "0".
        """
        self.id = experiment_id
        self.config = config
        self.device = torch.device(self.config.setup.device)

        self.get_env_metadata = get_env_metadata
        self.envs, self.env_metadata = self.build_envs() 
        self.env_counter = 0

        with open(config.setup.metadata_path) as f:
            self.metadata = json.load(f)

        key = "ordered_task_list"
        if key in self.env_metadata and self.env_metadata[key]:
            ordered_task_dict = {
                task: index for index, task in enumerate(self.env_metadata[key])
            }
        else:
            ordered_task_dict = {}
            raise NotImplemented

        key = "envs_to_exclude_during_training"
        if key in self.config.experiment and self.config.experiment[key]:
            self.envs_to_exclude_during_training = {
                ordered_task_dict[task] for task in self.config.experiment[key]
            }
            print(
                f"Excluding the following environments: {self.envs_to_exclude_during_training}"
            )
            self.env_indices = [id for task, id in ordered_task_dict.items() if task not in self.config.experiment[key]]
        else:
            self.envs_to_exclude_during_training = set()
            self.env_indices = [id for task, id in ordered_task_dict.items()]
        
        self.env_indices_i = list(range(len(self.env_indices)))

        self.action_space = self.env_metadata["action_space"]
        assert self.action_space.low.min() >= -1
        assert self.action_space.high.max() <= 1

        self.env_obs_space = self.env_metadata["env_obs_space"]

        env_obs_shape = self.env_obs_space.shape
        action_shape = self.action_space.shape
        self.env_obs_shape = self.env_obs_space.shape[0]
        self.action_shape = self.action_space.shape[0]

        self.task_name_to_idx = ordered_task_dict

        self.config = prepare_config(config=self.config, env_metadata=self.env_metadata)
        should_resume_experiment = self.config.experiment.should_resume

        self.seq_len = self.config.transformer_collective_network.transformer_encoder.representation_transformer.sequence_len
        self.cls_token = self.config.transformer_collective_network.transformer_encoder.transformer.with_cls
        
        metadata_keys = list(self.metadata.keys())
        self.task_names = [x for x in ordered_task_dict.keys() if x not in self.config.experiment.envs_to_exclude_during_training]
        self.policy_import_names = self.task_names
        self.num_envs = len(self.env_indices)
        if "MT5" in config.env.benchmark._target_:
            self.task_embedding = [self.metadata[name[:-2]] for name in self.task_names]
            self.task_num = np.array([metadata_keys.index(name[:-2]) for name in self.task_names])
            self.policy_import_names = [name[:-2] for name in self.task_names]
        else:
            self.task_embedding = [self.metadata[name] for name in self.task_names]
            self.task_num = np.array([metadata_keys.index(name) for name in self.task_names])

        self.scripted_policies = [self.import_policy_class(name[:-3]) for name in self.policy_import_names]

        if self.config.experiment.mode == 'train_worker':
            # Worker
            self.agent = [hydra.utils.instantiate( 
                self.config.worker.builder,
                env_obs_shape=env_obs_shape,
                action_shape=action_shape,
                action_range=[
                    float(self.action_space.low.min()),
                    float(self.action_space.high.max()),
                ],
                device=self.device,
            ) for _ in range(self.num_envs)]

            self.model_dir = [utils.make_dir(
                os.path.join(self.config.setup.save_dir, f"model_dir/model_{i}_seed_{self.config.setup.seed}")
            ) for i in self.task_names]
            self.buffer_dir = [utils.make_dir(
                os.path.join(self.config.setup.save_dir, f"buffer/buffer/buffer_{i}_seed_{self.config.setup.seed}")
            ) for i in self.task_names]
            self.buffer_dir_distill_tmp = [utils.make_dir(
                os.path.join(self.config.setup.save_dir, f"buffer/buffer_distill_tmp/buffer_distill_tmp_{i}_seed_{self.config.setup.seed}")
            ) for i in self.task_names]
            self.buffer_dir_distill = [utils.make_dir(
                os.path.join(self.config.setup.save_dir, f"buffer/buffer_distill/buffer_distill_{i}_seed_{self.config.setup.seed}")
            ) for i in self.task_names]

            self.replay_buffer = [hydra.utils.instantiate(
                self.config.replay_buffer.replay_buffer,
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
            ) for _ in range(self.num_envs)]

            self.replay_buffer_distill_tmp = [hydra.utils.instantiate(
                self.config.replay_buffer.col_tmp_replay_buffer,
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
            ) for _ in range(self.num_envs)]

            self.replay_buffer_distill = [hydra.utils.instantiate(
                self.config.replay_buffer.col_replay_buffer,
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
            ) for _ in range(self.num_envs)]

            assert self.config.experiment.col_sampling_freq==0 or self.config.replay_buffer.col_tmp_replay_buffer.capacity >= self.config.experiment.num_train_steps / self.config.experiment.col_sampling_freq * self.config.experiment.col_training_samples
            assert self.config.replay_buffer.col_replay_buffer.capacity >= self.config.replay_buffer.col_tmp_replay_buffer.capacity

            self.start_step = 0

            if should_resume_experiment:
                for i in range(self.num_envs):
                    self.start_step = self.agent[i].load_latest_step(model_dir=self.model_dir[i])
                    self.replay_buffer[i].load(save_dir=self.buffer_dir[i])
                    self.replay_buffer_distill_tmp[i].load(save_dir=self.buffer_dir_distill_tmp[i])

        elif self.config.experiment.mode == 'record':
            self.buffer_dir_distill = [utils.make_dir(
                os.path.join(self.config.setup.save_dir, f"buffer/buffer_distill/recording_buffer_distill_{i}_seed_{self.config.setup.seed}")
            ) for i in self.task_names]

            self.replay_buffer_distill = [hydra.utils.instantiate(
                self.config.replay_buffer.col_replay_buffer,
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
            ) for _ in range(self.num_envs)]

            self.start_step = 0
        elif self.config.experiment.mode == 'distill_collective_transformer':
            self.col_buffer_loc = utils.make_dir(
                os.path.join(self.config.setup.save_dir, "buffer/collective_buffer/train")
            )
            self.distill_buffer_names = os.listdir(self.col_buffer_loc)
            self.distill_names_with_seed = [name.replace("buffer_distill_", "") for name in self.distill_buffer_names]
            pattern = r'(-\d+)?_seed_\d+$'
            self.distill_names = [re.sub(pattern, '', name) for name in self.distill_names_with_seed]
            self.num_datasets = len(self.distill_names)

            # collective network
            self.col_agent = hydra.utils.instantiate(
                self.config.transformer_collective_network.builder,
                env_obs_shape=env_obs_shape,
                action_shape=action_shape,
                action_range=[
                    float(self.action_space.low.min()),
                    float(self.action_space.high.max()),
                ],
                device=self.device,
            )
                
            self.col_model_dir = utils.make_dir(
                os.path.join(self.config.setup.save_dir, "model_dir/model_col")
            )
            self.buffer_dir_distill = [utils.make_dir(
                os.path.join(self.config.setup.save_dir, f"buffer/collective_buffer/train/{i}")
            ) for i in self.distill_buffer_names]
            self.buffer_dir_val = [utils.make_dir(
                os.path.join(self.config.setup.save_dir, f"buffer/collective_buffer/validation/{i}")
            ) for i in self.distill_buffer_names]

            replay_buffer = hydra.utils.instantiate(
                self.config.replay_buffer.transformer_col_replay_buffer,
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
                seq_len=self.seq_len
            )

            replay_buffer.load_multiple_buffer(self.buffer_dir_distill)
            self.replay_buffer_distill = replay_buffer

            self.replay_buffer_val = hydra.utils.instantiate(
                self.config.replay_buffer.transformer_col_replay_buffer,
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
                seq_len=self.seq_len
            )
            
            self.replay_buffer_val.load_multiple_buffer(self.buffer_dir_val)
            
            self.col_start_step = 0

            if should_resume_experiment:
                self.col_start_step = self.col_agent.load_latest_step(model_dir=self.col_model_dir)
        elif self.config.experiment.mode == 'evaluate_collective_transformer':
            # collective network
            if self.config.experiment.evaluate_transformer == "collective_network":
                self.col_agent = hydra.utils.instantiate(
                    self.config.transformer_collective_network.builder,
                    env_obs_shape=env_obs_shape,
                    action_shape=action_shape,
                    action_range=[
                        float(self.action_space.low.min()),
                        float(self.action_space.high.max()),
                    ],
                    device=self.device,
                )

                self.col_model_dir = utils.make_dir(
                    os.path.join(self.config.setup.save_dir, "model_dir/model_col")
                )

                self.col_start_step = self.col_agent.load_latest_step(model_dir=self.col_model_dir)
            else:
                self.col_agent = hydra.utils.instantiate( 
                    self.config.worker.builder,
                    env_obs_shape=env_obs_shape,
                    action_shape=action_shape,
                    action_range=[
                        float(self.action_space.low.min()),
                        float(self.action_space.high.max()),
                    ],
                    device=self.device,
                )

                self.col_model_dir = utils.make_dir(
                    os.path.join(self.config.setup.save_dir, "evaluation_models")
                )

                self.col_start_step = self.col_agent.load_latest_step(model_dir=self.col_model_dir)

                self.agent = [self.col_agent]

            self.embedding_save_path = utils.make_dir(self.config.transformer_collective_network.embedding_save_path)
            self.embedding_save_path = self.config.transformer_collective_network.embedding_save_path + 'emb.pth'

            self.col_buffer_loc = utils.make_dir(
                os.path.join(self.config.setup.save_dir, "buffer/collective_buffer/validation")
            )
            self.distill_buffer_names = os.listdir(self.col_buffer_loc)
            self.distill_names = [name.replace("buffer_distill_", "") for name in self.distill_buffer_names]
            pattern = r'(-\d+)?_seed_\d+$'
            self.distill_names = [re.sub(pattern, '', name) for name in self.distill_names]
            self.num_datasets = len(self.distill_names)

            self.buffer_dir_distill = [utils.make_dir(
                os.path.join(self.config.setup.save_dir, f"buffer/collective_buffer/validation/{i}")
            ) for i in self.distill_buffer_names]

            self.replay_buffer_distill = [hydra.utils.instantiate(
                self.config.replay_buffer.transformer_col_replay_buffer,
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
                seq_len=self.seq_len,
            ) for _ in range(self.num_datasets)]
            
            for i in range(self.num_datasets):
                self.replay_buffer_distill[i].load(save_dir=self.buffer_dir_distill[i], seq_len=self.config.transformer_collective_network.transformer_encoder.representation_transformer.sequence_len)

        elif self.config.experiment.mode == 'online_distill_collective_transformer':
            # collective network
            self.col_agent = hydra.utils.instantiate(
                self.config.transformer_collective_network.builder,
                env_obs_shape=env_obs_shape,
                action_shape=action_shape,
                action_range=[
                    float(self.action_space.low.min()),
                    float(self.action_space.high.max()),
                ],
                device=self.device,
            )

            self.expert_model_dir = [utils.make_dir(
                os.path.join(self.config.setup.save_dir, f"model_dir/model_{i}_seed_{self.config.setup.seed}")  
            ) for i in self.task_names]
            
            self.buffer_dir = utils.make_dir(
                os.path.join(self.config.setup.save_dir, f"buffer/online_buffer_{self.task_names[0]}")
            )

            # Expert
            self.expert = [hydra.utils.instantiate( 
                self.config.worker.builder,
                env_obs_shape=env_obs_shape,
                action_shape=action_shape,
                action_range=[
                    float(self.action_space.low.min()),
                    float(self.action_space.high.max()),
                ],
                device=self.device,
            ) for _ in self.env_indices_i]
            
            self.replay_buffer = hydra.utils.instantiate(
                self.config.replay_buffer.transformer_col_replay_buffer,
                normalize_rewards=False,
                batch_size=self.config.replay_buffer['batch_size'],
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
                seq_len=self.seq_len
            )

            self.start_step = 0
            self.replay_buffer.load(save_dir=self.buffer_dir)
            for i in self.env_indices_i:
                self.expert[i].load_latest_step(model_dir=self.expert_model_dir[i])
        elif self.config.experiment.mode == 'train_student':
            # Student
            self.student = hydra.utils.instantiate(
                self.config.student.builder,
                env_obs_shape=env_obs_shape,
                action_shape=action_shape,
                action_range=[
                    float(self.action_space.low.min()),
                    float(self.action_space.high.max()),
                ],
                device=self.device,
            )

            # collective network
            self.col_agent = hydra.utils.instantiate(
                self.config.transformer_collective_network.builder,
                env_obs_shape=env_obs_shape,
                action_shape=action_shape,
                action_range=[
                    float(self.action_space.low.min()),
                    float(self.action_space.high.max()),
                ],
                device=self.device,
            )

            self.col_model_dir = utils.make_dir(
                os.path.join(self.config.setup.save_dir, "model_dir/model_col")
            )
            self.col_start_step = self.col_agent.load_latest_step(model_dir=self.col_model_dir)

            self.student_model_dir = utils.make_dir(
                os.path.join(self.config.setup.save_dir, f'model_dir/student_model_{self.task_names[0]}_seed_{self.config.setup.seed}')  
            )
            self.buffer_dir = utils.make_dir(
                os.path.join(self.config.setup.save_dir, f"buffer/student_buffer_{self.task_names[0]}_seed_{self.config.setup.seed}")
            )

            self.replay_buffer = hydra.utils.instantiate(
                self.config.replay_buffer.transformer_col_replay_buffer,
                normalize_rewards=False,
                capacity=self.config.replay_buffer.replay_buffer['capacity'],
                batch_size=self.config.replay_buffer.replay_buffer['batch_size'],
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
                seq_len=self.seq_len,
                compressed_state= False
            )
            
            self.start_step = 0
            if should_resume_experiment:
                self.start_step = self.student.load_latest_step(model_dir=self.student_model_dir)
                self.replay_buffer.load(save_dir=self.buffer_dir)

        elif self.config.experiment.mode == 'train_student_finetuning':
            # finetuning
            # Student
            self.student = hydra.utils.instantiate(
                self.config.transformer_collective_network.builder,
                env_obs_shape=env_obs_shape,
                action_shape=action_shape,
                action_range=[
                    float(self.action_space.low.min()),
                    float(self.action_space.high.max()),
                ],
                device=self.device,
                dropout=0.,
            )

            self.student_model_dir = utils.make_dir(
                os.path.join(self.config.setup.save_dir, f"model_dir/student_model_finetune")  
            )
            col_model_dir = utils.make_dir(
                os.path.join(self.config.setup.save_dir, "model_dir/model_col")
            )
            self.buffer_dir = utils.make_dir(
                os.path.join(self.config.setup.save_dir, f"buffer/student_buffer")
            )

            self.replay_buffer = hydra.utils.instantiate(
                self.config.replay_buffer.transformer_col_replay_buffer,
                normalize_rewards=False,
                batch_size=self.config.replay_buffer.replay_buffer['batch_size'],
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
                seq_len=self.seq_len
            )

            self.start_step = 0
            self.col_start_step = 0

            if self.start_step == 0 or not should_resume_experiment:
                self.student.load_latest_step(model_dir=col_model_dir)
        
        elif self.config.experiment.mode == 'distill_policy':
            self.col_agent = hydra.utils.instantiate(
                self.config.transformer_collective_network.builder,
                env_obs_shape=env_obs_shape,
                action_shape=action_shape,
                action_range=[
                    float(self.action_space.low.min()),
                    float(self.action_space.high.max()),
                ],
                device=self.device,
            )

            self.col_model_dir = utils.make_dir(
                os.path.join(self.config.setup.save_dir, "model_dir/model_col")
            )

            self.col_start_step = self.col_agent.load_latest_step(model_dir=self.col_model_dir)
            


            self.student_agent = hydra.utils.instantiate(
                self.config.student.builder,
                env_obs_shape=env_obs_shape,
                action_shape=action_shape,
                action_range=[
                    float(self.action_space.low.min()),
                    float(self.action_space.high.max()),
                ],
                device=self.device,
            )


            # Student model dir
            self.student_model_dir = utils.make_dir(
                os.path.join(self.config.setup.save_dir, "model_dir/student_model")
            )

            ## Replay buffer used for distillation
            #self.buffer_dir = utils.make_dir(
            #    os.path.join(self.config.setup.save_dir, f"buffer/policy_distill_{self.task_names[0]}_seed_{self.config.setup.seed}")
            #)

            self.replay_buffer = hydra.utils.instantiate(
                self.config.replay_buffer.transformer_col_replay_buffer,
                normalize_rewards=False,
                capacity=self.config.replay_buffer.replay_buffer['capacity'],
                batch_size=self.config.replay_buffer.replay_buffer['batch_size'],
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
                seq_len=self.seq_len,
                compressed_state= False
            )
            #self.replay_buffer.load(save_dir=self.buffer_dir)

            #self.start_step = 0

        else:
            raise NotImplementedError(
                f"experiment-mode {self.config.experiment.mode} is not supported"
            )

        self.logger = Logger(
            self.config.setup.save_dir,
            config=self.config,
            retain_logs=should_resume_experiment,
        )
        self.max_episode_steps = self.env_metadata[
            "max_episode_steps"
        ]  # maximum steps that the agent can take in one environment.

        self.video_dir = utils.make_dir(
            os.path.join(self.config.setup.save_dir, f"video")
        )
        self.video = video.VideoRecorder(
            self.video_dir if self.config.experiment.save_video else None
        ) 

        self.startup_logs()

    def import_policy_class(self, policy_name):
            if policy_name == "peg-insert-side":
                module_name, class_name = "peg_insertion_side", "PegInsertionSide"
            else:
                module_name, class_name = policy_name.replace("-", "_"), "".join([name.capitalize() for name in policy_name.split("-")])
            module_name = f"metaworld.policies.sawyer_{module_name}_v2_policy"
            policy_class = getattr(importlib.import_module(module_name), f"Sawyer{class_name}V2Policy") 
            return policy_class
    
    def build_envs(self) -> Tuple[EnvsDictType, EnvMetaDataType]:
        """Subclasses should implement this method to build the environments.

        Raises:
            NotImplementedError: this method should be implemented by the subclasses.

        Returns:
            Tuple[EnvsDictType, EnvMetaDataType]: Tuple of environment dictionary
            and environment metadata.
        """
        raise NotImplementedError(
            "`build_envs` is not defined for experiment.Experiment"
        )

    def startup_logs(self) -> None:
        """Write some logs at the start of the experiment."""
        config_file = f"{self.config.setup.save_dir}/config.json"
        with open(config_file, "w") as f:
            f.write(json.dumps(config_utils.to_dict(self.config)))

    def periodic_save(self, epoch: int) -> None:
        """Perioridically save the experiment.

        This is a utility method, built on top of the `save` method.
        It performs an extra check of wether the experiment is configured to
        be saved during the current epoch.
        Args:
            epoch (int): current epoch.
        """
        persist_frequency = self.config.experiment.persist_frequency
        if persist_frequency > 0 and epoch % persist_frequency == 0:
            self.save(epoch)

    def save(self, epoch: int) -> Any:  # type: ignore[override]
        raise NotImplementedError(
            "This method should be implemented by the subclasses."
        )

    def load(self, epoch: Optional[int]) -> Any:  # type: ignore[override]
        raise NotImplementedError(
            "This method should be implemented by the subclasses."
        )

    def run(self) -> None:
        """Run the experiment.

        Raises:
            NotImplementedError: This method should be implemented by the subclasses.
        """
        raise NotImplementedError(
            "This method should be implemented by the subclasses."
        )

    def close_envs(self):
        """Close all the environments."""
        for env in self.envs.values():
            env.close()
    
    def reset_goal_locations(self, env="train"):
        #new_tasks = [random.choice(self.env_name_task_dict[env]) for env in self.env_metadata["ordered_task_list"]]
        new_tasks = {key: random.choice(value) for key, value in self.env_name_task_dict.items()}
        self.envs[env].call("set_task_with_dict", new_tasks)

    def reset_goal_locations_in_order(self, env="train"):
        #new_tasks = [random.choice(self.env_name_task_dict[env]) for env in self.env_metadata["ordered_task_list"]]
        new_tasks = {key: value[self.env_counter] for key, value in self.env_name_task_dict.items()}
        self.env_counter = (self.env_counter+1) % 50
        self.envs[env].call("set_task_with_dict", new_tasks)

    def reset_goal_locations_for_listenv(self, index, env_name):
        new_tasks = {env_name: random.choice(self.env_name_task_dict[env_name])}
        self.list_envs[index].env.set_task_with_dict(new_tasks)

def prepare_config(config: ConfigType, env_metadata: EnvMetaDataType) -> ConfigType:
    """Infer some config attributes during runtime.

    Args:
        config (ConfigType): config to update.
        env_metadata (EnvMetaDataType): metadata of the environment.

    Returns:
        ConfigType: updated config.
    """
    config = config_utils.make_config_mutable(config_utils.unset_struct(config))
    key = "type_to_select"
    if key in config.worker.encoder:
        encoder_type_to_select = config.worker.encoder[key]
    else:
        encoder_type_to_select = config.worker.encoder.type
    if encoder_type_to_select in ["identity"]:
        # if the encoder is an identity encoder infer the shape of the input dim.
        config.worker.encoder_feature_dim = env_metadata["env_obs_space"].shape[0]

    if key in config.student.encoder:
        encoder_type_to_select = config.student.encoder[key]
    else:
        encoder_type_to_select = config.student.encoder.type
    if encoder_type_to_select in ["identity"]:
        # if the encoder is an identity encoder infer the shape of the input dim.
        config.student.encoder_feature_dim = env_metadata["env_obs_space"].shape[0]

    key = "ordered_task_list"
    if key in env_metadata and env_metadata[key]:
        config.env.ordered_task_list = deepcopy(env_metadata[key])
    config = config_utils.make_config_immutable(config)

    return config


def get_env_metadata(
    env: EnvType,
    max_episode_steps: Optional[int] = None,
    ordered_task_list: Optional[List[str]] = None,
) -> EnvMetaDataType:
    """Method to get the metadata from an environment"""
    dummy_env = env.env_fns[0]().env
    metadata: EnvMetaDataType = {
        "env_obs_space": dummy_env.observation_space,
        "action_space": dummy_env.action_space,
        "ordered_task_list": ordered_task_list,
    }
    if max_episode_steps is None:
        metadata["max_episode_steps"] = dummy_env._max_episode_steps
    else:
        metadata["max_episode_steps"] = max_episode_steps
    return metadata
