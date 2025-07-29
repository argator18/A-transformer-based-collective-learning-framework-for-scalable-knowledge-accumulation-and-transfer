# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""`Experiment` class manages the lifecycle of a multi-task model."""
import random

import time
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch

from mtrl.agent import utils as agent_utils
from mtrl.env.types import EnvType
from mtrl.env.vec_env import VecEnv  # type: ignore
from mtrl.experiment import collective_experiment
from mtrl.replay_buffer import ReplayBufferSample
from mtrl.utils.types import ConfigType, EnvMetaDataType, EnvsDictType, ListConfigType

import torch.distributions as D
from scipy.stats import norm
import math
import os, csv
from datetime import datetime


class Experiment(collective_experiment.Experiment):
    def __init__(self, config: ConfigType, experiment_id: str = "0"):
        """Experiment Class to manage the lifecycle of a collective learning model.

        Args:
            config (ConfigType):
            experiment_id (str, optional): Defaults to "0".
        """
        super().__init__(config, experiment_id)
        self.eval_modes_to_env_ids = self.create_eval_modes_to_env_ids()
        self.should_reset_env_manually = False
        self.metrics_to_track = {
            x[0] for x in self.config.metrics["train"] if not x[0].endswith("_")
        }
        # Early stopping:
        from collections import deque
        self.success_history = deque(maxlen=self.config.experiment.early_stopping_window)


    def build_envs(self) -> Tuple[EnvsDictType, EnvMetaDataType]:
        """Build environments and return env-related metadata"""
        if "dmcontrol" not in self.config.env.name:
            raise NotImplementedError
        envs: EnvsDictType = {}
        mode = "train"
        env_id_list = self.config.env[mode]
        num_envs = len(env_id_list)
        seed_list = list(range(1, num_envs + 1))
        mode_list = [mode for _ in range(num_envs)]

        envs[mode] = hydra.utils.instantiate(
            self.config.env.builder,
            env_id_list=env_id_list,
            seed_list=seed_list,
            mode_list=mode_list,
        )
        envs["eval"] = self._create_dmcontrol_vec_envs_for_eval()
        metadata = self.get_env_metadata(env=envs["train"])
        return envs, metadata

    def create_eval_modes_to_env_ids(self) -> Dict[str, List[int]]:
        """Map each eval mode to a list of environment index.

            The eval modes are of the form `eval_xyz` where `xyz` specifies
            the specific type of evaluation. For example. `eval_interpolation`
            means that we are using interpolation environments for evaluation.
            The eval moe can also be set to just `eval`.

        Returns:
            Dict[str, List[int]]: dictionary with different eval modes as
                keys and list of environment index as values.
        """
        eval_modes_to_env_ids: Dict[str, List[int]] = {}
        eval_modes = [
            key for key in self.config.metrics.keys() if not key.startswith("train")
        ]
        for mode in eval_modes:
            if "_" in mode:
                _mode, _submode = mode.split("_")
                env_ids = self.config.env[_mode][_submode]
                eval_modes_to_env_ids[mode] = env_ids
            elif mode != "eval":
                raise ValueError(f"eval mode = `{mode}`` is not supported.")
        return eval_modes_to_env_ids

    def _create_dmcontrol_vec_envs_for_eval(self) -> EnvType:
        """Method to create the vec env with multiple copies of the same
        environment. It is useful when evaluating the agent multiple times
        in the same env.

        The vec env is organized as follows - number of modes x number of tasks per mode x number of episodes per task

        """

        env_id_list: List[str] = []
        seed_list: List[int] = []
        mode_list: List[str] = []
        num_episodes_per_env = self.config.experiment.num_eval_episodes
        for mode in self.config.metrics.keys():
            if mode == "train":
                continue

            if "_" in mode:
                _mode, _submode = mode.split("_")
                if _mode != "eval":
                    raise ValueError("`mode` does not start with `eval_`")
                if not isinstance(self.config.env.eval, ConfigType):
                    raise ValueError(
                        f"""`self.config.env.eval` should either be a DictConfig.
                        Detected type is {type(self.config.env.eval)}"""
                    )
                if _submode in self.config.env.eval:
                    for _id in self.config.env[_mode][_submode]:
                        env_id_list += [_id for _ in range(num_episodes_per_env)]
                        seed_list += list(range(1, num_episodes_per_env + 1))
                        mode_list += [_submode for _ in range(num_episodes_per_env)]
            elif mode == "eval":
                if isinstance(self.config.env.eval, ListConfigType):
                    for _id in self.config.env[mode]:
                        env_id_list += [_id for _ in range(num_episodes_per_env)]
                        seed_list += list(range(1, num_episodes_per_env + 1))
                        mode_list += [mode for _ in range(num_episodes_per_env)]
            else:
                raise ValueError(f"eval mode = `{mode}` is not supported.")
        env = hydra.utils.instantiate(
            self.config.env.builder,
            env_id_list=env_id_list,
            seed_list=seed_list,
            mode_list=mode_list,
        )

        return env
    
    def run(self):
        if self.config.experiment.mode == 'train_worker':
            self.run_training_worker()
        elif self.config.experiment.mode == 'record':
            self.run_record()
        elif self.config.experiment.mode == 'distill_collective_transformer':
            self.distill_collective_transformer()
        elif self.config.experiment.mode == 'evaluate_collective_transformer':
            self.evaluate_collective_transformer()
        elif self.config.experiment.mode == 'online_distill_collective_transformer':
            self.run_online_distillation()
        elif self.config.experiment.mode == 'train_student':
            self.run_distilling_student_online()
        elif self.config.experiment.mode == 'train_student_finetuning':
            self.run_train_student_finetuning()
        elif self.config.experiment.mode == 'distill_policy':
            self.distill_policy()
        else:
            raise NotImplemented

    def save_all_buffers_and_models(self, step):
        print("Saving all buffers and models ...")
        exp_config = self.config.experiment
        for i in range(self.num_envs):
            if exp_config.save.model:
                self.agent[i].save(
                    self.model_dir[i],
                    step=step,
                    retain_last_n=exp_config.save.model.retain_last_n,
                )
            if exp_config.save.buffer.should_save:
                self.replay_buffer[i].save(
                    self.buffer_dir[i],
                    size_per_chunk=exp_config.save.buffer.size_per_chunk,
                    num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                )
                self.replay_buffer_distill_tmp[i].save(
                    self.buffer_dir_distill_tmp[i],
                    size_per_chunk=exp_config.save.buffer.size_per_chunk,
                    num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                )


    def run_training_worker(self):
        """Run the experiment."""
        exp_config = self.config.experiment


        if exp_config.name == None:
            # Generate a timestamp like 2025-07-29_06-45-12
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            reward_csv_path = os.path.join(
                self.config.setup.log_dir, 
                f"csv/worker/{self.config.env.benchmark.env_name}/reward_{timestamp}.csv"
            )
        else:
            reward_csv_path = os.path.join(
                self.config.setup.log_dir, 
                f"csv/worker/{self.config.env.benchmark.env_name}/{exp_config.name}.csv"
            )

        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(reward_csv_path), exist_ok=True)

        # Create the CSV file with header only once
        with open(reward_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Reward"])  # Step=global training step, Reward=episode reward


        vec_env = self.envs["train"]

        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]  # (num_envs, 1)

        if "success" in self.metrics_to_track:
            success = np.full(shape=vec_env.num_envs, fill_value=0.0)

        info = {}

        assert exp_config.col_sampling_freq >= exp_config.col_training_samples
        assert self.start_step >= 0
        assert exp_config.col_sampling_freq % self.max_episode_steps == 0
        episode = self.start_step // self.max_episode_steps

        start_time = time.time()

        col_obs = vec_env.reset()

        action = np.asarray([self.action_space.sample() for _ in range(vec_env.num_envs)])

        train_mode = ["train" for _ in range(vec_env.num_envs)]
        
        step = exp_config.num_train_step
        for step in range(self.start_step, exp_config.num_train_steps):
            # get action
            
            # steps without training
            if step < exp_config.init_steps:
                action = np.asarray(
                    [self.action_space.sample() for _ in range(vec_env.num_envs)]
                )  # (num_envs, action_dim)

            # scripted policy
            elif exp_config.follow_scripted:
                for i in self.env_indices_i:
                    env_obs_i = {
                        key: value[self.env_indices[i]] for key, value in col_obs.items()
                    }
                    action[self.env_indices[i]] = np.clip(self.scripted_policies[i].get_action(self.scripted_policies[i], env_obs_i["env_obs"].cpu().numpy()), -1, 1)

            # training agent
            else:
                for i in self.env_indices_i:
                    with agent_utils.eval_mode(self.agent[i]):
                        env_obs_i = { #watch: correct value
                            key: value[self.env_indices[i]] for key, value in col_obs.items()
                        }
                        action[self.env_indices[i]] = self.agent[i].sample_action(
                            multitask_obs=env_obs_i,
                            modes=[
                                train_mode,
                            ],
                        )

            # run training update
            if step >= exp_config.init_steps and exp_config.train_worker:
                # update init_steps times after skipping them before, otherwise update only once
                num_updates = (
                    exp_config.init_steps if step == exp_config.init_steps else 1
                )
                for _ in range(num_updates):
                    for i in self.env_indices_i:
                        self.agent[i].update(self.replay_buffer[i], self.logger, step)

            # generate data for distillation: copy latest transitions
            if step % exp_config.col_sampling_freq == 0 and step > exp_config.col_training_warmup:
                for i in self.env_indices_i:
                    idx = self.replay_buffer[i].idx
                    env_obs_arr = self.replay_buffer[i].env_obses[idx-exp_config.col_training_samples:idx]
                    actions_arr = self.replay_buffer[i].actions[idx-exp_config.col_training_samples:idx]
                    rewards_arr = self.replay_buffer[i].rewards[idx-exp_config.col_training_samples:idx]
                    next_env_obs_arr = self.replay_buffer[i].next_env_obses[idx-exp_config.col_training_samples:idx]
                    not_done_arr = np.logical_not(self.replay_buffer[i].not_dones[idx-exp_config.col_training_samples:idx])
                    task_obs_arr = self.replay_buffer[i].task_obs[idx-exp_config.col_training_samples:idx]
                    length = len(env_obs_arr)
                    if length < exp_config.col_training_samples:
                        env_obs_arr = np.concatenate((self.replay_buffer[i].env_obses[-exp_config.col_training_samples-length:], env_obs_arr))
                        actions_arr = np.concatenate((self.replay_buffer[i].actions[-exp_config.col_training_samples-length:], actions_arr))
                        rewards_arr = np.concatenate((self.replay_buffer[i].rewards[-exp_config.col_training_samples-length:], rewards_arr))
                        next_env_obs_arr = np.concatenate((self.replay_buffer[i].next_env_obses[-exp_config.col_training_samples-length:], next_env_obs_arr))
                        not_done_arr = np.concatenate((np.logical_not(self.replay_buffer[i].not_dones[-exp_config.col_training_samples-length:]), not_done_arr))
                        task_obs_arr = np.concatenate((self.replay_buffer[i].task_obs[-exp_config.col_training_samples-length:], task_obs_arr))
                    self.replay_buffer_distill_tmp[i].add_array(
                        env_obs_arr,
                        actions_arr,
                        rewards_arr,
                        next_env_obs_arr,
                        not_done_arr,
                        task_obs_arr,
                        exp_config.col_training_samples,
                    )
                    print(f"----- Distill tmp Idx {self.replay_buffer_distill_tmp[i].idx} -----")

            next_col_obs, reward, done, info = vec_env.step(action)
            if exp_config.use_unscaled:
                reward = info['unscaled_reward']

            episode_reward += reward
            if "success" in self.metrics_to_track:
                success += info['success']

            if self.should_reset_env_manually:
                if (episode_step[0] + 1) % self.max_episode_steps == 0:
                    # we do a +1 because we started the counting from 0 and episode_step is incremented after updating the buffer
                    # reset environment after episode
                    if exp_config.reset_goal_state:
                        # reset with an order
                        if exp_config.reset_goal_state_in_order:
                            self.reset_goal_locations_in_order()
                        # reset randomly
                        else:
                            self.reset_goal_locations()
                    next_col_obs = vec_env.reset()
                    
            # allow infinite bootstrap
            for index in self.env_indices_i:
                done_bool = (
                    0
                    if episode_step[self.env_indices[index]] + 1 == self.max_episode_steps
                    else float(done[self.env_indices[index]])
                )
                self.replay_buffer[index].add(
                    col_obs["env_obs"][self.env_indices[index]],
                    action[self.env_indices[index]],
                    reward[self.env_indices[index]],
                    next_col_obs["env_obs"][self.env_indices[index]],
                    done_bool,
                    task_obs=self.env_indices[index],
                )

            if step % self.max_episode_steps == 0:
                self.logger.log("train/episode", episode, step)
                if step > 0:
                    # Compute mean episode reward across all envs
                    avg_episode_reward = episode_reward.mean()

                    # Append to CSV
                    with open(reward_csv_path, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([step, avg_episode_reward])

                    if "success" in self.metrics_to_track:
                        success = (success > 0).astype("float")
                        for index in self.env_indices:
                            self.logger.log(
                                f"train/success_env_index_{index}",
                                success[index],
                                step,
                            )
                        success = success[self.env_indices]
                        self.logger.log("train/success", success.mean(), step)
                    for index in self.env_indices:
                        self.logger.log(
                            f"train/episode_reward_env_index_{index}",
                            episode_reward[index],
                            step,
                        )

                    self.logger.log("train/duration", time.time() - start_time, step)
                    start_time = time.time()
                    self.logger.dump(step)

                    # EARLY STOPPING: stop training if success rate is good enough
                    self.success_history.append(success.mean())
                    # Truncate to sliding window
                    if len(self.success_history) > exp_config.early_stopping_window:
                        self.success_history.pop(0)

                    # only stop if the last window many evaluations achieve a score above the threshold
                    if len(self.success_history) == exp_config.early_stopping_window and np.mean(self.success_history) >= exp_config.early_stopping_threshold:
                        print(f"Early stopping triggered at step {step} with avg success over {exp_config.early_stopping_window} evals: {np.mean(self.success_history):.2f}")
                        break



                # evaluate agent periodically
                if step % exp_config.eval_freq == 0:
                    self.evaluate_vec_env_of_tasks(
                        vec_env=self.envs["eval"], step=step, episode=episode, agent="worker"
                    )

                episode += 1
                episode_reward = np.full(shape=vec_env.num_envs, fill_value=0.0)
                if "success" in self.metrics_to_track:
                    success = np.full(shape=vec_env.num_envs, fill_value=0.0)
                
                if step % exp_config.save_freq == 0:
                    self.save_all_buffers_and_models(step)
                if step % self.config.experiment.save_video_freq == 0 and self.config.experiment.save_video and step > 0:
                    print('Recording videos ...')
                    self.record_videos(eval_agent="worker", tag=f"step{step}")

            col_obs = next_col_obs
            episode_step += 1

        print(f"End eval")

        # Evaluation of the policy
        for i in range(5):
            self.evaluate_vec_env_of_tasks(
                vec_env=self.envs["eval"], step=exp_config.num_train_steps+1+i, episode=episode, agent="worker"
            )
        if step != None:
            self.save_all_buffers_and_models(step)

        print(f"Creating the DistilledReplayBuffer - {self.replay_buffer_distill_tmp[0].idx} samples collected")

        for i in range(self.num_envs):
            for idx in range(0, self.replay_buffer_distill_tmp[i].idx, exp_config.col_training_samples):
                size = np.min((exp_config.col_training_samples, self.replay_buffer_distill_tmp[i].idx-idx))
                batch = ReplayBufferSample(
                    torch.as_tensor(self.replay_buffer_distill_tmp[i].env_obses[idx:idx+size], device=self.device).float(), 
                    torch.as_tensor(self.replay_buffer_distill_tmp[i].actions[idx:idx+size], device=self.device),
                    torch.as_tensor(self.replay_buffer_distill_tmp[i].rewards[idx:idx+size], device=self.device),
                    torch.as_tensor(self.replay_buffer_distill_tmp[i].next_env_obses[idx:idx+size], device=self.device),
                    torch.as_tensor(self.replay_buffer_distill_tmp[i].not_dones[idx:idx+size], device=self.device),
                    torch.as_tensor(self.replay_buffer_distill_tmp[i].task_obs[idx:idx+size], device=self.device), 
                    None, #idxs
                )
                q_targets, teacher_mu, teacher_log_std = self.agent[i].compute_q_target_and_policy_density_from_batch(batch)
                self.replay_buffer_distill[i].add_array(
                    self.replay_buffer_distill_tmp[i].env_obses[idx:idx+size],
                    self.replay_buffer_distill_tmp[i].actions[idx:idx+size],
                    self.replay_buffer_distill_tmp[i].rewards[idx:idx+size],
                    self.replay_buffer_distill_tmp[i].next_env_obses[idx:idx+size],
                    np.logical_not(self.replay_buffer_distill_tmp[i].not_dones[idx:idx+size]),
                    np.array([[self.task_num[i]] for _ in range(size)]),
                    q_targets.cpu().numpy(),
                    teacher_mu.cpu().numpy(), 
                    teacher_log_std.cpu().numpy(),
                    size,
                )

        for i in range(self.num_envs):
            self.replay_buffer_distill[i].save(
                    self.buffer_dir_distill[i],
                    size_per_chunk=exp_config.save.buffer.size_per_chunk,
                    num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                )

        if self.config.experiment.save_video:
            print('start recording videos ...')
            self.record_videos(eval_agent="worker")
            print('video recording finished. Check folder:{}'.format(self.video.dir_name))

        for i in range(self.num_envs):
            print(f"[{i}/{self.num_envs}] Replaybuffer-distill: {self.replay_buffer_distill[i].idx} | Replaybuffer: {self.replay_buffer[i].idx} \
                  | Replaybuffer-distill-tmp: {self.replay_buffer_distill_tmp[i].idx}")

        if self.config.experiment.save.buffer.delete_replaybuffer:
            for i in range(self.num_envs):
                self.replay_buffer[i].delete_from_filesystem(self.buffer_dir[i])
                self.replay_buffer_distill_tmp[i].delete_from_filesystem(self.buffer_dir_distill_tmp[i])
                self.replay_buffer_distill[i].delete_from_filesystem(self.buffer_dir_distill[i])

        self.close_envs()

        print("finished learning")

    def run_record(self):
        """Run the experiment."""
        exp_config = self.config.experiment

        vec_env = self.envs["train"]

        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]  # (num_envs, 1)

        if "success" in self.metrics_to_track:
            success = np.full(shape=vec_env.num_envs, fill_value=0.0)

        info = {}

        assert self.start_step >= 0
        episode = self.start_step // self.max_episode_steps

        start_time = time.time()

        col_obs = vec_env.reset()  # (num_envs, 9, 84, 84)

        mu = action = np.asarray([self.action_space.sample() for _ in range(vec_env.num_envs)])

        for step in range(self.start_step, exp_config.num_recording_steps):
            # sample action
            for i in self.env_indices_i:
                env_obs_i = {
                    key: value[self.env_indices[i]] for key, value in col_obs.items()
                }
                mu[self.env_indices[i]] = np.clip(self.scripted_policies[i].get_action(self.scripted_policies[i], env_obs_i["env_obs"].cpu().numpy()), -1, 1)
                if (np.random.rand(1) > exp_config.scripted_porb and exp_config.init_record_step < step) or \
                    step > exp_config.only_scripted_step:
                    action[self.env_indices[i]] = mu[self.env_indices[i]]
                else:
                    action[self.env_indices[i]] = self.action_space.sample()

            next_col_obs, reward, done, info = vec_env.step(action)
            if exp_config.use_unscaled:
                reward = info['unscaled_reward']

            episode_reward += reward
            if "success" in self.metrics_to_track:
                success += info['success']

            # allow infinite bootstrap
            for index in self.env_indices_i:
                done_bool = (
                    0
                    if episode_step[self.env_indices[index]] + 1 == self.max_episode_steps
                    else float(done[self.env_indices[index]])
                )
                
                self.replay_buffer_distill[index].add(
                    col_obs["env_obs"][self.env_indices[index]],
                    action[self.env_indices[index]],
                    reward[self.env_indices[index]],
                    next_col_obs["env_obs"][self.env_indices[index]],
                    done_bool,
                    task_obs=self.task_num[self.env_indices[index]],
                    #task_obs=self.env_indices[index],
                    q_value=0., 
                    mu=mu[self.env_indices[i]], 
                    log_std=np.random.rand(4),
                )

            if self.should_reset_env_manually:
                if (episode_step[0] + 1) % self.max_episode_steps == 0:
                    # we do a +1 because we started the counting from 0 and episode_step is incremented after updating the buffer
                    if exp_config.reset_goal_state:
                        self.reset_goal_locations_in_order()
                    next_col_obs = vec_env.reset()

            if step % self.max_episode_steps == 0 and step > 0:
                if step > 0:
                    if "success" in self.metrics_to_track:
                        success = (success > 0).astype("float")
                        for index in self.env_indices:
                            self.logger.log(
                                f"train/success_env_index_{index}",
                                success[index],
                                step,
                            )
                        success = success[self.env_indices]
                        self.logger.log("train/success", success.mean(), step)
                    for index in self.env_indices:
                        self.logger.log(
                            f"train/episode_reward_env_index_{index}",
                            episode_reward[index],
                            step,
                        )

                    self.logger.log("train/duration", time.time() - start_time, step)
                    start_time = time.time()
                    self.logger.dump(step)

                episode += 1
                episode_reward = np.full(shape=vec_env.num_envs, fill_value=0.0)
                if "success" in self.metrics_to_track:
                    success = np.full(shape=vec_env.num_envs, fill_value=0.0)

                self.logger.log("train/episode", episode, step)

                if step % exp_config.save_freq == 0:
                    if exp_config.save.buffer.should_save:
                        for i in range(self.num_envs):
                            self.replay_buffer_distill[i].save(
                                self.buffer_dir_distill[i],
                                size_per_chunk=exp_config.save.buffer.size_per_chunk,
                                num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                            )
                    
            col_obs = next_col_obs
            episode_step += 1

        print(f"End eval")
        print(f"Lengths of replaybuffers: {[x.idx for x in self.replay_buffer_distill]}")

        if exp_config.save.buffer.should_save:
            for i in range(self.num_envs):
                self.replay_buffer_distill[i].save(
                    self.buffer_dir_distill[i],
                    size_per_chunk=exp_config.save.buffer.size_per_chunk,
                    num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                )

        if self.config.experiment.save.buffer.delete_replaybuffer:
            for i in range(self.num_envs):
                self.replay_buffer_distill[i].delete_from_filesystem(self.buffer_dir_distill[i])

        self.close_envs()

        print("finished learning")

    def run_online_distillation(self):
        """Run training of student experiment."""
        exit()
        exp_config = self.config.experiment

        vec_env = self.envs["train"]

        assert self.start_step >= 0


        action = np.asarray([self.action_space.sample() for _ in range(vec_env.num_envs)])

        print(f"Starting online training")

        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]  # (num_envs, 1)

        if "success" in self.metrics_to_track:
            success = np.full(shape=vec_env.num_envs, fill_value=0.0)

        info = {}
        episode = self.start_step // self.max_episode_steps

        start_time = time.time()

        col_obs = vec_env.reset()
        states = torch.cat((col_obs['env_obs'][:,:18], col_obs['env_obs'][:,36:]), dim=1).float()[:, None]
        actions = torch.empty(vec_env.num_envs, 0, 4)
        rewards = torch.empty(vec_env.num_envs, 0, 1)

        action = np.asarray([self.action_space.sample() for _ in range(vec_env.num_envs)])
        eval_mode = ["eval" for _ in range(vec_env.num_envs)]

        for step in range(self.start_step, exp_config.num_online_train_step):
            if step % self.max_episode_steps == 0:
                if step > 0:
                    if "success" in self.metrics_to_track:
                        success = (success > 0).astype("float")
                        for index in self.env_indices:
                            self.logger.log(
                                f"train/success_env_index_{index}",
                                success[index],
                                step,
                            )
                        success = success[self.env_indices]
                        self.logger.log("train/success", success.mean(), step)
                    for index in self.env_indices:
                        self.logger.log(
                            f"train/episode_reward_env_index_{index}",
                            episode_reward[index],
                            step,
                        )

                    self.logger.log("train/duration", time.time() - start_time, step)
                    start_time = time.time()
                    self.logger.dump(step)

                # evaluate agent periodically
                if step % exp_config.eval_freq == 0:
                    self.evaluate_transformer(
                        agent=self.col_agent, vec_env=self.envs["eval"], step=step, episode=episode, seq_len=self.seq_len, sample_actions=True, reward_unscaled=exp_config.use_unscaled
                    )

                episode += 1
                episode_reward = np.full(shape=vec_env.num_envs, fill_value=0.0)
                if "success" in self.metrics_to_track:
                    success = np.full(shape=vec_env.num_envs, fill_value=0.0)

                self.logger.log("train/episode", episode, step)

                if step % exp_config.save_freq == 0:
                    if exp_config.save.buffer.should_save:
                        self.replay_buffer.save(
                            self.buffer_dir,
                            size_per_chunk=exp_config.save.buffer.size_per_chunk,
                            num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                        )

                if step % self.config.experiment.save_video_freq == 0 and self.config.experiment.save_video and step > 0:
                    print('Recording videos ...')
                    self.record_videos_for_transformer(self.col_agent, seq_len=self.seq_len)
            
            if step < exp_config.init_steps_online_tf and not exp_config.follow_col_network:
                action = np.asarray(
                    [self.action_space.sample() for _ in range(vec_env.num_envs)]
                )  # (num_envs, action_dim)
            elif step > exp_config.stu_expert_steps:
                for i in self.env_indices_i:
                    with agent_utils.eval_mode(self.expert[i]):
                        env_obs_i = {
                            key: value[self.env_indices[i]] for key, value in col_obs.items()
                        }
                        if exp_config.distill_scripted_policy:
                            action[self.env_indices[i]] = np.clip(self.scripted_policies[i].get_action(self.scripted_policies[i], env_obs_i["env_obs"].cpu().numpy()), -1, 1)
                        else:
                            action[self.env_indices[i]] = self.expert[i].sample_action(
                                multitask_obs=env_obs_i,
                                modes=[eval_mode,],
                            )
            else:
                with agent_utils.eval_mode(self.col_agent):
                    action[self.env_indices] = self.col_agent.sample_action(
                    #action[self.env_indices] = self.col_agent.select_action(
                        states=states[self.env_indices],
                        actions=actions[self.env_indices],
                        rewards=rewards[self.env_indices],
                        task_ids=torch.tensor(self.task_num[self.env_indices_i])
                    )

            # safe the encoding for later evalutation
            encoding = self.col_agent.calculate_task_encoding(
                states=states[self.env_indices],
                actions=actions[self.env_indices],
                rewards=rewards[self.env_indices],
                task_ids=torch.tensor(self.task_num[self.env_indices_i])
            )

            # run training update
            if step >= exp_config.init_steps_online_tf and step % exp_config.actor_update_freq == 0 and step < exp_config.stu_expert_steps:
                num_updates = (
                    exp_config.init_steps_online_tf if step == exp_config.init_steps_online_tf else 1
                )
                for _ in range(num_updates):
                    self.col_agent.distill_actor(self.replay_buffer, self.logger, step, cls_token=self.cls_token, tb_log=True)

            next_col_obs, reward, done, info = vec_env.step(action)

            if exp_config.use_unscaled:
                reward = info['unscaled_reward']

            episode_reward += reward
            if "success" in self.metrics_to_track:
                success += info['success']

            new_state = torch.cat((next_col_obs['env_obs'][:,:18], next_col_obs['env_obs'][:,36:]), dim=1).float()[:, None]
            new_action = torch.tensor(action).float()[:, None]
            new_reward = torch.tensor(reward).float()[:, None, None]

            states = torch.cat((states[:, -self.seq_len+1:], new_state), dim=1)
            actions = torch.cat((actions[:, -self.seq_len+2:], new_action), dim=1)
            rewards = torch.cat((rewards[:, -self.seq_len+2:], new_reward), dim=1)

            if self.should_reset_env_manually:
                if (episode_step[0] + 1) % self.max_episode_steps == 0:
                    if exp_config.reset_goal_state:
                        self.reset_goal_locations_in_order()
                    next_col_obs = vec_env.reset()
                    states = torch.cat((next_col_obs['env_obs'][:,:18], next_col_obs['env_obs'][:,36:]), dim=1).float()[:, None]
                    actions = torch.empty(vec_env.num_envs, 0, 4)
                    rewards = torch.empty(vec_env.num_envs, 0, 1)
                    
            # allow infinite bootstrap
            for index in self.env_indices_i:
                q_target, mu, log_std = self.expert[index].compute_q_target_and_policy_density(
                                            env_obs=col_obs["env_obs"][self.env_indices[index]][None].float().to(self.device),
                                            next_env_obs=next_col_obs["env_obs"][self.env_indices[index]][None].float().to(self.device),
                                            task_obs=torch.tensor(self.env_indices[index]).to(self.device)
                                        )
                if exp_config.distill_scripted_policy:
                    mu = np.clip(self.scripted_policies[index].get_action(self.scripted_policies[index], col_obs["env_obs"][self.env_indices[index]].cpu().numpy()), -1, 1)
                else:
                    mu=mu.cpu().numpy()
                done_bool = (
                    0
                    if episode_step[self.env_indices[index]] + 1 == self.max_episode_steps
                    else float(done[self.env_indices[index]])
                )
                self.replay_buffer.add(
                    col_obs["env_obs"][self.env_indices[index]],
                    next_col_obs["env_obs"][self.env_indices[index]],
                    action[self.env_indices[index]],
                    reward[self.env_indices[index]],
                    done=done_bool,
                    task_obs=self.task_num[index],
                    encoding=encoding[index].cpu().numpy(),
                    q_value=q_target.cpu().numpy(),
                    mu=mu,
                    log_std=log_std.cpu().numpy(),
                )

            col_obs = next_col_obs
            episode_step += 1

        print(f"Finished online training")

        if self.config.experiment.save_video:
            print('start recording videos ...')
            self.record_videos_for_transformer(self.col_agent, seq_len=self.seq_len)
            print('video recording finished. Check folder:{}'.format(self.video.dir_name))

        if self.config.experiment.save.buffer.delete_replaybuffer:
            self.replay_buffer.delete_from_filesystem(self.buffer_dir)

        self.close_envs()

        print("Finished online transformer learning")

    def run_train_student_finetuning(self):
        """Run training of student experiment."""
        exp_config = self.config.experiment

        vec_env = self.envs["train"]

        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]

        if "success" in self.metrics_to_track:
            success = np.full(shape=vec_env.num_envs, fill_value=0.0)

        info = {}

        assert self.start_step >= 0
        episode = self.start_step // self.max_episode_steps

        start_time = time.time()

        col_obs = vec_env.reset()
        states = torch.cat((col_obs['env_obs'][:,:18], col_obs['env_obs'][:,36:]), dim=1).float()[:, None]
        actions = torch.empty(vec_env.num_envs, 0, 4)
        rewards = torch.empty(vec_env.num_envs, 0, 1)

        action = np.asarray([self.action_space.sample() for _ in range(vec_env.num_envs)])

        train_mode = ["train" for _ in range(vec_env.num_envs)]

        print("Start finetuning ... ")

        for step in range(self.start_step, exp_config.num_student_online_trainsteps):
            if step % self.max_episode_steps == 0:
                if step > 0:
                    if "success" in self.metrics_to_track:
                        success = (success > 0).astype("float")
                        for index in self.env_indices:
                            self.logger.log(
                                f"train/success_env_index_{index}",
                                success[index],
                                step,
                            )
                        success = success[self.env_indices]
                        self.logger.log("train/success", success.mean(), step)
                    for index in self.env_indices:
                        self.logger.log(
                            f"train/episode_reward_env_index_{index}",
                            episode_reward[index],
                            step,
                        )

                    self.logger.log("train/duration", time.time() - start_time, step)
                    start_time = time.time()
                    self.logger.dump(step)

                # evaluate agent periodically
                if step % exp_config.eval_freq == 0:
                    self.evaluate_transformer(
                        agent=self.student, vec_env=self.envs["eval"], step=step, episode=episode, seq_len=self.seq_len, sample_actions=True, reward_unscaled=exp_config.use_unscaled
                    )

                episode += 1
                episode_reward = np.full(shape=vec_env.num_envs, fill_value=0.0)
                if "success" in self.metrics_to_track:
                    success = np.full(shape=vec_env.num_envs, fill_value=0.0)

                self.logger.log("train/episode", episode, step)

                if step % exp_config.save_freq == 0:
                    if exp_config.save.model:
                        self.student.save(
                            self.student_model_dir,
                            step=step,
                            retain_last_n=exp_config.save.model.retain_last_n,
                        )
                    if exp_config.save.buffer.should_save:
                        self.replay_buffer.save(
                            self.buffer_dir,
                            size_per_chunk=exp_config.save.buffer.size_per_chunk,
                            num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                        )

                if step % self.config.experiment.save_video_freq == 0 and self.config.experiment.save_video and step == 0:
                    print('Recording videos ...')
                    self.record_videos_for_transformer(self.student, seq_len=self.seq_len)

            # safe the encoding for later evalutation
            encoding = self.student.calculate_task_encoding(
                states=states[self.env_indices],
                actions=actions[self.env_indices],
                rewards=rewards[self.env_indices],
                task_ids=torch.tensor(self.task_num[self.env_indices_i])
            )

            with agent_utils.eval_mode(self.student):
                action[self.env_indices] = self.student.sample_action(
                    states=states[self.env_indices],
                    actions=actions[self.env_indices],
                    rewards=rewards[self.env_indices],
                    task_ids=torch.tensor(self.task_num[self.env_indices_i])
                )

            # run training update
            if step >= exp_config.init_steps_stu:
                num_updates = (
                    exp_config.init_steps_stu if step == exp_config.init_steps_stu else 1
                )
                for _ in range(num_updates):
                   self.student.update(self.replay_buffer, self.logger, step)

            next_col_obs, reward, done, info = vec_env.step(action)

            if exp_config.use_unscaled:
                reward = info['unscaled_reward']

            episode_reward += reward
            if "success" in self.metrics_to_track:
                success += info['success']

            new_state = torch.cat((next_col_obs['env_obs'][:,:18], next_col_obs['env_obs'][:,36:]), dim=1).float()[:, None]
            new_action = torch.tensor(action).float()[:, None]
            new_reward = torch.tensor(reward).float()[:, None, None]

            states = torch.cat((states[:, -self.seq_len+1:], new_state), dim=1)
            actions = torch.cat((actions[:, -self.seq_len+2:], new_action), dim=1)
            rewards = torch.cat((rewards[:, -self.seq_len+2:], new_reward), dim=1)

            if self.should_reset_env_manually:
                if (episode_step[0] + 1) % self.max_episode_steps == 0:
                    next_col_obs = vec_env.reset()
                    states = torch.cat((next_col_obs['env_obs'][:,:18], next_col_obs['env_obs'][:,36:]), dim=1).float()[:, None]
                    actions = torch.empty(vec_env.num_envs, 0, 4)
                    rewards = torch.empty(vec_env.num_envs, 0, 1)
                    
            # allow infinite bootstrap
            for index in self.env_indices_i:
                done_bool = (
                    0
                    if episode_step[self.env_indices[index]] + 1 == self.max_episode_steps
                    else float(done[self.env_indices[index]])
                )
                self.replay_buffer.add(
                    col_obs["env_obs"][self.env_indices[index]],
                    next_col_obs["env_obs"][self.env_indices[index]],
                    action[self.env_indices[index]],
                    reward[self.env_indices[index]],
                    done=done_bool,
                    task_obs=self.env_indices[index],
                    encoding=encoding.detach().cpu().numpy(),
                    q_value=0,
                    mu=[0,0,0,0],
                    log_std=[0,0,0,0],
                )

            col_obs = next_col_obs
            episode_step += 1

        print(f"Finished online training")

        if self.config.experiment.save_video:
            print('start recording videos ...')
            self.record_videos_for_transformer(self.student, tag=f"result", seq_len=self.seq_len)
            print('video recording finished. Check folder:{}'.format(self.video.dir_name))

        if self.config.experiment.save.buffer.delete_replaybuffer:
            self.replay_buffer.delete_from_filesystem(self.buffer_dir)

        self.close_envs()

        print("Finished student learning")

    def run_distilling_student_online(self):
        """Run training of student experiment."""
        exp_config = self.config.experiment

        vec_env = self.envs["train"]

        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]  # (num_envs, 1)

        if "success" in self.metrics_to_track:
            success = np.full(shape=vec_env.num_envs, fill_value=0.0)

        info = {}

        assert self.start_step >= 0
        episode = self.start_step // self.max_episode_steps

        col_obs = vec_env.reset()

        action = np.asarray([self.action_space.sample() for _ in range(vec_env.num_envs)])

        train_mode = ["train" for _ in range(vec_env.num_envs)]
        
        print(f"Starting online training")

        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]  # (num_envs, 1)

        if "success" in self.metrics_to_track:
            success = np.full(shape=vec_env.num_envs, fill_value=0.0)

        info = {}
        episode = self.start_step // self.max_episode_steps

        start_time = time.time()

        col_obs = vec_env.reset()
        states = torch.cat((col_obs['env_obs'][:,:18], col_obs['env_obs'][:,36:]), dim=1).float()[:, None]
        actions = torch.empty(vec_env.num_envs, 0, 4)
        rewards = torch.empty(vec_env.num_envs, 0, 1)

        action = np.asarray([self.action_space.sample() for _ in range(vec_env.num_envs)])

        self.evaluate_transformer(
            agent=self.col_agent, vec_env=self.envs["eval"], step=0, episode=episode, seq_len=self.seq_len, sample_actions=True, reward_unscaled=exp_config.use_unscaled
        )
        self.logger.dump(0)

        for step in range(self.start_step, exp_config.num_student_online_trainsteps2):
            if step % self.max_episode_steps == 0:
                if step > 0:
                    if "success" in self.metrics_to_track:
                        success = (success > 0).astype("float")
                        for index in self.env_indices:
                            self.logger.log(
                                f"train/success_env_index_{index}",
                                success[index],
                                step,
                            )
                        success = success[self.env_indices]
                        self.logger.log("train/success", success.mean(), step)
                    for index in self.env_indices:
                        self.logger.log(
                            f"train/episode_reward_env_index_{index}",
                            episode_reward[index],
                            step,
                        )

                    self.logger.log("train/duration", time.time() - start_time, step)
                    start_time = time.time()
                    self.logger.dump(step)

                # evaluate agent periodically
                if step % exp_config.eval_freq == 0:
                    self.evaluate_vec_env_of_tasks(
                        vec_env=self.envs["eval"], step=step, episode=episode, agent="student"
                    )

                episode += 1
                episode_reward = np.full(shape=vec_env.num_envs, fill_value=0.0)
                if "success" in self.metrics_to_track:
                    success = np.full(shape=vec_env.num_envs, fill_value=0.0)

                self.logger.log("train/episode", episode, step)

                if step % exp_config.save_freq == 0:
                    if exp_config.save.model:
                        self.student.save(
                            self.student_model_dir,
                            step=step,
                            retain_last_n=exp_config.save.model.retain_last_n,
                        )
                    if exp_config.save.buffer.should_save:
                        self.replay_buffer.save(
                            self.buffer_dir,
                            size_per_chunk=exp_config.save.buffer.size_per_chunk,
                            num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                        )

                if step % self.config.experiment.save_video_freq == 0 and self.config.experiment.save_video and step > 0:
                    print('Recording videos ...')
                    self.record_videos(eval_agent="student", tag=f"student_step{step}")

            if step < exp_config.init_steps_stu2:
                action = np.asarray(
                    [self.action_space.sample() for _ in range(vec_env.num_envs)]
                )
            else:
                with agent_utils.eval_mode(self.student):
                    env_obs_i = { 
                        key: value[self.env_indices] for key, value in col_obs.items()
                    }
                    action[self.env_indices] = self.student.sample_action(
                        multitask_obs=env_obs_i,
                        modes=[
                            train_mode,
                        ],
                    )

            # calculate q_target and best action
            if step < exp_config.expert_train_step:
                if not exp_config.use_scripted_mu:

                    mu, log_std, _ = self.col_agent.calculate_q_targets_expert_action(
                                                states=states[self.env_indices],
                                                actions=actions[self.env_indices],
                                                rewards=rewards[self.env_indices],
                                                task_ids=torch.tensor(self.task_num[self.env_indices_i])
                                            )
                else:
                    env_obs_i = {
                        key: value[0] for key, value in col_obs.items()
                    }
                    mu = np.clip(self.scripted_policies[0].get_action(self.scripted_policies[0], env_obs_i["env_obs"].cpu().numpy()), -1, 1)
                    log_std = 1e-1
                    
                expert_sigma = np.clip(np.exp(log_std), a_min=1e-4, a_max=None)
                diff = action - mu
                normalization_term = (1 / (expert_sigma * np.sqrt(2 * math.pi))).prod(axis=1)
                likelihood = ((1 / (expert_sigma * np.sqrt(2 * math.pi))) * np.exp(-0.5 * (diff / expert_sigma) ** 2)).prod(axis=1)
                q_target = likelihood / normalization_term
                
                # safe the encoding for later evalutation
                encoding = self.col_agent.calculate_task_encoding(
                    states=states[self.env_indices],
                    actions=actions[self.env_indices],
                    rewards=rewards[self.env_indices],
                    task_ids=torch.tensor(self.task_num[self.env_indices_i])
                )
            else:
                mu, log_std, q_target = 0., 0., 0.
                encoding = torch.zeros((1,6))

            # run training update
            if step >= exp_config.init_steps_stu2:
                num_updates = (
                    exp_config.init_steps_stu2 if step == exp_config.init_steps_stu2 else 1
                )
                for _ in range(num_updates):
                    if step < exp_config.expert_train_step:
                        expert_weight = exp_config.start_weight * np.clip((1-((step-exp_config.init_steps_stu2-exp_config.constant_expert_weight) /
                                                                            (exp_config.expert_train_step-exp_config.init_steps_stu2-exp_config.constant_expert_weight))), 0., 1.)
                        self.student.update_with_reward_shapening(self.replay_buffer, self.logger, step, expert_weight)
                    else:
                        self.student.update(self.replay_buffer, self.logger, step)

            next_col_obs, reward, done, info = vec_env.step(action)

            if exp_config.use_unscaled:
                reward = info['unscaled_reward']

            episode_reward += reward
            if "success" in self.metrics_to_track:
                success += info['success']

            new_state = torch.cat((next_col_obs['env_obs'][:,:18], next_col_obs['env_obs'][:,36:]), dim=1).float()[:, None]
            new_action = torch.tensor(action).float()[:, None]
            new_reward = torch.tensor(reward).float()[:, None, None]

            states = torch.cat((states[:, -self.seq_len+1:], new_state), dim=1)
            actions = torch.cat((actions[:, -self.seq_len+2:], new_action), dim=1)
            rewards = torch.cat((rewards[:, -self.seq_len+2:], new_reward), dim=1)

            if self.should_reset_env_manually:
                if (episode_step[0] + 1) % self.max_episode_steps == 0:
                    if exp_config.reset_goal_state:
                        self.reset_goal_locations_in_order()
                    next_col_obs = vec_env.reset()
                    states = torch.cat((next_col_obs['env_obs'][:,:18], next_col_obs['env_obs'][:,36:]), dim=1).float()[:, None]
                    actions = torch.empty(vec_env.num_envs, 0, 4)
                    rewards = torch.empty(vec_env.num_envs, 0, 1)
                    
            # allow infinite bootstrap
            for index in self.env_indices_i:
                done_bool = (
                    0
                    if episode_step[self.env_indices[index]] + 1 == self.max_episode_steps
                    else float(done[self.env_indices[index]])
                )
                self.replay_buffer.add(
                    col_obs["env_obs"][self.env_indices[index]],
                    next_col_obs["env_obs"][self.env_indices[index]],
                    action[self.env_indices[index]],
                    reward[self.env_indices[index]],
                    done=done_bool,
                    task_obs=self.env_indices[index],
                    encoding=encoding.detach().cpu().numpy(),
                    q_value=q_target,
                    mu=mu,
                    log_std=log_std,
                )

            col_obs = next_col_obs
            episode_step += 1

        print(f"Finished online training")

        if self.config.experiment.save_video:
            print('start recording videos ...')
            self.record_videos(eval_agent="student", tag="student_result")
            print('video recording finished. Check folder:{}'.format(self.video.dir_name))

        if self.config.experiment.save.buffer.delete_replaybuffer:
            self.replay_buffer.delete_from_filesystem(self.buffer_dir)

        self.close_envs()

        print("Finished student learning")

    def distill_collective_transformer(self):
        """Run the experiment."""
        exp_config = self.config.experiment

        assert self.col_start_step >= 0
        episode = self.col_start_step // self.max_episode_steps
        start_time = time.time()

        if self.col_start_step < exp_config.num_actor_train_step:
            for step in range(self.col_start_step, exp_config.num_actor_train_step):
                # evaluate agent periodically
                if step % exp_config.col_eval_freq == 0:
                    self.logger.log("train/duration", time.time() - start_time, step)
                    start_time = time.time()
                    if step % exp_config.col_eval_freq * 3 == 0:
                        self.evaluate_transformer(
                            agent=self.col_agent, vec_env=self.envs["eval"], step=step, episode=episode, seq_len=self.seq_len, sample_actions=True, reward_unscaled=exp_config.use_unscaled
                        )

                # run distillation updates
                self.col_agent.distill_actor(self.replay_buffer_distill, self.logger, step, tb_log=True)
                self.col_start_step += 1

                if exp_config.save.model and step % exp_config.save_freq_transformer == 0:
                    self.col_agent.save(
                        self.col_model_dir,
                        step=step,
                        retain_last_n=exp_config.save.model.retain_last_n,
                    )

        if self.col_start_step < exp_config.num_actor_train_step+exp_config.num_critic_train_step:
            for step in range(self.col_start_step, exp_config.num_actor_train_step+exp_config.num_critic_train_step):
                # evaluate agent periodically
                if step % exp_config.col_eval_freq == 0:
                    self.logger.log("train/duration", time.time() - start_time, step)
                    start_time = time.time()
                    self.logger.dump(step)

                # run distillation updates
                self.col_agent.distill_critic(self.replay_buffer_distill, self.logger, step, tb_log=True, action_space=self.action_space)
                self.col_start_step += 1

                if exp_config.save.model and step % exp_config.save_freq_transformer == 0:
                    self.col_agent.save(
                        self.col_model_dir,
                        step=step,
                        retain_last_n=exp_config.save.model.retain_last_n,
                    )

        if self.config.experiment.save_video:
            print('start recording videos ...')
            self.record_videos_for_transformer(self.col_agent, seq_len=self.seq_len)
            print('video recording finished. Check folder:{}'.format(self.video.dir_name))

        if exp_config.save.model:
            self.col_agent.save(
                self.col_model_dir,
                step=exp_config.num_actor_train_step+exp_config.num_critic_train_step,
                retain_last_n=exp_config.save.model.retain_last_n,
            )

        self.close_envs()

        print("finished learning")

    def evaluate_collective_transformer(self):
        """Run the experiment."""
        exp_config = self.config.experiment

        assert self.col_start_step >= 0
        episode = self.col_start_step // self.max_episode_steps
        start_time = time.time()
        count = np.zeros(self.config.worker.multitask.num_envs)

        for step in range(exp_config.evaluate_episodes):
            # evaluate agent periodically
            self.logger.log("train/duration", time.time() - start_time, step)
            start_time = time.time()
            if exp_config.evaluate_transformer=="collective_network":
                result = self.evaluate_transformer(
                    agent=self.col_agent, vec_env=self.envs["eval"], step=step, episode=episode, seq_len=self.seq_len, sample_actions=(step==0), reward_unscaled=exp_config.use_unscaled
                )
            elif exp_config.evaluate_transformer=="agent":
                result = self.evaluate_vec_env_of_tasks(
                        vec_env=self.envs["eval"], step=step, episode=episode, agent="worker"
                    )
            else:
                result = self.evaluate_vec_env_of_tasks(
                        vec_env=self.envs["eval"], step=step, episode=episode, agent="scripted"
                    )
            count += result
            self.reset_goal_locations_in_order(env="eval")

        if exp_config.evaluate_transformer=="collective_network":
            for step in range(exp_config.evaluate_critic_rounds):
                self.col_agent.evaluate_critic(self.replay_buffer_distill, self.logger, step, seq_len=self.seq_len, tb_log=True)
                self.logger.log("train/duration", time.time() - start_time, step)
                start_time = time.time()
                self.logger.dump(step)

        if self.config.experiment.save_video:
            for i in range(exp_config.recording_eval_episodes):
                print('start recording videos ...')
                if exp_config.evaluate_transformer=="collective_network":
                    self.record_videos_for_transformer(self.col_agent, tag=f"sample_{i}", seq_len=self.seq_len)
                elif exp_config.evaluate_transformer=="agent":
                    self.record_videos(eval_agent="worker", tag=f"sample_{i}")
                else:
                    self.record_videos(eval_agent="scripted", tag=f"sample_{i}")

                print('video recording finished. Check folder:{}'.format(self.video.dir_name))

        self.close_envs()

        print("finished learning")
        print(f"Evaluation: {count} from {exp_config.evaluate_episodes}")

    def record_videos(self, eval_agent: str = "worker", tag="result", num_eps_per_task_to_record=1):
        """
        Record videos of all envs, each env performs one episodes.
        """
        num_record = num_eps_per_task_to_record
        video_envs = self.list_envs
        
        for env_idx in self.env_indices_i:
            print(f'start recording env {self.env_indices[env_idx]} ...')
            self.reset_goal_locations_for_listenv(env_idx, video_envs[self.env_indices[env_idx]].env.name)
            self.video.reset()
            episode_step = 0           
            env_obs = []
            success = 0.0

            obs = video_envs[self.env_indices[env_idx]].reset()[0]
            env_obs.append(obs)
            multitask_obs = {"env_obs": torch.tensor(env_obs), "task_obs": torch.tensor([self.env_indices[env_idx]])}

            self.video.record(frame=video_envs[self.env_indices[env_idx]].render())
            
            video_envs[self.env_indices[env_idx]].env.mujoco_renderer.viewer.cam.azimuth = 140.0
            video_envs[self.env_indices[env_idx]].env.mujoco_renderer.viewer.cam.elevation = -40.0
            video_envs[self.env_indices[env_idx]].env.mujoco_renderer.viewer.cam.lookat = np.array([0, 0.5, 0.0])
            video_envs[self.env_indices[env_idx]].env.mujoco_renderer.viewer.cam.distance = 2.0

            episode_reward = 0

            for episode_step in range(self.max_episode_steps * num_record):

                # record for env_idx env
                self.video.record(frame=video_envs[self.env_indices[env_idx]].render())
                
                # agent select action
                if eval_agent=="worker":
                    with agent_utils.eval_mode(self.agent[env_idx]):
                        action = self.agent[env_idx].select_action(
                            multitask_obs=multitask_obs, modes=["eval"]
                        )
                elif eval_agent=="scripted":
                    action = np.clip(self.scripted_policies[env_idx].get_action(self.scripted_policies[env_idx], multitask_obs["env_obs"][0].cpu().numpy()), -1, 1)[None]
                else:
                    with agent_utils.eval_mode(self.student):
                        action = self.student.select_action(
                            multitask_obs=multitask_obs, modes=["eval"]
                        )
                
                # interactive with envs get new obs
                env_obs = []
                obs, reward, _, _, info = video_envs[self.env_indices[env_idx]].step(action[0])

                episode_reward += reward
                
                if (episode_step+1) % self.max_episode_steps == 0:
                    obs = video_envs[self.env_indices[env_idx]].reset()[0]
                    self.video.record(frame=self.list_envs[self.env_indices[env_idx]].render())
                env_obs.append(obs)
                success += info['success']

                multitask_obs = {"env_obs": torch.tensor(env_obs), "task_obs": torch.tensor([self.env_indices[env_idx]])}
                episode_step += 1

            success = float(success > 0)
            self.video.save(file_name=f'{eval_agent}_env{self.env_indices[env_idx]}_success_{success}_reward_{int(episode_reward)}_{tag}')
            #video_envs[self.env_indices[env_idx]].close()

    def record_videos_for_transformer(self, agent, tag="result", seq_len=16, num_eps_per_task_to_record=1):
        """
        Record videos of all envs, each env performs one episodes.
        """
        num_record = num_eps_per_task_to_record
        video_envs = self.list_envs
        
        for env_idx in self.env_indices_i:
            print(f'start recording env {self.env_indices[env_idx]} ...')
            self.reset_goal_locations_for_listenv(env_idx, video_envs[self.env_indices[env_idx]].env.name)
            self.video.reset()
            episode_step = 0           
            env_obs = []
            success = 0.0

            obs = video_envs[self.env_indices[env_idx]].reset()[0]
            env_obs.append(obs)
            multitask_obs = {"env_obs": torch.tensor(env_obs), "task_obs": torch.tensor([self.env_indices[env_idx]])}
            states = torch.cat((multitask_obs['env_obs'][:,:18], multitask_obs['env_obs'][:,36:]), dim=1).float()[:, None]
            actions = torch.empty(1, 0, 4)
            rewards = torch.empty(1, 0, 1)

            self.video.record(frame=video_envs[self.env_indices[env_idx]].render())
            
            video_envs[self.env_indices[env_idx]].env.mujoco_renderer.viewer.cam.azimuth = 140.0
            video_envs[self.env_indices[env_idx]].env.mujoco_renderer.viewer.cam.elevation = -40.0
            video_envs[self.env_indices[env_idx]].env.mujoco_renderer.viewer.cam.lookat = np.array([0, 0.5, 0.0])
            video_envs[self.env_indices[env_idx]].env.mujoco_renderer.viewer.cam.distance = 2.0

            episode_reward = 0

            for episode_step in range(self.max_episode_steps * num_record):

                # record for env_idx env
                self.video.record(frame=video_envs[self.env_indices[env_idx]].render())
                
                # agent select action
                with agent_utils.eval_mode(agent):
                    action = agent.select_action(
                        states=states,
                        actions=actions,
                        rewards=rewards,
                        task_ids=torch.tensor(self.task_num[self.env_indices_i]),
                    )
                
                # interactive with envs get new obs
                env_obs = []
                obs, reward, _, _, info = video_envs[self.env_indices[env_idx]].step(action[0])

                new_state = torch.cat((multitask_obs['env_obs'][:,:18], multitask_obs['env_obs'][:,36:]), dim=1).float()[:, None]
                new_action = torch.tensor(action).float()[:, None]
                new_reward = torch.tensor([reward]).float()[:,None,None]
                
                states = torch.cat((states[:, -seq_len+1:], new_state), dim=1)
                actions = torch.cat((actions[:, -seq_len+2:], new_action), dim=1)
                rewards = torch.cat((rewards[:, -seq_len+2:], new_reward), dim=1)

                episode_reward += reward
                
                if (episode_step+1) % self.max_episode_steps == 0:
                    obs = video_envs[self.env_indices[env_idx]].reset()[0]
                    self.video.record(frame=self.list_envs[self.env_indices[env_idx]].render())
                env_obs.append(obs)
                success += info['success']

                multitask_obs = {"env_obs": torch.tensor(env_obs), "task_obs": torch.tensor([self.env_indices[env_idx]])}
                episode_step += 1

            success = float(success > 0)
            self.video.save(file_name=f'transformer_env{self.env_indices[env_idx]}_success_{success}_reward_{int(episode_reward)}_{tag}')


    def collect_trajectory(self, vec_env: VecEnv, num_steps: int) -> None:
        """Collect some trajectories, by unrolling the policy (in train mode),
        and update the replay buffer.
        Args:
            vec_env (VecEnv): environment to collect data from.
            num_steps (int): number of steps to collect data for.

        """
        raise NotImplementedError

    def generate_data_from_col_agent(self, tasks: List[str], num_episodes_per_task: int = 5) -> None:
        """Generate rollouts by running the collective agent policy for each task.
        Transitions are stored with Q-values, policy parameters, and encodings.
        """
        seq_len = self.seq_len
        action_dim = self.action_shape

        print(f"Starting data generation from collective agent for {len(tasks)} tasks...")

        for task_idx, task_name in enumerate(tasks):
            print(f"\n[Task {task_idx}] Generating data for task: {task_name}")
            task_data = random.choice(self.env_name_task_dict[task_name])
            self.envs["train"].call("set_task", task_data)

            for ep in range(num_episodes_per_task):
                print(f"  Episode {ep+1}/{num_episodes_per_task}")
                obs = self.envs["train"].reset()
                done = False
                step = 0
                episode_reward = 0.0

                raw = obs["env_obs"][0]
                raw_np = raw.cpu().numpy() if hasattr(raw, 'cpu') else raw
                s = np.concatenate([raw_np[:18], raw_np[36:]], axis=-1)
                states = torch.tensor(s, dtype=torch.float32, device=self.device)[None, None, :]
                actions = torch.empty(1, 0, action_dim, device=self.device)
                rewards = torch.empty(1, 0, 1, device=self.device)

                while not done and step < self.config.experiment.distill_ds_steps:
                    with agent_utils.eval_mode(self.col_agent):
                        mu, log_std, q_val = self.col_agent.calculate_q_targets_expert_action(
                            states=states,
                            actions=actions,
                            rewards=rewards,
                            task_ids=torch.tensor([[task_idx]], device=self.device),
                        )
                        encoding = self.col_agent.calculate_task_encoding(
                            states=states,
                            actions=actions,
                            rewards=rewards,
                            task_ids=torch.tensor([[task_idx]], device=self.device),
                        )

                    action_np = mu[0]

                    next_obs, reward_arr, done_arr, info = self.envs["train"].step(action_np)
                    reward = float(reward_arr[0])
                    done = bool(done_arr[0])
                    episode_reward += reward

                    raw_next = next_obs["env_obs"][0]
                    raw_next_np = raw_next.cpu().numpy() if hasattr(raw_next, 'cpu') else raw_next
                    s_next = np.concatenate([raw_next_np[:18], raw_next_np[36:]], axis=-1)

                    self.replay_buffer.add(
                        env_obs=raw_np,
                        next_env_obs=raw_next_np,
                        action=action_np,
                        reward=np.array([reward], dtype=np.float32),
                        done=done,
                        task_obs=np.array([task_idx], dtype=np.int64),
                        encoding=encoding.detach().cpu().numpy(),
                        q_value=q_val,
                        mu=mu,
                        log_std=log_std
                    )

                    s = s_next
                    new_state = torch.tensor(s, dtype=torch.float32, device=self.device)[None, None, :]
                    new_action = torch.tensor(action_np, dtype=torch.float32, device=self.device)[None, None, :]
                    new_reward = torch.tensor([[reward]], dtype=torch.float32, device=self.device)[None, :, :]

                    states = torch.cat((states[:, -seq_len+1:], new_state), dim=1)
                    actions = torch.cat((actions[:, -seq_len+2:], new_action), dim=1)
                    rewards = torch.cat((rewards[:, -seq_len+2:], new_reward), dim=1)

                    obs = next_obs
                    step += 1

                print(f"    → Episode done in {step} steps, total reward: {episode_reward:.2f}")

        print("\nData generation complete.")
        print(f"Replay buffer now contains {len(self.replay_buffer)} transitions.")

    def distill_policy(self):
        """Run distillation from col_agent to student using KL divergence on logits."""
        exp_config = self.config.experiment
        #assert hasattr(self, "col_agent") and hasattr(self, "student_agent")

        self.student_agent.train()  # Only the student needs to train

        optimizer = self.student_agent._optimizers["actor"]

        distill_steps = exp_config.distill_steps
        batch_size = exp_config.batch_size
        temperature = exp_config.temperature
        start_time = time.time()

        if self.replay_buffer.is_empty():
            tasks = self.config.experiment.distill_tasks
            self.generate_data_from_col_agent(tasks, num_episodes_per_task=5)


        for step in range(distill_steps):
            print(f"Buffer size: {len(self.replay_buffer)}, batch_size: {batch_size}")
            if len(self.replay_buffer) < batch_size:
                print("Buffer not sufficiently filled. Abort or wait.")
                return

            sample: ReplayBufferSample = self.replay_buffer.sample(batch_size)
            obs = sample.env_obs.to(self.device)


            # Get teacher logits (e.g., Q-values or action logits)
            with torch.no_grad():
                teacher_logits = self.col_agent.get_action_logits(obs) / temperature
                teacher_probs = torch.softmax(teacher_logits, dim=-1)

            # Get student logits
            student_logits = self.student_agent.get_action_logits(obs) / temperature
            student_log_probs = torch.log_softmax(student_logits, dim=-1)

            # KL Divergence loss
            loss = torch.nn.functional.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean',
                log_target=False
            ) * (temperature ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % exp_config.log_interval == 0:
                self.logger.log("distill/loss", loss.item(), step)
                self.logger.log("distill/duration", time.time() - start_time, step)
                self.logger.dump(step)
                print(f"[Distill Step {step}] Loss: {loss.item():.4f}")
                start_time = time.time()

            if exp_config.save.model and step % exp_config.save_freq == 0:
                self.student_agent.save(
                    self.student_model_dir,
                    step=step,
                    retain_last_n=exp_config.save.model.retain_last_n,
                )

        self.close_envs()
        print("Finished policy distillation.")

