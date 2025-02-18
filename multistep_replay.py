import numpy as np
import numpy.random as npr
import torch
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
import copy
from sortedcontainers import SortedSet
import pickle as pkl

import multistep_utils as utils

class ReplayBufferMultistep(object):
    """Buffer to store environment transitions."""
    def __init__(self, 
                obs_shape, 
                action_shape, 
                capacity, 
                batch_size, 
                horizon, 
                device, 
                normalize_obs=False):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.capacity = capacity
        self.batch_size = batch_size
        self.horizon = horizon
        self.device = device

        self.pixels = len(obs_shape) > 1
        self.empty_data()

        self.done_idxs = SortedSet()
        self.global_idx = 0
        self.global_last_save = 0

        self.normalize_obs = normalize_obs

        if normalize_obs:
            assert not self.pixels
            self.welford = utils.Welford()

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        del d['obses'], d['next_obses'], d['actions'], d['rewards'], \
          d['not_dones']
        return d

    def __setstate__(self, d):
        self.__dict__ = d

        # Manually need to re-load the transitions with load()
        self.empty_data()


    def empty_data(self):
        obs_dtype = np.float32 if not self.pixels else np.uint8
        obs_shape = self.obs_shape
        action_shape = self.action_shape
        capacity = self.capacity

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.payload = []
        self.done_idxs = None


    def __len__(self):
        return self.capacity if self.full else self.idx

    def get_obs_stats(self):
        assert not self.pixels
        MIN_STD = 1e-1
        MAX_STD = 10
        mean = self.welford.mean()
        std = self.welford.std()
        std[std < MIN_STD] = MIN_STD
        std[std > MAX_STD] = MAX_STD
        return mean, std

    def add(self, obs, action, reward, next_obs, done):
        if self.normalize_obs:
            self.welford.add_data(obs)

        if done:
            self.done_idxs.add(self.idx)
        elif self.full:
            self.done_idxs.discard(self.idx)

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.global_idx += 1
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx,
            size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        if self.normalize_obs:
            mu, sigma = self.get_obs_stats()
            obses = (obses-mu)/sigma
            next_obses = (next_obses-mu)/sigma

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones


    def sample_multistep(self):
        assert self.batch_size < self.idx or self.full

        T = self.horizon

        last_idx = self.capacity if self.full else self.idx
        last_idx -= T

        done_idxs_sorted = np.array(list(self.done_idxs) + [last_idx])
        
        n_done = len(done_idxs_sorted)
        done_idxs_raw = done_idxs_sorted - np.arange(1, n_done+1)*T

        samples_raw = npr.choice(
            last_idx-(T+1)*n_done, size=self.batch_size,
            replace=True # for speed
        )
        samples_raw = sorted(samples_raw)
        js = np.searchsorted(done_idxs_raw, samples_raw)
        offsets = done_idxs_raw[js] - samples_raw + T
        start_idxs = done_idxs_sorted[js] - offsets
    
        obses, actions, rewards, not_dones = [], [], [], []

        for t in range(T):
            obses.append(self.obses[start_idxs + t])
            actions.append(self.actions[start_idxs + t])
            rewards.append(self.rewards[start_idxs + t])
            not_dones.append(self.not_dones[start_idxs + t])
            assert np.all(self.not_dones[start_idxs + t])

        obses = np.stack(obses)
        actions = np.stack(actions)
        rewards = np.stack(rewards).squeeze(2)
        not_dones = np.stack(not_dones).squeeze(2)

        if self.normalize_obs:
            mu, sigma = self.get_obs_stats()
            obses = (obses-mu)/sigma

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        not_dones = torch.as_tensor(not_dones, device=self.device)

        return obses, actions, rewards, not_dones

    def save(self, save_dir):
        # TODO: The serialization code and logic can be significantly improved.

        if self.global_idx == self.global_last_save:
            return
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path = os.path.join(
            save_dir, f'{self.global_last_save:08d}_{self.global_idx:08d}.pt')

        payload = list(zip(*self.payload))
        payload = [np.vstack(x) for x in payload]
        self.global_last_save = self.global_idx
        torch.save(payload, path)
        self.payload = []


    def load(self, save_dir):
        def parse_chunk(chunk):
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            return (start, end)


        self.idx = 0

        chunks = os.listdir(save_dir)
        chunks = filter(lambda fname: 'stats' not in fname, chunks)
        chunks = sorted(chunks, key=lambda x: int(x.split('_')[0]))

        self.full = self.global_idx > self.capacity
        global_beginning = self.global_idx - self.capacity if self.full else 0

        for chunk in chunks:
            global_start, global_end = parse_chunk(chunk)
            if global_start >= self.global_idx:
                continue
            start = global_start - global_beginning
            end = global_end - global_beginning
            if end <= 0:
                continue

            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            if start < 0:
                payload = [x[-start:] for x in payload]
                start = 0
            assert self.idx == start

            obses = payload[0]
            next_obses = payload[1]

            self.obses[start:end] = obses
            self.next_obses[start:end] = next_obses
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

        self.last_save = self.idx

        if self.full:
            assert self.idx == self.capacity
            self.idx = 0

        last_idx = self.capacity if self.full else self.idx
        self.done_idxs = SortedSet(np.where(1.-self.not_dones[:last_idx])[0])

class ReplayBufferPixelMultistep(object):
    """Buffer to store environment transitions."""
    def __init__(self, 
                obs_shape, 
                action_shape, 
                capacity, 
                batch_size, 
                horizon, 
                device, 
                normalize_obs=False):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.capacity = capacity
        self.batch_size = batch_size
        self.horizon = horizon
        self.device = device

        self.pixels = len(obs_shape) > 1
        self.empty_data()

        self.done_idxs = SortedSet()
        self.global_idx = 0
        self.global_last_save = 0

        self.normalize_obs = normalize_obs

        if normalize_obs:
            assert not self.pixels
            self.welford = utils.Welford()

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        del d['obses'], d['next_obses'], d['actions'], d['rewards'], \
          d['not_dones']
        return d

    def __setstate__(self, d):
        self.__dict__ = d

        # Manually need to re-load the transitions with load()
        self.empty_data()


    def empty_data(self):
        obs_dtype = np.float32 if not self.pixels else np.uint8
        obs_shape = self.obs_shape
        action_shape = self.action_shape
        capacity = self.capacity

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.payload = []
        self.done_idxs = None


    def __len__(self):
        return self.capacity if self.full else self.idx

    def get_obs_stats(self):
        assert not self.pixels
        MIN_STD = 1e-1
        MAX_STD = 10
        mean = self.welford.mean()
        std = self.welford.std()
        std[std < MIN_STD] = MIN_STD
        std[std > MAX_STD] = MAX_STD
        return mean, std

    def add(self, obs, action, reward, next_obs, done):
        if self.normalize_obs:
            self.welford.add_data(obs)

        if done:
            self.done_idxs.add(self.idx)
        elif self.full:
            self.done_idxs.discard(self.idx)

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.global_idx += 1
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx,
            size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        if self.normalize_obs:
            mu, sigma = self.get_obs_stats()
            obses = (obses-mu)/sigma
            next_obses = (next_obses-mu)/sigma

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones

    def sample_multistep(self, augs_funcs=None):
        assert self.batch_size < self.idx or self.full

        T = self.horizon

        last_idx = self.capacity if self.full else self.idx
        last_idx -= T

        done_idxs_sorted = np.array(list(self.done_idxs) + [last_idx])
        
        n_done = len(done_idxs_sorted)
        done_idxs_raw = done_idxs_sorted - np.arange(1, n_done+1)*T

        samples_raw = npr.choice(
            last_idx-(T+1)*n_done, size=self.batch_size,
            replace=True # for speed
        )
        samples_raw = sorted(samples_raw)
        js = np.searchsorted(done_idxs_raw, samples_raw)
        offsets = done_idxs_raw[js] - samples_raw + T
        start_idxs = done_idxs_sorted[js] - offsets
    
        obses, actions, rewards, not_dones = [], [], [], []

        for t in range(T):
            obs = self.obses[start_idxs + t]
            if augs_funcs:
                for aug, func in augs_funcs.items():
                    # apply crop and cutout first
                    if 'crop' in aug or 'cutout' in aug:
                        obs = func(obs)
                        # obses.append(obs)
                        # next_obs = func(next_obs)
            obses.append(obs)
            actions.append(self.actions[start_idxs + t])
            rewards.append(self.rewards[start_idxs + t])
            not_dones.append(self.not_dones[start_idxs + t])
            assert np.all(self.not_dones[start_idxs + t])

        obses = np.stack(obses)
        actions = np.stack(actions)
        rewards = np.stack(rewards).squeeze(2)
        not_dones = np.stack(not_dones).squeeze(2)

        if self.normalize_obs:
            mu, sigma = self.get_obs_stats()
            obses = (obses-mu)/sigma

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        not_dones = torch.as_tensor(not_dones, device=self.device)

        return obses, actions, rewards, not_dones

    def save(self, save_dir):
        # TODO: The serialization code and logic can be significantly improved.

        if self.global_idx == self.global_last_save:
            return
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path = os.path.join(
            save_dir, f'{self.global_last_save:08d}_{self.global_idx:08d}.pt')

        payload = list(zip(*self.payload))
        payload = [np.vstack(x) for x in payload]
        self.global_last_save = self.global_idx
        torch.save(payload, path)
        self.payload = []


    def load(self, save_dir):
        def parse_chunk(chunk):
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            return (start, end)


        self.idx = 0

        chunks = os.listdir(save_dir)
        chunks = filter(lambda fname: 'stats' not in fname, chunks)
        chunks = sorted(chunks, key=lambda x: int(x.split('_')[0]))

        self.full = self.global_idx > self.capacity
        global_beginning = self.global_idx - self.capacity if self.full else 0

        for chunk in chunks:
            global_start, global_end = parse_chunk(chunk)
            if global_start >= self.global_idx:
                continue
            start = global_start - global_beginning
            end = global_end - global_beginning
            if end <= 0:
                continue

            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            if start < 0:
                payload = [x[-start:] for x in payload]
                start = 0
            assert self.idx == start

            obses = payload[0]
            next_obses = payload[1]

            self.obses[start:end] = obses
            self.next_obses[start:end] = next_obses
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

        self.last_save = self.idx

        if self.full:
            assert self.idx == self.capacity
            self.idx = 0

        last_idx = self.capacity if self.full else self.idx
        self.done_idxs = SortedSet(np.where(1.-self.not_dones[:last_idx])[0])
