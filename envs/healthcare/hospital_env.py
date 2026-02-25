# envs/healthcare/hospital_env.py
from __future__ import annotations

import functools
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

from envs.healthcare.hospital import Hospital
from envs.healthcare.patients import Acuity, Patient, PatientGenerator

NUM_ACTIONS = 7
MAX_QUEUE_SIZE = 20


class HealthcareEnv(ParallelEnv):
    metadata = {"name": "healthcare_v0"}

    def __init__(
        self,
        num_hospitals: int = 3,
        beds_per_hospital: dict[str, int] | None = None,
        num_objectives: int = 3,
        patient_arrival_rate: float = 2.0,
        acuity_distribution: list[float] | None = None,
        episode_length: int = 168,
    ):
        if beds_per_hospital is None:
            beds_per_hospital = {"general": 50, "icu": 10, "emergency": 5}
        if acuity_distribution is None:
            acuity_distribution = [0.6, 0.3, 0.1]

        self.num_hospitals = num_hospitals
        self.beds_config = beds_per_hospital
        self.num_objectives = num_objectives
        self.patient_arrival_rate = patient_arrival_rate
        self.acuity_distribution = acuity_distribution
        self.episode_length = episode_length

        self.possible_agents = [f"hospital_{i}" for i in range(num_hospitals)]
        self.agents = []

        self._hospitals: dict[str, Hospital] = {}
        self._queues: dict[str, list[Patient]] = {}
        self._generators: dict[str, PatientGenerator] = {}
        self._step_count = 0
        self._rng: np.random.Generator | None = None
        self._patients_treated: dict[str, int] = {}
        self._patients_waiting_too_long: dict[str, int] = {}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> Box:
        # 6 (bed occ+cap for 3 types) + 1 (queue_size) + 3 (acuity counts)
        # + 1 (time fraction) + (num_hospitals-1) neighbour occupancies
        size = 6 + 1 + 3 + 1 + (self.num_hospitals - 1)
        return Box(low=0.0, high=np.float32(np.inf), shape=(size,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> Discrete:
        return Discrete(NUM_ACTIONS)

    def reset(self, seed: int | None = None, options: dict | None = None):
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self.agents = self.possible_agents[:]

        for a in self.agents:
            self._hospitals[a] = Hospital(beds=dict(self.beds_config))
            self._queues[a] = []
            self._generators[a] = PatientGenerator(
                arrival_rate=self.patient_arrival_rate,
                acuity_distribution=self.acuity_distribution,
                rng=np.random.default_rng(self._rng.integers(0, 2**32)),
            )
            self._patients_treated[a] = 0
            self._patients_waiting_too_long[a] = 0

        # Generate initial patients
        for a in self.agents:
            self._queues[a] = self._generators[a].generate(timestep=0)

        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: self._get_info(a, np.zeros(self.num_objectives)) for a in self.agents}
        return obs, infos

    def step(self, actions: dict[str, int]):
        self._step_count += 1

        step_treated = {a: 0 for a in self.agents}
        step_expired = {a: 0 for a in self.agents}

        for agent, action in actions.items():
            hospital = self._hospitals[agent]
            queue = self._queues[agent]

            if action == 6 or not queue:
                # Hold / no-op
                pass
            elif action == 0:
                # Accept to preferred bed type
                patient = queue[0]
                if hospital.admit(patient, patient.preferred_bed_type):
                    queue.pop(0)
                    step_treated[agent] += 1
            elif action in (1, 2, 3):
                # Accept to specific bed type
                bed_types = {1: "general", 2: "icu", 3: "emergency"}
                patient = queue[0]
                if hospital.admit(patient, bed_types[action]):
                    queue.pop(0)
                    step_treated[agent] += 1
            elif action == 4:
                # Divert to neighbour
                if queue:
                    patient = queue.pop(0)
                    neighbours = [a for a in self.agents if a != agent]
                    if neighbours:
                        target = neighbours[self._step_count % len(neighbours)]
                        self._queues[target].append(patient)
            elif action == 5:
                # Discharge earliest stable patient
                all_patients = hospital.all_patients()
                stable = [p for p in all_patients if p.acuity == Acuity.LOW]
                if stable:
                    stable[0].length_of_stay = 0

        # Tick all hospitals (discharge patients whose LOS is met)
        for a in self.agents:
            self._hospitals[a].tick(current_time=self._step_count)

        # Process queue expirations
        for a in self.agents:
            remaining = []
            expired_count = 0
            for p in self._queues[a]:
                wait = self._step_count - p.arrival_time
                if wait >= p.max_wait_time:
                    expired_count += 1
                else:
                    remaining.append(p)
            step_expired[a] = expired_count
            self._queues[a] = remaining
            self._patients_waiting_too_long[a] += expired_count
            self._patients_treated[a] += step_treated[a]

        # Generate new patients
        for a in self.agents:
            new_patients = self._generators[a].generate(timestep=self._step_count)
            self._queues[a].extend(new_patients)

        # Compute rewards
        vec_rewards = {}
        for a in self.agents:
            vec_rewards[a] = self._compute_reward(a, step_treated[a], step_expired[a])

        team_reward = np.sum(list(vec_rewards.values()), axis=0)

        done = self._step_count >= self.episode_length

        obs = {a: self._get_obs(a) for a in self.agents}
        rewards = {a: float(np.sum(vec_rewards[a])) for a in self.agents}
        terminations = {a: done for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: self._get_info(a, vec_rewards[a], team_reward) for a in self.agents}

        if done:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def _compute_reward(self, agent: str, treated: int, expired: int) -> np.ndarray:
        hospital = self._hospitals[agent]

        total_processed = treated + expired
        mortality_reduction = treated / max(total_processed, 1)

        queue_sizes = [len(self._queues[a]) for a in self.possible_agents if a in self._hospitals]
        if len(queue_sizes) > 1 and max(queue_sizes) > 0:
            mean_q = np.mean(queue_sizes)
            std_q = np.std(queue_sizes)
            equity = 1.0 - (std_q / max(mean_q, 1.0))
        else:
            equity = 1.0

        total_cap = hospital.total_capacity()
        total_occ = hospital.total_occupancy()
        utilization = total_occ / max(total_cap, 1)
        cost_efficiency = 1.0 - abs(utilization - 0.85) / 0.85

        return np.array([mortality_reduction, equity, cost_efficiency], dtype=np.float64)

    def _get_obs(self, agent: str) -> np.ndarray:
        hospital = self._hospitals[agent]
        queue = self._queues[agent]

        obs_parts: list[float] = []
        for bt in ["general", "icu", "emergency"]:
            obs_parts.extend([float(hospital.occupancy(bt)), float(hospital.capacity(bt))])

        obs_parts.append(float(min(len(queue), MAX_QUEUE_SIZE)))

        acuity_counts = [0.0, 0.0, 0.0]
        for p in queue[:MAX_QUEUE_SIZE]:
            acuity_counts[p.acuity.value] += 1.0
        obs_parts.extend(acuity_counts)

        obs_parts.append(float(self._step_count) / max(self.episode_length, 1))

        for other in self.possible_agents:
            if other != agent:
                if other in self._hospitals:
                    h = self._hospitals[other]
                    obs_parts.append(float(h.total_occupancy()) / max(h.total_capacity(), 1))
                else:
                    obs_parts.append(0.0)

        return np.array(obs_parts, dtype=np.float32)

    def _get_info(
        self,
        agent: str,
        vec_reward: np.ndarray,
        team_reward: np.ndarray | None = None,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {"vec_reward": vec_reward}
        if team_reward is not None:
            info["team_vec_reward"] = team_reward
        else:
            info["team_vec_reward"] = vec_reward.copy()
        return info
