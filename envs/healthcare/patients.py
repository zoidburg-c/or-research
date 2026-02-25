# envs/healthcare/patients.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import itertools

import numpy as np


class Acuity(Enum):
    LOW = 0
    ACUTE = 1
    CRITICAL = 2


_ACUITY_PARAMS = {
    Acuity.LOW:      {"max_wait": (8, 24), "los": (4, 24)},
    Acuity.ACUTE:    {"max_wait": (2, 8),  "los": (12, 72)},
    Acuity.CRITICAL: {"max_wait": (1, 3),  "los": (24, 168)},
}

_patient_counter = itertools.count()


@dataclass
class Patient:
    acuity: Acuity
    arrival_time: int
    max_wait_time: int
    length_of_stay: int
    patient_id: int = field(default_factory=lambda: next(_patient_counter))
    admitted_time: int | None = None

    @property
    def preferred_bed_type(self) -> str:
        if self.acuity == Acuity.CRITICAL:
            return "icu"
        return "general"


class PatientGenerator:
    def __init__(
        self,
        arrival_rate: float,
        acuity_distribution: list[float],
        rng: np.random.Generator,
    ):
        self.arrival_rate = arrival_rate
        self.acuity_probs = np.array(acuity_distribution, dtype=np.float64)
        self.acuity_probs /= self.acuity_probs.sum()
        self._rng = rng

    def generate(self, timestep: int) -> list[Patient]:
        n = self._rng.poisson(self.arrival_rate)
        patients = []
        for _ in range(n):
            acuity_idx = self._rng.choice(len(Acuity), p=self.acuity_probs)
            acuity = list(Acuity)[acuity_idx]
            params = _ACUITY_PARAMS[acuity]
            max_wait = int(self._rng.integers(params["max_wait"][0], params["max_wait"][1] + 1))
            los = int(self._rng.integers(params["los"][0], params["los"][1] + 1))
            patients.append(
                Patient(
                    acuity=acuity,
                    arrival_time=timestep,
                    max_wait_time=max_wait,
                    length_of_stay=los,
                )
            )
        return patients
