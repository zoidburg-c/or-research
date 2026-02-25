# envs/healthcare/hospital.py
from __future__ import annotations

from envs.healthcare.patients import Patient


class Hospital:
    def __init__(self, beds: dict[str, int]):
        self._capacity = dict(beds)
        self._patients: dict[str, list[Patient]] = {bt: [] for bt in beds}

    def capacity(self, bed_type: str) -> int:
        return self._capacity[bed_type]

    def occupancy(self, bed_type: str) -> int:
        return len(self._patients[bed_type])

    def available(self, bed_type: str) -> int:
        return self._capacity[bed_type] - self.occupancy(bed_type)

    def total_capacity(self) -> int:
        return sum(self._capacity.values())

    def total_occupancy(self) -> int:
        return sum(len(ps) for ps in self._patients.values())

    def admit(self, patient: Patient, bed_type: str) -> bool:
        if self.available(bed_type) <= 0:
            return False
        patient.admitted_time = patient.arrival_time
        self._patients[bed_type].append(patient)
        return True

    def tick(self, current_time: int) -> list[Patient]:
        discharged = []
        for bed_type in self._patients:
            remaining = []
            for p in self._patients[bed_type]:
                if p.admitted_time is not None and (current_time - p.admitted_time) >= p.length_of_stay:
                    discharged.append(p)
                else:
                    remaining.append(p)
            self._patients[bed_type] = remaining
        return discharged

    def get_patients(self, bed_type: str) -> list[Patient]:
        return list(self._patients[bed_type])

    def all_patients(self) -> list[Patient]:
        result = []
        for ps in self._patients.values():
            result.extend(ps)
        return result
