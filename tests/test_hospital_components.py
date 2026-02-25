# tests/test_hospital_components.py
import numpy as np
import pytest
from envs.healthcare.patients import Patient, Acuity, PatientGenerator
from envs.healthcare.hospital import Hospital


class TestPatientGenerator:
    def test_generates_patients(self):
        rng = np.random.default_rng(42)
        gen = PatientGenerator(
            arrival_rate=5.0,
            acuity_distribution=[0.6, 0.3, 0.1],
            rng=rng,
        )
        patients = gen.generate(timestep=0)
        assert isinstance(patients, list)
        assert all(isinstance(p, Patient) for p in patients)

    def test_acuity_distribution_over_many(self):
        rng = np.random.default_rng(42)
        gen = PatientGenerator(arrival_rate=100.0, acuity_distribution=[0.6, 0.3, 0.1], rng=rng)
        patients = []
        for t in range(100):
            patients.extend(gen.generate(timestep=t))
        acuities = [p.acuity for p in patients]
        low_frac = sum(1 for a in acuities if a == Acuity.LOW) / len(acuities)
        assert 0.5 < low_frac < 0.7

    def test_patient_has_time_sensitivity(self):
        rng = np.random.default_rng(42)
        gen = PatientGenerator(arrival_rate=10.0, acuity_distribution=[0.6, 0.3, 0.1], rng=rng)
        patients = gen.generate(timestep=0)
        assert len(patients) > 0
        for p in patients:
            assert p.max_wait_time > 0
            assert p.length_of_stay > 0


class TestHospital:
    def test_admit_patient(self):
        h = Hospital(beds={"general": 5, "icu": 2, "emergency": 1})
        p = Patient(acuity=Acuity.ACUTE, arrival_time=0, max_wait_time=5, length_of_stay=10)
        assert h.admit(p, bed_type="general") is True
        assert h.occupancy("general") == 1

    def test_admit_full(self):
        h = Hospital(beds={"general": 1, "icu": 0, "emergency": 0})
        p1 = Patient(acuity=Acuity.ACUTE, arrival_time=0, max_wait_time=5, length_of_stay=10)
        p2 = Patient(acuity=Acuity.ACUTE, arrival_time=0, max_wait_time=5, length_of_stay=10)
        assert h.admit(p1, bed_type="general") is True
        assert h.admit(p2, bed_type="general") is False

    def test_discharge_after_los(self):
        h = Hospital(beds={"general": 5, "icu": 2, "emergency": 1})
        p = Patient(acuity=Acuity.LOW, arrival_time=0, max_wait_time=10, length_of_stay=3)
        h.admit(p, bed_type="general")
        discharged = h.tick(current_time=1)
        assert len(discharged) == 0
        discharged = h.tick(current_time=3)
        assert len(discharged) == 1

    def test_capacity(self):
        h = Hospital(beds={"general": 5, "icu": 2, "emergency": 1})
        assert h.capacity("general") == 5
        assert h.available("general") == 5

    def test_total_occupancy(self):
        h = Hospital(beds={"general": 5, "icu": 2, "emergency": 1})
        p = Patient(acuity=Acuity.CRITICAL, arrival_time=0, max_wait_time=2, length_of_stay=5)
        h.admit(p, bed_type="icu")
        assert h.total_occupancy() == 1
        assert h.total_capacity() == 8
