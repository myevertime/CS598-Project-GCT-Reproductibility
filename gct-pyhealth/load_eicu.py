import csv
import os
import pickle
import sys
import time

import sklearn.model_selection as ms
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
import numpy as np

import pyhealth
from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import eICUDataset
from pyhealth.tasks import readmission_prediction_eicu_fn
from pyhealth.tasks import mortality_prediction_eicu_fn


# Dropping patient with less than 24 hours duration minute
# should be stated in the data entry 'unitdischargeoffset'
# aka visit.discharge_time - visit.encounter_time
def remove_short_admission(ds, hour_threshold=24):
    dataset_processed = ds
    encounter_processed_count = 0
    encounter_deleted_count = 0

    for patient_id, patient in ds.patients.items():
        visits = patient.visits.copy()
        for visit_id, visit in visits.items():
            encounter_processed_count += 1
            if (visit.discharge_time - visit.encounter_time) < np.timedelta64(hour_threshold, 'h'):
                # print("Dropping patient {} visit {} due to less than {} hours duration".format(patient_id, visit_id, hour_threshold))
                encounter_deleted_count += 1
                del dataset_processed.patients[patient_id].visits[visit_id]

    print("Processed {} encounters, deleted {} encounters".format(encounter_processed_count, encounter_deleted_count))
    return dataset_processed


def readmission_prediction_eicu_fn_basic(patient: Patient, time_window=5):
    """Processes a single patient for the readmission prediction task.

    Readmission prediction aims at predicting whether the patient will be readmitted
    into hospital within time_window days based on the clinical information from
    current visit (e.g., conditions and procedures).

    Args:
        patient: a Patient object
        time_window: the time window threshold (gap < time_window means label=1 for
            the task)

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key

    Note that we define the task as a binary classification task.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> eicu_base = eICUDataset(
        ...     root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        ...     tables=["diagnosis", "medication"],
        ...     code_mapping={},
        ...     dev=True
        ... )
        >>> from pyhealth.tasks import readmission_prediction_eicu_fn
        >>> eicu_sample = eicu_base.set_task(readmission_prediction_eicu_fn)
        >>> eicu_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '98', '663', '58', '51']], 'procedures': [['1']], 'label': 1}]
    """
    samples = []
    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        # get time difference between current visit and next visit
        time_diff = (next_visit.encounter_time - visit.encounter_time).days
        readmission_label = 1 if time_diff < time_window else 0
        #
        # conditions = visit.get_code_list(table="diagnosis")
        # procedures = visit.get_code_list(table="physicalExam")
        # drugs = visit.get_code_list(table="medication")
        # treatment = visit.get_code_list(table="treatment")
        # admissionDx = visit.get_code_list(table="admissionDx")
        # conditions_str = visit.get_code_list(table="diagnosisString")

        # exclude: visits without any codes/events
        if visit.num_events == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                # "conditions": [conditions],
                # "procedures": [procedures],
                # "drugs": [drugs],
                # "treatment": [treatment],
                # "admissionDx": [admissionDx],
                # "conditions_str": [conditions_str],
                "label": readmission_label,
            }
        )
    # no cohort selection
    return samples


if __name__ == "__main__":
    np.random.seed(1234)

    """
    The following csv files are required
        patient.csv
        diagnosis.csv
        treatment.csv
        lab.csv
        medication.csv
        physicalExam.csv
        admissionDx.csv
    """
    dataset = eICUDataset(
        root='../../eicu_csv',
        tables=["diagnosis", "treatment", "admissionDx", "physicalExam", "medication", "lab"],
        refresh_cache=False,
        dev=True,
    )

    dataset_const = eICUDataset(
        root='../../eicu_csv',
        tables=["diagnosis", "treatment", "admissionDx", "physicalExam", "medication", "lab"],
        refresh_cache=False,
        dev=True
    )

    # Processed 200859 encounters, deleted 67959 encounters
    dataset_processed = remove_short_admission(dataset)
    print(dataset_processed.info())

    # set the prediction task
    dataset_mortality = dataset_const.set_task(mortality_prediction_eicu_fn)
    print(dataset_mortality.available_keys)
    print(dataset_mortality.samples[0])
