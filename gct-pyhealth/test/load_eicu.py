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
    tables=["diagnosis", "treatment", "admissionDx", "physicalExam", "medication"],
    dev=True,
    refresh_cache=False,
)

dataset_const = eICUDataset(
    root='../../eicu_csv',
    tables=["diagnosis", "treatment", "admissionDx", "physicalExam", "medication"],
    refresh_cache=False,
    dev=True
)


# Dropping patient with less than 24 hours duration minute
# should be stated in the data entry 'unitdischargeoffset'
# aka visit.discharge_time - visit.encounter_time
def process_patient(ds, hour_threshold=24):
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


# Processed 200859 encounters, deleted 67959 encounters
dataset_processed = process_patient(dataset)
print(dataset_processed.info())

# set the prediction task
dataset_mortality = dataset_const.set_task(mortality_prediction_eicu_fn)
print(dataset_mortality.available_keys)
print(dataset_mortality.samples[0])
