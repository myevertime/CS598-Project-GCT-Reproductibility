import pyhealth
from pyhealth.data import Event, Visit, Patient

import numpy as np
np.random.seed(1234)

from pyhealth.datasets import eICUDataset
dataset = eICUDataset(
    root='../eicu_csv',
    tables=["diagnosis", "treatment", "admissionDx"]
)

dataset.stat()
dataset.info()