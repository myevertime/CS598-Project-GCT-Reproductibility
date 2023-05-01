import numpy as np
import pickle
import os

from pyhealth.data import Patient, Visit


def count_conditional_prob_dp(eicu_samples_dict, output_path, train_key_set=None):
    """
    This is a Python function called count_conditional_prob_dp that takes in a list of encoded features,
    an output path to save the conditional probabilities, and a training key set, and calculates the empirical
    conditional probabilities for diagnosis-procedure (DP) and procedure-diagnosis (PD) pairs.
    """
    dx_freqs = {}  # diagnosis
    proc_freqs = {}  # treatment
    dp_freqs = {}  # diagnosis-treatment

    total_visit = 0
    for sample in eicu_samples_dict:
        key = sample['visit_id']
        if train_key_set is not None and key not in train_key_set:
            total_visit += 1
            continue
        conditions = sample['conditions']
        procedures = sample['procedures']
        for dx in conditions:
            if dx not in dx_freqs:
                dx_freqs[dx] = 0
            dx_freqs[dx] += 1
        for proc in procedures:
            if proc not in proc_freqs:
                proc_freqs[proc] = 0
            proc_freqs[proc] += 1
        for dx in conditions:
            for proc in procedures:
                dp = dx + ',' + proc
                if dp not in dp_freqs:
                    dp_freqs[dp] = 0
                dp_freqs[dp] += 1
        total_visit += 1

    dx_probs = dict([(k, v / float(total_visit)) for k, v in dx_freqs.items()])
    proc_probs = dict([(k, v / float(total_visit)) for k, v in proc_freqs.items()])
    dp_probs = dict([(k, v / float(total_visit)) for k, v in dp_freqs.items()])

    dp_cond_probs = {}
    pd_cond_probs = {}
    for dx, dx_prob in dx_probs.items():
        for proc, proc_prob in proc_probs.items():
            dp = dx + ',' + proc
            pd = proc + ',' + dx
            if dp in dp_probs:
                dp_cond_probs[dp] = dp_probs[dp] / dx_prob
                pd_cond_probs[pd] = dp_probs[dp] / proc_prob
            else:
                dp_cond_probs[dp] = 0.0
                pd_cond_probs[pd] = 0.0
    # originally supposed to pickle. but for now just return the 2 cond prob dicts that are used
    # return dp_cond_probs, pd_cond_probs
    pickle.dump(dp_cond_probs, open(os.path.join(output_path, 'dp_cond_probs.empirical.p'), 'wb'))
    pickle.dump(pd_cond_probs, open(os.path.join(output_path, 'pd_cond_probs.empirical.p'), 'wb'))


def mortality_prediction_eicu_fn_gct(patient: Patient):
    """Processes a single patient for the mortality prediction task.

    Mortality prediction aims at predicting whether the patient will decease in the
    next hospital visit based on the clinical information from current visit
    (e.g., conditions and procedures).

    Similar to mortality_prediction_eicu_fn, but with different code mapping:
    - using admissionDx and diagnosisString table as condition codes
    - using treatment table as procedure codes

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id,
            visit_id, and other task-specific attributes as key

    Note that we define the task as a binary classification task.
    """
    samples = []

    # dropping short encounter
    drop_hour_threshold = 24
    dropped_long_encounter_counts = 0

    # dropping encounter with too much codes or no codes
    min_num_codes = 1
    max_num_codes = 50
    dropped_codes_overlimit_counts = 0

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        if next_visit.discharge_status not in ["Alive", "Expired"]:
            mortality_label = 0
        else:
            mortality_label = 0 if next_visit.discharge_status == "Alive" else 1

        # extract the code list
        admissionDx = [dx.lower() for dx in visit.get_code_list(table="admissionDx")]
        diagnosisString = [cond.lower() for cond in visit.get_code_list(table="diagnosisString")]
        conditions = admissionDx + diagnosisString
        procedures = [treat.lower() for treat in visit.get_code_list(table="treatment")]

        conditions = sorted(list(set(conditions)))
        procedures = sorted(list(set(procedures)))

        # exclude: visits without treatment and admissionDx/diagnosisString
        if len(conditions) < min_num_codes \
                or len(conditions) > max_num_codes \
                or len(procedures) < min_num_codes \
                or len(procedures) > max_num_codes:
            dropped_codes_overlimit_counts += 1
            continue

        # dropping patient with more than 24 hours duration minute
        if (visit.discharge_time - visit.encounter_time) > np.timedelta64(drop_hour_threshold, 'h'):
            dropped_long_encounter_counts += 1
            continue

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "conditions_enc": [],
                "conditions_mask": [],
                "procedures": procedures,
                "procedures_enc": [],
                "procedures_mask": [],
                # "prior_indices": [],
                # "prior_values": [],
                "label": mortality_label,
            }
        )

    # no cohort selection
    return samples

# TODO
# def readmission_prediction_eicu_fn_gct(patient: Patient, time_window=5):
#     """Processes a single patient for the readmission prediction task.
#
#     Readmission prediction aims at predicting whether the patient will be readmitted
#     into hospital within time_window days based on the clinical information from
#     current visit (e.g., conditions and procedures).
#
#     Similar to readmission_prediction_eicu_fn, but with different code mapping:
#     - using admissionDx and diagnosisString table as condition codes
#     - using treatment table as procedure codes
#
#     Args:
#         patient: a Patient object
#         time_window: the time window threshold (gap < time_window means label=1 for
#             the task)
#
#     Returns:
#         samples: a list of samples, each sample is a dict with patient_id, visit_id,
#             and other task-specific attributes as key
#
#     Note that we define the task as a binary classification task.
#     """
#
#     samples = []
#     # we will drop the last visit
#     for i in range(len(patient) - 1):
#         visit: Visit = patient[i]
#         next_visit: Visit = patient[i + 1]
#         # get time difference between current visit and next visit
#         time_diff = (next_visit.encounter_time - visit.encounter_time).days
#         readmission_label = 1 if time_diff < time_window else 0
#
#         admissionDx = visit.get_code_list(table="admissionDx")
#         diagnosisString = visit.get_code_list(table="diagnosisString")
#         treatment = visit.get_code_list(table="treatment")
#
#         # exclude: visits without treatment, admissionDx, diagnosisString
#         if len(admissionDx) * len(diagnosisString) * len(treatment) == 0:
#             continue
#
#         # TODO: dropping patient with more than 24 hours duration minute
#
#         samples.append(
#             {
#                 "visit_id": visit.visit_id,
#                 "patient_id": patient.patient_id,
#                 "conditions": [admissionDx] + [diagnosisString],
#                 "procedures": [treatment],
#                 "label": readmission_label,
#             }
#         )
#     # no cohort selection
#     return samples


if __name__ == "__main__":
    from prediction_tasks import mortality_prediction_eicu_fn_gct
    from pyhealth.datasets.sample_dataset import SampleEHRDataset
    from tqdm import tqdm

    from pyhealth.datasets import eICUDataset

    print('Loading eICU dataset')
    eicu_ds = eICUDataset(
        root='../../eicu_csv',
        tables=["admissionDx", "diagnosisString", "treatment"],
        refresh_cache=False,
        # dev=True
    )

    eicu_ds.stat()
    eicu_ds.info()

    # under the hook of:
    # eicu_ds_mortality = eicu_ds.set_task(task_fn=mortality_prediction_eicu_fn_gct)
    task_name = "mortality_prediction_eicu_fn_gct"
    samples_list = []

    # diagnosis and treatment codes mapping list
    cond_enc_map = {}
    proc_enc_map = {}

    min_num_codes = 1
    max_num_codes = 50

    # add labeling
    for patient_id, patient in tqdm(eicu_ds.patients.items(),
                                    desc=f"Generating samples for {task_name}"):
        samples = mortality_prediction_eicu_fn_gct(patient)

        # mapping the string to int
        for sample in samples:
            conditions = sample['conditions']
            procedures = sample['procedures']
            for dx_id in conditions:
                if dx_id not in cond_enc_map:
                    cond_enc_map[dx_id] = len(cond_enc_map)
            for treat_id in procedures:
                if treat_id not in proc_enc_map:
                    proc_enc_map[treat_id] = len(proc_enc_map)

            sample['conditions_enc'].extend([cond_enc_map[item] for item in sample['conditions']])
            sample['procedures_enc'].extend([proc_enc_map[item] for item in sample['procedures']])

        if len(samples) != 0:
            samples_list.extend(samples)


    # TODO: update the mask with paddings
    for sample in samples_list:
        # add padding for conditions
        dx_padding_idx = len(cond_enc_map)
        if len(sample['conditions_enc']) < max_num_codes:
            sample['conditions_enc'].extend([dx_padding_idx] * (max_num_codes - len(sample['conditions_enc'])))
        sample['conditions_mask'] = [0 if i == dx_padding_idx else 1 for i in sample['conditions_enc']]

        # add padding for procedures
        proc_padding_idx = len(proc_enc_map)
        if len(sample['procedures_enc']) < max_num_codes:
            sample['procedures_enc'].extend([proc_padding_idx] * (max_num_codes - len(sample['procedures_enc'])))
        sample['procedures_mask'] = [0 if i == proc_padding_idx else 1 for i in sample['procedures_enc']]

    # TODO: count conditional probability

    # create a new dataset with task function
    eicu_ds_mortality = SampleEHRDataset(
        samples_list,
        dataset_name=eicu_ds.dataset_name,
        task_name=task_name,
    )

    # stats info
    eicu_ds_mortality.stat()