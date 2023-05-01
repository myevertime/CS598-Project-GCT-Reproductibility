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


class EncounterInfo:
    def __init__(self, patient_id, visit_id, encounter_timestamp,
                 label_expired, label_readmission):
        # information written during initialization
        self.patient_id = patient_id
        self.visit_id = visit_id
        self.encounter_timestamp = encounter_timestamp
        self.label_expired = label_expired
        self.label_readmission = label_readmission

        # update afterwards
        self.conditions = []
        self.procedures = []


class EncounterFeatures:
    def __init__(self, patient_id, visit_id,
                 label_expired, label_readmission,
                 conditions, conditions_hash,
                 procedures, procedures_hash):
        # information written during initialization
        self.patient_id = patient_id
        self.visit_id = visit_id
        self.encounter_key = patient_id + ':' + visit_id

        self.label_expired = label_expired
        self.label_readmission = label_readmission
        self.conditions = conditions
        self.conditions_hash = conditions_hash
        self.procedures = procedures
        self.procedures_hash = procedures_hash

        # update afterwards
        self.prior_indices = None
        self.prior_values = None
        self.conditions_mask = None
        self.procedures_mask = None

    def getKeyValueDict(self, task='mortality'):

        label = None
        if task == 'mortality':
            label = self.label_expired
        elif task == 'readmission':
            label = self.label_readmission
        else:
            raise ValueError('task should be either mortality or readmission')

        return {
            "visit_id": self.visit_id,
            "patient_id": self.patient_id,
            "conditions": self.conditions,
            "conditions_hash": self.conditions_hash,
            "conditions_mask": self.conditions_mask,
            "procedures": self.procedures,
            "procedures_hash": self.procedures_hash,
            "procedures_mask": self.procedures_mask,
            "prior_indices": self.prior_indices,
            "prior_values": self.prior_values,
            "label": label,
        }


def get_encounter_features(encounter_infos, skip_duplicate=False,
                           min_num_codes=1, max_num_codes=50):
    """
    In the original tf implementation, dx_ints and proc_ints are serialized as variable length sequences,
    which are converted to SparseTensors when retrieved and converted to dense tensors when the lookup method
    is called to retrieve the embeddings, where max_num_codes and vocab_sizes are used to shape the tensors.

    Instead, here I explicitly store them with the proper shape, and skip the reshaping step in embedding lookup.

    """
    key_list = []
    enc_features_list = []
    conditions_hash_map = {}
    procedures_hash_map = {}
    num_cut = 0
    num_duplicate = 0
    count = 0
    num_conditions = 0
    num_procedures = 0
    num_unique_conditions = 0
    num_unique_procedures = 0
    min_conditions_cut = 0
    min_procedures_cut = 0
    max_conditions_cut = 0
    max_procedures_cut = 0
    num_expired = 0
    num_readmission = 0

    for _, enc in encounter_infos.items():
        if skip_duplicate:
            if len(enc.conditions) > len(set(enc.conditions)) or \
                    len(enc.procedures) > len(set(enc.procedures)):
                num_duplicate += 1
                continue
        if len(set(enc.conditions)) < min_num_codes:
            min_conditions_cut += 1
            continue
        if len(set(enc.procedures)) < min_num_codes:
            min_procedures_cut += 1
            continue
        if len(set(enc.conditions)) > max_num_codes:
            max_conditions_cut += 1
            continue
        if len(set(enc.procedures)) > max_num_codes:
            max_procedures_cut += 1
            continue

        count += 1
        num_conditions += len(enc.conditions)
        num_procedures += len(enc.procedures)
        num_unique_conditions += len(set(enc.conditions))
        num_unique_procedures += len(set(enc.procedures))

        # mapping the string to int
        for dx_id in enc.conditions:
            if dx_id not in conditions_hash_map:
                conditions_hash_map[dx_id] = len(conditions_hash_map)
        for treat_id in enc.procedures:
            if treat_id not in procedures_hash_map:
                procedures_hash_map[treat_id] = len(procedures_hash_map)

        if enc.label_expired:
            label_expired = 1
            num_expired += 1
        else:
            label_expired = 0

        if enc.label_readmission:
            label_readmission = 1
            num_readmission += 1
        else:
            label_readmission = 0

        conditions = sorted(list(set(enc.conditions)))
        conditions_hash = [conditions_hash_map[item] for item in conditions]
        procedures = sorted(list(set(enc.procedures)))
        procedures_hash = [procedures_hash_map[item] for item in procedures]

        enc_features = EncounterFeatures(enc.patient_id, enc.visit_id,
                                         label_expired, label_readmission,
                                         conditions, conditions_hash,
                                         procedures, procedures_hash)

        key_list.append(enc_features.encounter_key)
        enc_features_list.append(enc_features)

    # add padding
    for ef in enc_features_list:
        dx_padding_idx = len(conditions_hash_map)
        proc_padding_idx = len(procedures_hash_map)
        if len(ef.conditions_hash) < max_num_codes:
            ef.conditions_hash.extend([dx_padding_idx] * (max_num_codes - len(ef.conditions_hash)))
        if len(ef.procedures_hash) < max_num_codes:
            ef.procedures_hash.extend([proc_padding_idx] * (max_num_codes - len(ef.procedures_hash)))
        ef.conditions_mask = [0 if i == dx_padding_idx else 1 for i in ef.conditions_hash]
        ef.procedures_mask = [0 if i == proc_padding_idx else 1 for i in ef.procedures_hash]

    print('Filtered encounters due to duplicate codes: %d' % num_duplicate)
    print('Filtered encounters due to thresholding: %d' % num_cut)

    print('Min conditions cut: %d' % min_conditions_cut)
    print('Min procedures cut: %d' % min_procedures_cut)
    print('Max conditions cut: %d' % max_conditions_cut)
    print('Max procedures cut: %d' % max_procedures_cut)
    print('Number of expired: %d' % num_expired)
    print('Number of readmission: %d' % num_readmission)

    if count != 0:
        print('Average num_conditions: %f' % (num_conditions / count))
        print('Average num_procedures: %f' % (num_procedures / count))
        print('Average num_unique_conditions: %f' % (num_unique_conditions / count))
        print('Average num_unique_procedures: %f' % (num_unique_procedures / count))

    return key_list, enc_features_list, conditions_hash_map, procedures_hash_map


def select_train_valid_test(key_list, random_seed=1234):
    key_train, key_temp = ms.train_test_split(key_list, test_size=0.2, random_state=random_seed)
    key_valid, key_test = ms.train_test_split(key_temp, test_size=0.5, random_state=random_seed)
    return key_train, key_valid, key_test


def count_conditional_prob_dp(enc_features_list, output_path, train_key_set=None):
    """
    This is a Python function called count_conditional_prob_dp that takes in a list of encoded features,
    an output path to save the conditional probabilities, and a training key set, and calculates the empirical
    conditional probabilities for diagnosis-procedure (DP) and procedure-diagnosis (PD) pairs.
    """
    dx_freqs = {}  # diagnosis
    proc_freqs = {}  # treatment
    dp_freqs = {}  # diagnosis-treatment

    total_visit = 0
    for enc_feature in enc_features_list:
        key = enc_feature.encounter_key
        if train_key_set is not None and key not in train_key_set:
            total_visit += 1
            continue
        dx_ids = enc_feature.conditions
        proc_ids = enc_feature.procedures
        for dx in dx_ids:
            if dx not in dx_freqs:
                dx_freqs[dx] = 0
            dx_freqs[dx] += 1
        for proc in proc_ids:
            if proc not in proc_freqs:
                proc_freqs[proc] = 0
            proc_freqs[proc] += 1
        for dx in dx_ids:
            for proc in proc_ids:
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


def add_sparse_prior_guide_dp(enc_features_list, stats_path, key_set=None, max_num_codes=50):
    dp_cond_probs = pickle.load(open(os.path.join(stats_path, 'dp_cond_probs.empirical.p'), 'rb'))
    pd_cond_probs = pickle.load(open(os.path.join(stats_path, 'pd_cond_probs.empirical.p'), 'rb'))

    print('Adding prior guide')
    total_visit = 0
    new_enc_features_list = []
    # prior_guide_list = []
    for enc_features in enc_features_list:
        key = enc_features.encounter_key
        if key_set is not None and key not in key_set:
            total_visit += 1
            continue
        dx_ids = enc_features.conditions
        proc_ids = enc_features.procedures
        indices = []
        values = []
        for i, dx in enumerate(dx_ids):
            for j, proc in enumerate(proc_ids):
                dp = dx + ',' + proc
                indices.append((i, max_num_codes + j))
                prob = 0.0 if dp not in dp_cond_probs else dp_cond_probs[dp]
                values.append(prob)
        for i, proc in enumerate(proc_ids):
            for j, dx in enumerate(dx_ids):
                pd = proc + ',' + dx
                indices.append((max_num_codes + i, j))
                prob = 0.0 if pd not in pd_cond_probs else pd_cond_probs[pd]
                values.append(prob)
        # indices = list(np.array(indices).reshape([-1]))

        enc_features.prior_indices = indices
        enc_features.prior_values = values
        new_enc_features_list.append(enc_features)

        total_visit += 1
    return new_enc_features_list


def convert_features_to_tensors(enc_features):
    all_conditions_hash = torch.tensor([f.conditions_hash for f in enc_features], dtype=torch.long)
    all_procedures_hash = torch.tensor([f.procedures_hash for f in enc_features], dtype=torch.long)
    all_conditions_masks = torch.tensor([f.conditions_mask for f in enc_features], dtype=torch.float)
    all_procedures_masks = torch.tensor([f.procedures_mask for f in enc_features], dtype=torch.float)
    all_readmission_labels = torch.tensor([f.label_readmission for f in enc_features], dtype=torch.long)
    all_expired_labels = torch.tensor([f.label_expired for f in enc_features], dtype=torch.long)
    # all_prior_indices = torch.tensor([f.prior_indices for f in enc_features], dtype=torch.long)
    # all_prior_values = torch.tensor([f.prior_values for f in enc_features], dtype=torch.float)
    dataset = TensorDataset(all_conditions_hash, all_procedures_hash,
                            all_conditions_masks, all_procedures_masks,
                            all_readmission_labels, all_expired_labels)

    return dataset


def get_prior_guide(enc_features):
    prior_guide_list = []
    for feats in enc_features:
        indices = torch.tensor(list(zip(*feats.prior_indices))).reshape(2, -1)
        values = torch.tensor(feats.prior_values)
        prior_guide_list.append((indices, values))
    return prior_guide_list
