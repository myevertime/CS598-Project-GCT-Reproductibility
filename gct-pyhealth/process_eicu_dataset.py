from pyhealth.datasets import eICUDataset
from process_encounters import *


def readmission_prediction_eicu_fn_basic(patient: Patient, time_window=5):
    samples = []
    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        # get time difference between current visit and next visit
        time_diff = (next_visit.encounter_time - visit.encounter_time).days
        readmission_label = 1 if time_diff < time_window else 0

        # exclude: visits without any codes/events
        if visit.num_events == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "label": readmission_label,
            }
        )
    # no cohort selection
    return samples


# parse the eicu data using pyhealth
# return encounter_dict
def get_encounter_dict(eicu_dataset: eICUDataset):
    encounter_dict = {}
    encounter_counts = 0
    hour_threshold = 24
    dropped_short_encounter_counts = 0

    # parse patient information
    for patient_id, patient in eicu_dataset.patients.items():
        patient_id = patient.patient_id

        # readmission labels
        readmission_samples = readmission_prediction_eicu_fn_basic(patient, time_window=30)

        # process visit information
        for encounter_id, visit in patient.visits.items():
            encounter_timestamp = visit.encounter_time

            # dropping patient with more than 24 hours duration minute
            if (visit.discharge_time - visit.encounter_time) > np.timedelta64(hour_threshold, 'h'):
                dropped_short_encounter_counts += 1
                continue

            # mortality labels
            expired = True if visit.discharge_status == 'Expired' else False

            # readmission labels, check if the encounter is in the readmission samples
            readmission = 0
            for sample in readmission_samples:
                if sample['visit_id'] == encounter_id:
                    readmission = sample['label']

            # pack it to EncounterInfo
            encounter = EncounterInfo(patient_id, encounter_id, encounter_timestamp, expired, readmission)

            # extract codes list of the visit
            conditions = [cond.lower() for cond in visit.get_code_list(table="diagnosisString")]
            admissionDx = [dx.lower() for dx in visit.get_code_list(table="admissionDx")]
            treatment = [treat.lower() for treat in visit.get_code_list(table="treatment")]
            # procedures = visit.get_code_list(table="physicalExam")
            # drugs = visit.get_code_list(table="medication")
            # lab = visit.get_code_list(table="lab")

            # parse diagnosis ids
            encounter.dx_ids = admissionDx + conditions
            # parse treatment ids
            encounter.treatments = treatment

            if encounter_id in encounter_dict:
                print('duplicate encounter id! skip')
                sys.exit(0)
            encounter_dict[encounter_id] = encounter
            encounter_counts += 1

    print('encounter counts: ', encounter_counts)
    print('dropped short encounter counts: ', dropped_short_encounter_counts)
    return encounter_dict


def get_eicu_datasets(data_dir, fold=0):
    # instead of generating 5 folds manually prior to training using 2 separate scripts, let's generate 1 fold in
    # same script patient_file = os.path.join(data_dir, 'patient.csv') admission_dx_file = os.path.join(data_dir,
    # 'admissionDx.csv') diagnosis_file = os.path.join(data_dir, 'diagnosis.csv') treatment_file = os.path.join(
    # data_dir, 'treatment.csv')

    fold_path = os.path.join(data_dir, 'fold_{}'.format(fold))
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)

    stats_path = os.path.join(fold_path, 'train_stats')
    if not os.path.exists(stats_path):
        os.makedirs(stats_path)

    cached_path = os.path.join(fold_path, 'cached')
    if os.path.exists(cached_path):
        start = time.time()
        train_dataset = torch.load(os.path.join(cached_path, 'train_dataset.pt'))
        validation_dataset = torch.load(os.path.join(cached_path, 'valid_dataset.pt'))
        test_dataset = torch.load(os.path.join(cached_path, 'test_dataset.pt'))

        train_prior_guide = torch.load(os.path.join(cached_path, 'train_priors.pt'))
        validation_prior_guide = torch.load(os.path.join(cached_path, 'valid_priors.pt'))
        test_prior_guide = torch.load(os.path.join(cached_path, 'test_priors.pt'))

    else:
        os.makedirs(cached_path)

        # Loading pyhealth eICUDataset, parse it into encounter_dict
        print('Loading eICU dataset')
        eicu_dataset = eICUDataset(
            root='../eicu_csv',
            tables=["treatment", "admissionDx", "diagnosisString"],
            refresh_cache=False,
            # dev=True,
        )

        # parse the eICU dataset to encounter dict
        encounter_dict = get_encounter_dict(eicu_dataset=eicu_dataset)

        key_list, enc_features_list, dx_map, proc_map = get_encounter_features(encounter_dict, skip_duplicate=False,
                                                                               min_num_codes=1, max_num_codes=50)
        pickle.dump(dx_map, open(os.path.join(fold_path, 'dx_map.p'), 'wb'))
        pickle.dump(proc_map, open(os.path.join(fold_path, 'proc_map.p'), 'wb'))

        key_train, key_valid, key_test = select_train_valid_test(key_list, random_seed=fold)

        count_conditional_prob_dp(enc_features_list, stats_path, set(key_train))

        train_enc_features = add_sparse_prior_guide_dp(enc_features_list, stats_path, set(key_train), max_num_codes=50)
        validation_enc_features = add_sparse_prior_guide_dp(enc_features_list, stats_path, set(key_valid),
                                                            max_num_codes=50)
        test_enc_features = add_sparse_prior_guide_dp(enc_features_list, stats_path, set(key_test), max_num_codes=50)

        train_dataset = convert_features_to_tensors(train_enc_features)
        validation_dataset = convert_features_to_tensors(validation_enc_features)
        test_dataset = convert_features_to_tensors(test_enc_features)

        torch.save(train_dataset, os.path.join(cached_path, 'train_dataset.pt'))
        torch.save(validation_dataset, os.path.join(cached_path, 'valid_dataset.pt'))
        torch.save(test_dataset, os.path.join(cached_path, 'test_dataset.pt'))

        # get prior_indices and prior_values for each split and save as list of tensors
        train_prior_guide = get_prior_guide(train_enc_features)
        validation_prior_guide = get_prior_guide(validation_enc_features)
        test_prior_guide = get_prior_guide(test_enc_features)

        # save the prior_indices and prior_values
        torch.save(train_prior_guide, os.path.join(cached_path, 'train_priors.pt'))
        torch.save(validation_prior_guide, os.path.join(cached_path, 'valid_priors.pt'))
        torch.save(test_prior_guide, os.path.join(cached_path, 'test_priors.pt'))

    return (
        [train_dataset, validation_dataset, test_dataset],
        [train_prior_guide, validation_prior_guide, test_prior_guide])


if __name__ == "__main__":
    data_dir = './eicu_data'
    get_eicu_datasets(data_dir, fold=1)
