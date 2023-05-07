from .process_encounters import *
from pyhealth.data import Event, Visit, Patient


def readmission_prediction_eicu_fn_basic(patient: Patient, time_window=5):
    samples = []
    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        # get time difference between current visit and next visit
        time_diff = (next_visit.encounter_time - visit.encounter_time).days
        label = 1 if time_diff < time_window else 0

        # exclude: visits without any codes/events
        if visit.num_events == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "label": label,
            }
        )
    # no cohort selection
    return samples


# parse the eicu data using pyhealth
# return encounter_info
def get_encounter_infos(eicu_dataset: eICUDataset):
    encounter_infos = {}
    encounter_counts = 0
    hour_threshold = 24
    dropped_short_encounter_counts = 0

    # parse patient information
    for patient_id, patient in eicu_dataset.patients.items():
        patient_id = patient.patient_id

        # readmission labels
        readmission_samples = readmission_prediction_eicu_fn_basic(patient, time_window=30)

        # process visit information
        for visit_id, visit in patient.visits.items():
            encounter_timestamp = visit.encounter_time

            # dropping patient with more than 24 hours duration minute
            if (visit.discharge_time - visit.encounter_time) > np.timedelta64(hour_threshold, 'h'):
                dropped_short_encounter_counts += 1
                continue

            # mortality labels
            label_expired = 1 if visit.discharge_status == 'Expired' else 0

            # readmission labels, check if the encounter is in the readmission samples
            label_readmission = 0
            for sample in readmission_samples:
                if sample['visit_id'] == visit_id:
                    label_readmission = sample['label']

            # pack it to EncounterInfo
            encounter = EncounterInfo(patient_id, visit_id, encounter_timestamp,
                                      label_expired, label_readmission)

            # extract codes list of the visit
            # diagnosis = [cond.lower() for cond in visit.get_code_list(table="diagnosisString")]
            diagnosis = list(set([dx.attr_dict["diagnosisString"] for dx in visit.get_event_list("diagnosis")]))
            admissionDx = [dx.lower() for dx in visit.get_code_list(table="admissionDx")]
            treatment = [treat.lower() for treat in visit.get_code_list(table="treatment")]
            # procedures = visit.get_code_list(table="physicalExam")
            # drugs = visit.get_code_list(table="medication")
            # lab = visit.get_code_list(table="lab")

            # parse diagnosis ids
            encounter.conditions = admissionDx + diagnosis
            # parse procedures ids
            encounter.procedures = treatment

            if visit_id in encounter_infos:
                print('duplicate encounter id! skip')
                sys.exit(0)
            encounter_infos[visit_id] = encounter
            encounter_counts += 1

    print('encounter counts: ', encounter_counts)
    print('dropped short encounter counts: ', dropped_short_encounter_counts)
    return encounter_infos


def get_eicu_datasets(data_dir, eicu_csv_dir, fold=0):
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
        print("Found cached data, loading...")
        start = time.time()
        train_dataset = torch.load(os.path.join(cached_path, 'train_dataset.pt'))
        validation_dataset = torch.load(os.path.join(cached_path, 'valid_dataset.pt'))
        test_dataset = torch.load(os.path.join(cached_path, 'test_dataset.pt'))

        train_prior_guide = torch.load(os.path.join(cached_path, 'train_priors.pt'))
        validation_prior_guide = torch.load(os.path.join(cached_path, 'valid_priors.pt'))
        test_prior_guide = torch.load(os.path.join(cached_path, 'test_priors.pt'))
        print('loading cached data takes: {}s'.format(time.time() - start))

    else:
        os.makedirs(cached_path)

        # Loading pyhealth eICUDataset, parse it into encounter_infos
        print('Loading eICU dataset')
        eicu_dataset = eICUDataset(
            root=eicu_csv_dir,
            tables=["treatment", "admissionDx", "diagnosis"],
            refresh_cache=False,
            # dev=True,
        )

        # parse the eICU dataset to encounter dict
        encounter_infos = get_encounter_infos(eicu_dataset=eicu_dataset)

        key_list, enc_features_list, dx_map, proc_map = \
            get_encounter_features(encounter_infos, skip_duplicate=False,
                                   min_num_codes=1, max_num_codes=50)

        pickle.dump(dx_map, open(os.path.join(fold_path, 'dx_map.p'), 'wb'))
        pickle.dump(proc_map, open(os.path.join(fold_path, 'proc_map.p'), 'wb'))

        key_train, key_valid, key_test = select_train_valid_test(key_list, random_seed=fold)

        # compute and store the conditional probability mapping
        count_conditional_prob_dp(enc_features_list, stats_path, set(key_train))
        train_enc_features = add_sparse_prior_guide_dp(enc_features_list, stats_path,
                                                       set(key_train), max_num_codes=50)
        validation_enc_features = add_sparse_prior_guide_dp(enc_features_list, stats_path,
                                                            set(key_valid), max_num_codes=50)
        test_enc_features = add_sparse_prior_guide_dp(enc_features_list, stats_path,
                                                      set(key_test), max_num_codes=50)

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
    cache_dir = './eicu_data'
    eicu_csv_dir = '../eicu_csv'
    get_eicu_datasets(cache_dir, eicu_csv_dir, fold=0)
