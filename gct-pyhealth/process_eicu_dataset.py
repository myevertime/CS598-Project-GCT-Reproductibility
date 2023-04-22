from pyhealth.datasets import eICUDataset
from process_encounters import *


def process_eicu_dataset(data_dir, eicu_dataset: eICUDataset, fold=0):
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
        encounter_dict = {}

        # Loading pyhealth eICUDataset, parse it into encounter_dict
        print('Loading eICU dataset')
        encounter_dict = process_eicudataset(encounter_dict, eicu_dataset=eicu_dataset)

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
    eicu_dataset = eICUDataset(
        root='../eicu_csv',
        tables=["treatment", "admissionDx", "diagnosisString"],
        refresh_cache=False,
        # dev=True,
    )
    data_dir = './eicu_data'
    process_eicu_dataset(data_dir, eicu_dataset, fold=0)
