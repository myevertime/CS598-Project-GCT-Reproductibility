## Graph Convolution Transformer

### prerequisites

- PyHealth: [modified version](https://github.com/lycpaul/PyHealth)

### Getting started

1. Parse the dataset

```shell
python process_eicu_dataset.py
```

Expected output / statistics:

Without dropping short encounter (<24 hours discharge time)
```shell
INFO: Pandarallel will run on 6 workers.
INFO: Pandarallel will use Memory file system to transfer data between the main process and workers.
Parsing patients: 100%|██████████| 166355/166355 [04:03<00:00, 682.45it/s]
finish basic patient information parsing : 247.49314141273499s
finish parsing treatment : 103.23857140541077s
finish parsing admissionDx : 59.03062725067139s
finish parsing diagnosisString : 108.00234651565552s
Mapping codes: 100%|██████████| 166355/166355 [00:17<00:00, 9767.95it/s] 
Loading eICU dataset
encounter counts:  200859

Filtered encounters due to duplicate codes: 0
Filtered encounters due to thresholding: 0

Min dx cut: 20846
Min treatment cut: 27611
Max dx cut: 138
Max treatment cut: 635
Number of expired: 9220
Number of readmission: 19047

Average num_dx_ids: 7.463895
Average num_treatments: 6.999591
Average num_unique_dx_ids: 7.463895
Average num_unique_treatments: 6.999591

Adding prior guide
Adding prior guide
Adding prior guide
```


Dropping long encounter (>24 hours discharge time)
```shell
Loading eICU dataset
encounter counts:  68076
dropped short encounter counts:  132783
Filtered encounters due to duplicate codes: 0
Filtered encounters due to thresholding: 0

Min dx cut: 16670
Min treatment cut: 10373
Max dx cut: 1
Max treatment cut: 6
Number of expired: 2983
Number of readmission: 5282

Average num_dx_ids: 6.462268
Average num_treatments: 5.026276
Average num_unique_dx_ids: 6.462268
Average num_unique_treatments: 5.026276

Adding prior guide
Adding prior guide
Adding prior guide

```