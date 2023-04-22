# Graph Convolutional Transformer in pytorch

The following gct-pytorch implementation mostly adopted from the [dchang56/gct-pytorch](https://github.com/dchang56/gct-pytorch) with slight modifications to fit our project purpose. Output of the training results is also provided for reference.

Viewing the result with Tensorboard
```shell
tensorboard --logdir=eicu_output/model_0.00022_0.08_readmission
```

### Process CSV files output

```shell
200859it [00:01, 152783.21it/s]
dropped 132783 encounters due to duration > 24 hours
Processing admission diagnosis.csv
626858it [00:01, 427847.95it/s]

Admission Diagnosis without encounter id: 450589
Processing diagnosis.csv
2710672it [00:05, 497254.82it/s]
Diagnosis without encounter id: 2483092
Processing treatment.csv
3688745it [00:06, 554454.24it/s]

Treatment without encounter id: 3372000
accepted treatment: 316745

Filtered encounters due to duplicate codes: 0
Filtered encounters due to thresholding: 0

Min dx cut: 16670
Min treatment cut: 10373
Max dx cut: 1
Max treatment cut: 6
Number of expired: 2983
Number of readmission: 7051

Average num_dx_ids: 8.253912
Average num_treatments: 7.697826
Average num_unique_dx_ids: 6.462268
Average num_unique_treatments: 5.026276

Adding prior guide
Adding prior guide
Adding prior guide


```