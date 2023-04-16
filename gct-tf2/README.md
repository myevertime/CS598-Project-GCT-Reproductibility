# Graph Convolutional Transformer in Tensorflow2 and Python3

This branch involves reimplementation of the official code repository of [this paper](https://arxiv.org/pdf/1906.04716.pdf) by Choi et al, following the materials provided.

First you should add eICU csv data files under eicu_sampels folder.
> patient, admissionDx, diagnosis, treatment CSV files.

Then run eicu_process.py to convert data in TFRecord format.
python process_eicu.py <path to CSV files> <output path>

By default, this will generate 5 folds of sets of train/validation/test data.

Finally, run train.py that trains the GCT model on eICU data.
python train.py <path to TFRecords> <output path>.

Contributor: [Hyeonjae Cho](https://github.com/myevertime)