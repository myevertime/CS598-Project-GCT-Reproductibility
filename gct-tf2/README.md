# Graph Convolutional Transformer in TensorFlow 2 and Python 3

This branch involves reimplementation of the official code repository of [this paper](https://arxiv.org/pdf/1906.04716.pdf) by Choi et al, following the materials provided.

While the official code is written in Python 2 and TensorFlow 1, this repository is an upgraded version that uses TensorFlow 2 and Python 3.

To run the code, please follow the instructions below:

1. First, you should have eICU CSV data files under the *eicu_samples* folder. This involves CITI training participation. We need four CSV files as below:
- patient.csv
- admissionDx.csv 
- diagnosis.csv 
- treatment.csv

2. Then, run eicu_process.py to convert raw data in TFRecord format so that the model can read training data.

```python process_eicu.py <path to CSV files> <output path>```

By default, this will generate 5 folds of sets of train/validation/test data.

3. Finally, run train.py to train the GCT model on eICU data.

```python train.py <path to TFRecords> <output path>```

Contributor: [Hyeonjae Cho](https://github.com/myevertime)
