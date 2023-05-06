# Welcome to GCT Reproducibility Test Repo!
This repository contains the multiple ways of implementing the Graph Convolutional Transformer model, reproduced by Team 30 as part of the UIUC 2023 Spring CS598 Deep Learning for Healthcare course.

## Team info
- @[lycpaul](https://github.com/lycpaul)
- @[myevertime](https://github.com/myevertime)

## Project Goal
Our team's objective was to reproduce the Graph Convolutional Transformer model outlined in [the paper](https://arxiv.org/pdf/1906.04716.pdf) by Choi et al. (2020). While there is an official [code repository](https://github.com/Google-Health/records-research/tree/master/graph-convolutional-transformer) available, it was outdated, utilizing Python 2.7 and TensorFlow 1.13. 

To overcome this, we re-implemented the codes using the latest versions of Python 3 and TensorFlow 2. Additionally, we ported the codes to PyTorch, as our long-term goal was to integrate the model with PyHealth, a platform for medical practitioners and ML researchers to deploy healthcare AI applications and customize models more easily.

To enhance the understanding of GCT, we also added auxiliary functions, such as visualizing the attention map of the model, to demonstrate the superiority of the GCT compared to original Transformer-based models. Furthermore, we expanded our training results by experimenting with different model parameters.

## Directory Structure

The repository is structured as follows:
```
.
├── gct-pyhealth/
│   ├── archive/
│   ├── eicu_output/
│   ├── gctpyhealth/
│   └── training_scripts/
│   └── ...
├── gct-pytorch/
│   ├── eicu_output/
│   ├── notebooks/
│   ├── training_scripts/
│   └── ...
└── gct-tf2/
    ├── eicu_samples/
    └── ...
```

The repository contains three main folders: `gct-pytorch`, `gct-pyhealth`, and `gct-tf2`. Each folder contains the implementation of the Graph Convolutional Transformer model in a specific framework. Within each folder, there are subdirectories for `eicu_output`, `training_cripts`, `eicu_samples`, and other supporting files. The `eicu_output` directory contains the model outputs with `eval_results.txt` and `test_results.txt`.  The `eicu_samples` directory contains the eICU dataset and the preprocessed data will be saved under the specified directory from the argument parameter. The `training_scripts` directory contains the bash code for specific parameters provided by the original paper.

We have also included descriptive notebooks under the `gct-pyhealth` and `gct-pytorch/notebooks` directories to help with understanding the overall code flow. These notebooks provide a step-by-step walkthrough of the data preprocessing, model training, and evaluation process. They also include visualizations of the model's attention maps.

## Dataset
The Graph Convolutional Transformer model utilizes the eICU dataset, which consists of electronic health records for over 200,000 patients who received intensive care. The dataset includes various patient information, such as vital signs, lab results, medications, demographics, and clinical notes.

To access the original data used in the paper, you can download it from [here](https://eicu-crd.mit.edu/gettingstarted/access/). However, to download the data, you are required to participate in the CITI training program, and you can find the detailed guidelines on the eICU website.

The model's training data includes four CSV files: `admissionDx`, `patient`, `treatment`, and `diagnosis`. The data is then preprocessed using a processing script, which extracts relevant features and constructs a graph adjacency matrix. The graph adjacency matrix illustrates the relationships between different clinical events, such as diagnoses, medications, and procedures, for each patient. This graph is then utilized as input to the Graph Convolutional Transformer model to predict patient outcomes.

## Usage
1.  Clone the repository to your local machine.
2.  Install the necessary dependencies by running `pip install -r requirements.txt`.
3.  Run the preprocessing code as described in the corresponding section below.
4.  Run the training code as described in the corresponding section below.

## Dependencies

Please check `requirements.txt`

## Preprocessing/Training command
For instructions on how to preprocess the data and train the model, please navigate to the corresponding folder and refer to the `README.md` file.

## Table of Reproducibility Test Results

|                |Test AUCPR                          |Test AUROC                        |
|----------------|-------------------------------|-----------------------------|
|Readmission (Ours-GCT)|0.4081            |0.6088            |
|Mortality (Ours-GCT)|0.5931            |0.8118            |

For more detailed information on the hyperparameter settings and the training process, please refer to the `report.pdf` file included in the repository. This report provides a comprehensive overview of our experimental setup and analysis of the results.
