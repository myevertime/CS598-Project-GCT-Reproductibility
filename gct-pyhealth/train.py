#!/usr/bin/env python
# coding: utf-8

# # Graph Convolution Transformer (GCT) for eICU dataset

# ### **Step 0: Import libaries, prepare proper parameters**
# - **[README]:** We provide a set of default parameters for users to run the experiments. Users can also change the parameters to fit their own needs.

# In[14]:


import torch
import numpy as np
import os
import sys
import math
import logging
import json
import datetime

from tqdm import tqdm, trange
from gctpyhealth.process_eicu_dataset import get_eicu_datasets
from gctpyhealth.utils import *
from gctpyhealth.gct import GCT

from tensorboardX import SummaryWriter
import torchsummary as summary


# In[15]:


class Args:
    def __init__(self, prediction_task: str):
        if prediction_task == "expired":
            self.label_key = "expired"
            self.learning_rate = 0.00011
            self.reg_coef = 1.5
            self.hidden_dropout = 0.72
        elif prediction_task == "readmission":
            self.label_key = "readmission"
            self.learning_rate = 0.00022
            self.reg_coef = 0.1
            self.hidden_dropout = 0.08
        else:
            raise ValueError("Invalid prediction task: {}".format(prediction_task))

        # Training arguments
        self.max_steps = 1000000
        self.warmup = 0.05  # default
        self.logging_steps = 100  # default
        self.num_train_epochs = 1  # default
        self.seed = 42  # default

        # Model parameters arguments
        self.embedding_dim = 128
        self.max_num_codes = 50
        self.num_stacks = 3
        self.batch_size = 32
        self.prior_scalar = 0.5
        self.num_heads = 1

        # save and load the cache/dataset/env path (required)
        self.fold = 0
        self.data_dir = "eicu_data"
        self.eicu_csv_dir = "../eicu_csv"
        # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # self.output_dir = "eicu_output/model_pyhealth_" + timestamp
        self.output_dir = "eicu_output/model_pyhealth_" + self.label_key

        # save and load the models (optional)
        self.save_model = True
        self.load_prev_model = False
        self.prev_model_path = "eicu_output/model_pyhealth_" + self.label_key + "/model.pt"


args = Args("expired")
set_seed(args.seed)

# ### **Step 1: Load dataset**
# - **[README]:** We call [pyhealth.datasets](https://pyhealth.readthedocs.io/en/latest/api/datasets.html) to process and obtain the dataset.
#   - `root` is the arguments directing to the data folder.
#   - `tables` is a list of table names from raw databases, which specifies the information that will be used in building the pipeline. Currently, we provide [MIMIC3Dataset](https://pyhealth.readthedocs.io/en/latest/api/datasets/pyhealth.datasets.MIMIC3Dataset.html), [MIMIC4Dataset](https://pyhealth.readthedocs.io/en/latest/api/datasets/pyhealth.datasets.MIMIC4Dataset.html), [eICUDataset](https://pyhealth.readthedocs.io/en/latest/api/datasets/pyhealth.datasets.eICUDataset.html), [OMOPDataset](https://pyhealth.readthedocs.io/en/latest/api/datasets/pyhealth.datasets.OMOPDataset.html).
#   - `code_mapping [default: None]` asks a directionary input, specifying the new coding systems for each data table. For example, `{"NDC": ("ATC", {"target_kwargs": {"level": 3}})}` means that our pyhealth will automatically change the codings from `NDC` into ATC-3 level for tables if any.
#   - `dev`: if set `True`, will only load a smaller set of patients.
# - **[Next Step]:** This `pyhealth.datasets` object will be used in **Step 2**.
# - **[Advanced Use Case]:** Researchers can use the dict-based output alone `dataset.patients` alone for supporting their own tasks.

# In[16]:


# Store the log data
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
logging_dir = os.path.join(args.output_dir, 'logging')
if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)
tb_writer = SummaryWriter(log_dir=logging_dir)

# In[17]:


# loading the eICU dataset
from pyhealth.datasets import eICUDataset

print('Loading eICU dataset')
eicu_ds = eICUDataset(
    root=args.eicu_csv_dir,
    tables=["admissionDx", "diagnosisString", "treatment"],
    refresh_cache=False,
    dev=True
)

print(eicu_ds.stat())
print(eicu_ds.info())

# ### **Step 2: Create Dataloader**
# - **[README]:** We can also load the preprocessed datasets dict from cache and create the dataloader accordingly.

# In[18]:


# fetch the datatset from caches
datasets, prior_guides = get_eicu_datasets(args.data_dir, args.eicu_csv_dir, fold=args.fold)
train_dataset, eval_dataset, test_dataset = datasets
train_priors, eval_priors, test_priors = prior_guides
train_priors_dataset = eICUPriorDataset(train_priors)
eval_priors_dataset = eICUPriorDataset(eval_priors)
test_priors_dataset = eICUPriorDataset(test_priors)

# prepare data loader
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

train_priors_dataloader = DataLoader(train_priors_dataset,
                                     batch_size=args.batch_size, collate_fn=priors_collate_fn)
eval_priors_dataloader = DataLoader(eval_priors_dataset,
                                    batch_size=args.batch_size, collate_fn=priors_collate_fn)
test_priors_dataloader = DataLoader(test_priors_dataset,
                                    batch_size=args.batch_size, collate_fn=priors_collate_fn)

# In[19]:


# check if gpu/cuda is available
n_gpu = torch.cuda.device_count()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.cuda.set_device(device)
    logger.info('***** Using CUDA device *****')

# ### **Step 3: Define ML Model**
# - **[README]:** We initialize an ML model for the healthcare task by calling [pyhealth.models](https://pyhealth.readthedocs.io/en/latest/api/models.html).
# - **[Next Step]:** This `pyhealth.models` object will be used in **Step 4**.
# - **[Other Use Case]:** Our `pyhealth.models` object is as general as any instance from `torch.nn.Module`. Users may use it separately for supporting any other customized pipeline.

# In[20]:


# from pyhealth.models import Transformer
from gctpyhealth.gct import GCT

model = GCT(
    dataset=eicu_ds,
    feature_keys=['conditions_hash',
                  'procedures_hash'],
    label_key=args.label_key,
    mode="binary",
    embedding_dim=args.embedding_dim,
    max_num_codes=args.max_num_codes,
    num_stacks=args.num_stacks,
    batch_size=args.batch_size,
    reg_coef=args.reg_coef,
    prior_scalar=args.prior_scalar,
    hidden_dropout=args.hidden_dropout,
    num_heads=args.num_heads,
)

# loading previous checkpoint if available
checkpoint = None
if args.load_prev_model:
    checkpoint = torch.load(args.prev_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(device)

# ### **Step 4: Model Training**
# - **[README]:** We call our [pyhealth.train.Trainer](https://pyhealth.readthedocs.io/en/latest/api/trainer.html) to train the model by giving the `train_loader`, the `val_loader`, val_metric, and specify other arguemnts, such as epochs, optimizer, learning rate, etc. The trainer will automatically save the best model and output the path in the end.
# - **[Next Step]:** The best model will be used in **Step 5** for evaluation.
# 

# In[21]:


# compute how steps and epoch is required
num_update_steps_per_epoch = len(train_dataloader)
if args.max_steps > 0:
    max_steps = args.max_steps
    num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
        args.max_steps % num_update_steps_per_epoch > 0)
else:
    max_steps = int(num_update_steps_per_epoch * args.num_train_epochs)
    num_train_epochs = args.num_train_epochs
num_train_epochs = int(np.ceil(num_train_epochs))

args.eval_steps = num_update_steps_per_epoch // 2

# prepare optimizer, scheduler
optimizer = torch.optim.Adamax(model.parameters(), lr=args.learning_rate)
warmup_steps = max_steps // (1 / args.warmup)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps=max_steps)

logger.info('***** Running Training *****')
logger.info(' Num examples = {}'.format(len(train_dataloader.dataset)))
logger.info(' Num epochs = {}'.format(num_train_epochs))
logger.info(' Train batch size = {}'.format(args.batch_size))
logger.info(' Total optimization steps = {}'.format(max_steps))

epochs_trained = 0
global_step = 0
tr_loss = torch.tensor(0.0).to(device)
logging_loss_scalar = 0.0
model.zero_grad()

# check if we have previous checkpoint
if args.load_prev_model and checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epochs_trained = checkpoint['epochs_trained']
    global_step = checkpoint['global_step']

# In[22]:


training_outputs = None
train_pbar = trange(epochs_trained, num_train_epochs, desc='Epoch')
for epoch in range(epochs_trained, num_train_epochs):
    epoch_pbar = tqdm(train_dataloader, desc='Iteration')
    for data, priors_data in zip(train_dataloader, train_priors_dataloader):
        model.train()
        data, priors_data = prepare_data(data, priors_data, device)

        # [loss, logits, all_hidden_states, all_attentions]
        training_outputs = model(data, priors_data)
        loss = training_outputs['loss']

        if n_gpu > 1:
            loss = loss.mean()
        loss.backward()

        tr_loss += loss.detach()
        optimizer.step()
        scheduler.step()
        model.zero_grad()

        # update the global step
        global_step += 1

        # print out the training results
        if args.logging_steps > 0 and global_step % args.logging_steps == 0:
            logs = {}
            tr_loss_scalar = tr_loss.item()
            logs['loss'] = (tr_loss_scalar - logging_loss_scalar) / args.logging_steps
            logs['learning_rate'] = scheduler.get_last_lr()[0]
            logging_loss_scalar = tr_loss_scalar
            if tb_writer:
                for k, v in logs.items():
                    if isinstance(v, (int, float)):
                        tb_writer.add_scalar(k, v, global_step)
                tb_writer.flush()
            output = {**logs, **{"step": global_step}}
            print(output)

        # print out the evaluation results
        if args.eval_steps > 0 and global_step % args.eval_steps == 0:
            metrics = prediction_loop(device, args.label_key,
                                      model, eval_dataloader, eval_priors_dataloader)
            logger.info('**** Checkpoint Eval Results ****')
            for key, value in metrics.items():
                logger.info('{} = {}'.format(key, value))
                tb_writer.add_scalar(key, value, global_step)

        epoch_pbar.update(1)
        if global_step >= max_steps:
            break

    epoch_pbar.close()
    train_pbar.update(1)
    if global_step >= max_steps:
        break

train_pbar.close()
if tb_writer:
    tb_writer.close()

logging.info('\n\nTraining completed')

# ### **Step 5: Evaluation**

# In[23]:


# Evaluation
eval_results = {}

logger.info('*** Evaluate ***')
logger.info(' Num examples = {}'.format(len(eval_dataloader.dataset)))
eval_result = prediction_loop(device, args.label_key, model, eval_dataloader, eval_priors_dataloader)
output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')

with open(output_eval_file, 'a') as writer:
    logger.info('*** Eval results @ steps:{} ***\n'.format(global_step))
    writer.write('*** Eval results @ steps:{} ***\n'.format(global_step))
    for key, value in eval_result.items():
        logger.info('{} = {}\n'.format(key, value))
        writer.write('{} = {}\n'.format(key, value))
eval_results.update(eval_result)

# ### **Step 6: Inference**

# In[24]:


# Test and predict
logging.info('*** Test ***')
test_result = prediction_loop(device, args.label_key, model, test_dataloader, test_priors_dataloader,
                              description='Testing')
output_test_file = os.path.join(args.output_dir, 'test_results.txt')
with open(output_test_file, 'a') as writer:
    logger.info('*** Test results @ steps:{} ***\n'.format(global_step))
    writer.write('*** Test results @ steps:{} ***\n'.format(global_step))
    for key, value in test_result.items():
        logger.info('{} = {}\n'.format(key, value))
        writer.write('{} = {}\n'.format(key, value))
eval_results.update(test_result)

# ### **Step 7: Save model**
# Reference: [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training)
# 

# In[25]:


# print model's state_dict
logger.info('Model state_dict:')
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# In[26]:


# if enable save model option, save the model
if args.save_model:
    torch.save({
        'epochs_trained': epochs_trained,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': training_outputs['loss'],
        'all_hidden_states': training_outputs['all_hidden_states'],
        'all_attentions': training_outputs['all_attentions']
    }, args.output_dir + '/model.pt')
