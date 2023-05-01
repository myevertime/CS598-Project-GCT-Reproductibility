from pyhealth.datasets import eICUDataset

cache_dir = "eicu_data"
eicu_csv_dir = "../eicu_csv"

print('Loading eICU dataset')
eicu_ds = eICUDataset(
    root=eicu_csv_dir,
    tables=["admissionDx", "diagnosisString", "treatment"],
    refresh_cache=False,
    dev=True
)

eicu_ds.stat()

# data format
eicu_ds.info()

# from pyhealth.tasks import mortality_prediction_eicu_fn2
#
# eicu_ds_mortality = eicu_ds.set_task(task_fn=mortality_prediction_eicu_fn2)
# # stats info
# eicu_ds_mortality.stat()

from gctpyhealth.process_eicu_dataset import get_eicu_datasets
from gctpyhealth.utils import eICUPriorDataset

# loading the preprocessed eicu dataset from the cache
datasets, prior_guides = get_eicu_datasets(cache_dir, eicu_csv_dir, fold=0)

train_dataset, eval_dataset, test_dataset = datasets
train_priors, eval_priors, test_priors = prior_guides
train_priors_dataset = eICUPriorDataset(train_priors)
eval_priors_dataset = eICUPriorDataset(eval_priors)
test_priors_dataset = eICUPriorDataset(test_priors)

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from gctpyhealth.utils import priors_collate_fn

batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

train_priors_dataloader = DataLoader(train_priors_dataset, batch_size=batch_size, collate_fn=priors_collate_fn)
eval_priors_dataloader = DataLoader(eval_priors_dataset, batch_size=batch_size, collate_fn=priors_collate_fn)
test_priors_dataloader = DataLoader(test_priors_dataset, batch_size=batch_size, collate_fn=priors_collate_fn)

# from pyhealth.models import Transformer
from gctpyhealth.gct import GCT

# hyperparameters
learning_rate = 0.00011
reg_coef = 1.5
hidden_dropout = 0.72
post_mlp_dropout = 0.005

model = GCT(
    dataset=eicu_ds,
    feature_keys=['conditions_hash', 'procedures_hash'],
    label_key="expired",
    mode="binary",
    hidden_dropout=hidden_dropout,
    reg_coef=reg_coef
)

from gctpyhealth.trainer import Trainer

trainer = Trainer(model=model)
trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=eval_dataloader,
    test_dataloader=test_dataloader,
    train_prior_dataloader=train_priors_dataloader,
    val_prior_dataloader=eval_priors_dataloader,
    test_prior_dataloader=test_priors_dataloader,
    epochs=1000,
    monitor="pr_auc",
    optimizer_class=torch.optim.Adamax,
    optimizer_params=dict(lr=learning_rate)
)

# option 1: use our built-in evaluation metric
score = trainer.evaluate(train_dataloader, train_priors_dataloader)
print(score)

# option 2: use our pyhealth.metrics to evaluate
from pyhealth.metrics.binary import binary_metrics_fn
import numpy as np

y_true, y_prob, loss = trainer.inference(test_dataloader, test_priors_dataloader)
# TODO: hacks, interpreting the output as binary classification
y_prob = np.argmax(y_prob, axis=1)

binary_metrics_fn(y_true, y_prob, metrics=["pr_auc", "roc_auc", "f1"])
