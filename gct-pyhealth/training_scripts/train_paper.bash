# Using the hyperparameters stated in the paper Table 6:

export DATA_DIR='eicu_data'

# readmission prediction on eICU
# Learning Rate: 0.00022
# MLP dropoutrate: 0.08
# Post-MLP dropout rate: 0.024
# Regularization coef.: 0.1

# evaluation
# Validation AUCPR: 0.5313 (0.0124)
# Test AUCPR: 0.5244 (0.0142)
# Validation AUROC: 0.7525 (0.0128)
# Test AUROC: 0.7502 (0.0114)

LR=0.00022
DROPOUT=0.08
REG_COEF=0.1
LABEL_KEY='readmission'
OUTPUT_DIR='eicu_output/model_'${LR}'_'${DROPOUT}'_'${LABEL_KEY}

mkdir -p $OUTPUT_DIR
python train.py \
    --data_dir $DATA_DIR \
    --fold 0 \
    --output_dir $OUTPUT_DIR \
    --use_prior \
    --use_guide \
    --output_hidden_states \
    --output_attentions \
    --do_train \
    --do_eval \
    --do_test \
    --label_key $LABEL_KEY \
    --max_steps 1000000 \
    --hidden_dropout_prob $DROPOUT \
    --num_stacks 2 \
    --learning_rate $LR \
    --reg_coef $REG_COEF

# mortality prediction on eICU
# Learning Rate: 0.00011
# MLP dropoutrate: 0.72
# Post-MLP dropout rate: 0.005
# Regularization coef.: 1.5

# evaluation
# Validation AUCPR: 0.6196 (0.0259)
# Test AUCPR: 0.5992 (0.0223)
# Validation AUROC: 0.9089 (0.0052)
# Test AUROC: 0.9120 (0.0048)

LR=0.00011
DROPOUT=0.72
REG_COEF=1.5
LABEL_KEY='expired'
OUTPUT_DIR='eicu_output/model_'${LR}'_'${DROPOUT}'_'${LABEL_KEY}

mkdir -p $OUTPUT_DIR
python train.py \
    --data_dir $DATA_DIR \
    --fold 0 \
    --output_dir $OUTPUT_DIR \
    --use_prior \
    --use_guide \
    --output_hidden_states \
    --output_attentions \
    --do_train \
    --do_eval \
    --do_test \
    --label_key $LABEL_KEY \
    --max_steps 1000000 \
    --hidden_dropout_prob $DROPOUT \
    --num_stacks 2 \
    --learning_rate $LR \
    --reg_coef $REG_COEF