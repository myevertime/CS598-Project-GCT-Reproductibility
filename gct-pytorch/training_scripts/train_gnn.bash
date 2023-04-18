# Using the hyperparameters stated in the paper Table 6:
# Explaination: the transformer basically is GCT without M (the -inf guided mask)
#               , the conditional probability P (prior) and regularization term (aka reg coeff = 0)

export DATA_DIR='eicu_data'

LR=0.00015
MLP_DROPOUT=0.01
POS_MLP_DROPOUT=0.01
LABEL_KEY='expired'
OUTPUT_DIR='eicu_output/gnn_model_'${LR}'_'${DROPOUT}'_'${LABEL_KEY}

#    --use_prior \
#    --use_guide \
#    --reg_coef $REG_COEF

mkdir -p $OUTPUT_DIR
python train.py \
    --data_dir $DATA_DIR \
    --fold 0 \
    --output_dir $OUTPUT_DIR \
    --output_hidden_states \
    --output_attentions \
    --do_train \
    --do_eval \
    --do_test \
    --use_prior \
    --label_key $LABEL_KEY \
    --max_steps 1000000 \
    --hidden_dropout_prob $MLP_DROPOUT \
    --post_mlp_dropout_rate $POS_MLP_DROPOUT \
    --num_stacks 1 \
    --learning_rate $LR