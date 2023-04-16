# Using the hyperparameters stated in the paper Table 6:
# Explaination: the transformer basically is GCT without M (the -inf guided mask)
#               , the conditional probability P (prior) and regularization term (aka reg coeff = 0)

export DATA_DIR='eicu_data'

LR=0.0006
DROPOUT=0.88
LABEL_KEY='expired'
OUTPUT_DIR='eicu_output/transformer_model_'${LR}'_'${DROPOUT}'_'${LABEL_KEY}

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
    --label_key $LABEL_KEY \
    --max_steps 1000000 \
    --hidden_dropout_prob $DROPOUT \
    --num_stacks 2 \
    --learning_rate $LR