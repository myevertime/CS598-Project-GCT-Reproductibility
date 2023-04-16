export DATA_DIR='eicu_data'
# export CUDA_VISIBLE_DEVICES="2"

LR=2e-3
DROPOUT=0.5
REG_COEF=0.1
LABEL_KEY='readmission'
OUTPUT_DIR='eicu_output/model_'${LR}'_'${DROPOUT}'_'${LABEL_KEY}

mkdir -p $OUTPUT_DIR
python train_tensorboard.py \
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
    --num_stacks 4 \
    --learning_rate $LR \
    --reg_coef $REG_COEF \
    --batch_size 256
