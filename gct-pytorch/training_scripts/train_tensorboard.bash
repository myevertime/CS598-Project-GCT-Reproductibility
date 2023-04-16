export DATA_DIR='eicu_data'
# export CUDA_VISIBLE_DEVICES="2"

LR=1e-3
DROPOUT=0.1
REG_COEF=1.5
LABEL_KEY='expired'
OUTPUT_DIR='eicu_output/tb_model_'${LR}'_'${DROPOUT}'_'${LABEL_KEY}

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
    --max_steps 1000 \
    --hidden_dropout_prob $DROPOUT \
    --num_stacks 4 \
    --learning_rate $LR \
    --reg_coef $REG_COEF \
    --batch_size 64
