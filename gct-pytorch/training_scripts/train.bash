export DATA_DIR='eicu_data'
# export CUDA_VISIBLE_DEVICES="2"
LABEL_KEY=readmission

for LR in 1e-3 ; do
    for DROPOUT in 0.5 0.6 0.7; do
        OUTPUT_DIR='eicu_output/model_'${LR}'_'${DROPOUT}
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
            --learning_rate $LR
    done
done
