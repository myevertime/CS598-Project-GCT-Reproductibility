# ADAMMAX with scheduler
LABEL_KEY='expired'
LR=0.00011
PDROPOUT=0.5
MAX_STEP=1000000
python auto_train.py \
  --label_key $LABEL_KEY \
  --learning_rate $LR \
  --post_mlp_dropout $PDROPOUT \
  --num_stacks 3 \
  --num_heads 1 \
  --max_steps $MAX_STEP \
  --load_prev_model \
  --prev_model_path 'eicu_output/model_pyhealth_expired_2023-05-04_19-16-05/model.pt'

# ADAMMAX with scheduler
LABEL_KEY='expired'
LR=0.00011
PDROPOUT=0.5
MAX_STEP=1000000
python auto_train.py \
  --label_key $LABEL_KEY \
  --learning_rate $LR \
  --post_mlp_dropout $PDROPOUT \
  --num_stacks 3 \
  --num_heads 2 \
  --max_steps $MAX_STEP