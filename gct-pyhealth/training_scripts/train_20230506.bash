# ADAMMAX with scheduler
LABEL_KEY='readmission'
LR=0.00022
PDROPOUT=0.5
MAX_STEP=1000000
python auto_train.py \
  --label_key $LABEL_KEY \
  --learning_rate $LR \
  --post_mlp_dropout $PDROPOUT \
  --num_stacks 3 \
  --num_heads 1 \
  --max_steps $MAX_STEP

# ADAMMAX with scheduler
LABEL_KEY='readmission'
LR=0.00022
PDROPOUT=0.5
MAX_STEP=1000000
python auto_train.py \
  --label_key $LABEL_KEY \
  --learning_rate $LR \
  --post_mlp_dropout $PDROPOUT \
  --num_stacks 3 \
  --num_heads 2 \
  --max_steps $MAX_STEP

# ADAMMAX with scheduler
LABEL_KEY='readmission'
LR=0.00022
PDROPOUT=0.5
MAX_STEP=1000000
python auto_train.py \
  --label_key $LABEL_KEY \
  --learning_rate $LR \
  --post_mlp_dropout $PDROPOUT \
  --num_stacks 2 \
  --num_heads 1 \
  --max_steps $MAX_STEP
