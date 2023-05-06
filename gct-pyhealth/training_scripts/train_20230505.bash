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
  --max_steps $MAX_STEP

# ADAMMAX with scheduler
LABEL_KEY='expired'
LR=0.00011
PDROPOUT=0.72
MAX_STEP=1000000
python auto_train.py \
  --label_key $LABEL_KEY \
  --learning_rate $LR \
  --post_mlp_dropout $PDROPOUT \
  --num_stacks 2 \
  --num_heads 1 \
  --max_steps $MAX_STEP


# ADAMMAX with scheduler
LABEL_KEY='expired'
LR=0.00011
PDROPOUT=0.72
MAX_STEP=1000000
python auto_train.py \
  --label_key $LABEL_KEY \
  --learning_rate $LR \
  --post_mlp_dropout $PDROPOUT \
  --num_stacks 3 \
  --num_heads 1 \
  --max_steps $MAX_STEP


# eval aucpr	eval auroc	test aucpr	test auroc
# model1	0.5655	0.769	0.6061	0.782
# model2	0.5583	0.7974	0.5837	0.8146
# model3	0.5662	0.7716	0.5971	0.7821
