
# ADAM with no scheduler
LABEL_KEY='expired'
LR=0.00011
python auto_train_no_scheduler.py \
   --label_key $LABEL_KEY \
   --learning_rate $LR \
   --num_stacks 3 \
   --num_heads 1

# ADAM with no scheduler
LABEL_KEY='readmission'
LR=0.00022
python auto_train_no_scheduler.py \
   --label_key $LABEL_KEY \
   --learning_rate $LR \
   --num_stacks 3 \
   --num_heads 1

# ADAMMAX with scheduler, same learning rate
LABEL_KEY='expired'
LR=0.00011
python auto_train.py \
  --label_key $LABEL_KEY \
  --learning_rate $LR \
  --num_stacks 3 \
  --num_heads 1

# ADAMMAX with scheduler, same learning rate
LABEL_KEY='readmission'
LR=0.00022
python auto_train.py \
  --label_key $LABEL_KEY \
  --learning_rate $LR \
  --num_stacks 3 \
  --num_heads 1

# ADAMMAX with scheduler, higher learning rate
#LABEL_KEY='readmission'
#LR=0.001
#python auto_train.py \
#  --label_key $LABEL_KEY \
#  --learning_rate $LR \
#  --num_stacks 3 \
#  --num_heads 1
