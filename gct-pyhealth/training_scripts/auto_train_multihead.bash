# LABEL_KEY='expired'

# python auto_train.py \
#     --label_key $LABEL_KEY \
#     --num_stacks 2 \
#     --num_heads 1


# python auto_train.py \
#     --label_key $LABEL_KEY \
#     --num_stacks 3 \
#     --num_heads 2


LABEL_KEY='readmission'

python auto_train.py \
    --label_key $LABEL_KEY \
    --num_stacks 2 \
    --num_heads 1

python auto_train.py \
    --label_key $LABEL_KEY \
    --num_stacks 2 \
    --num_heads 2

python auto_train.py \
    --label_key $LABEL_KEY \
    --num_stacks 3 \
    --num_heads 2
