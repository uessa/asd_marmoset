#!/bin/sh

# nohup python train.py \
#        subset_test \
#        --batch_size 2 \
#        --lr 0.01 \
#        > out.log \
#        2> err.log \
#        < /dev/null \
#        &

python train.py \
       subset_marmoset_23ue_muesaka_48kHz \
       --batch_size 2 \
       --lr 0.01