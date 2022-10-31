#!/bin/sh

nohup python train.py \
       subset_marmoset_23ue \
       --batch_size 2 \
       --lr 0.01 \
       > out.log \
       2> err.log \
       < /dev/null \
       & \