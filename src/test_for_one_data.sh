#!/bin/sh

# nohup python test_for_one_data.py \
python test_for_one_data.py \
       subset_marmoset_23ue_muesaka test \
       --model ../models/subset_marmoset_23ue_muesaka/trial02/model.pth \
       --batch_size 1
       # --batch_size 1 \
       # > out.log \
       # 2> err.log \
       # &