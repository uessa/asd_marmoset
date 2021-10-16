#!/bin/sh

python data_randomize.py \
       "/datanet/users/hkawauchi/vad_marmoset/raw/test" \
       "/datanet/users/hkawauchi/vad_marmoset/datasets/subset_test/raw" \
       0.7 0.1 0.2 \
       -l 1
