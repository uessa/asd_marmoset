#!/bin/sh

python data_randomize.py \
       "/home/muesaka/projects/hkawauchi/marmoset/raw/marmoset_11vpa_wav" \
       "/home/muesaka/projects/hkawauchi/marmoset/datasets/subset_marmoset_11vpa/raw" \
       0.7 0.1 0.2 \
       -l 1
