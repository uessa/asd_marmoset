#!/bin/sh

python data_randomize.py \
       "/datanet/users/hkawauchi/vad_marmoset/raw/ayapyon_phee_fftlen1024_wav" \
       "/datanet/users/hkawauchi/vad_marmoset/datasets/subset_ayapyon_phee_fftlen1024/raw" \
       0.7 0.1 0.2 \
       -l 1
