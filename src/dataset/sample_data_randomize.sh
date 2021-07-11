#!/bin/sh

python data_randomize.py \
       "/datanet/users/hkawauchi/vad_marmoset/raw/ayapyon_calltype_fftlen16384_wav" \
       "/datanet/users/hkawauchi/vad_marmoset/datasets/subset_ayapyon_calltype_fftlen16384/raw" \
       0.7 0.1 0.2 \
       -l 1
