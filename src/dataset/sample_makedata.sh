#!/bin/sh

python makedata.py "subset_ayapyon_calltype_fftlen4096" \
       -r 96000 \
       -l 4096 \
       -s 2048 \
       -w hamming
