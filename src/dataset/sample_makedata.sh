#!/bin/sh

python makedata.py "subset_ayapyon_vad_fftlen16384" \
       -r 96000 \
       -l 16384 \
       -s 8192 \
       -w hamming
