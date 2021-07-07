#!/bin/sh

python makedata.py "subset_ayapyon_vad_fftlen1024" \
       -r 96000 \
       -l 1024 \
       -s 512 \
       -w hamming
