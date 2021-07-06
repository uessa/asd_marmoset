#!/bin/sh

python makedata.py "subset_ayapyon_phee" \
       -r 96000 \
       -l 2048 \
       -s 1024 \
       -w hamming
