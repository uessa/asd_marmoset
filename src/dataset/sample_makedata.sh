#!/bin/sh

python makedata.py "subset_marmoset_23ue_muesaka" \
       -r 96000 \
       -l 2048 \
       -s 1024 \
       -w hamming
