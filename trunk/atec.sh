#!/bin/bash
# make
./word2vec -train train -output vectors.txt -cbow 1 -size 100 -window 8 -negative 5 -hs 0 -sample 1e-4 -threads 80 -binary 0 -iter 200 -min-count 0
