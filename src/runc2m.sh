#!/bin/bash

source /opt/python/3.7/venv/pytorch1.0_cuda10.0/bin/activate && python3.7 char2morph.py  \
									  --train_file        ../nplm/data/Condition.all/train \
									  --dev_file          ../nplm/data/Condition.all/dev   \
									  --test_file         ../nplm/data/Condition.all/test  \
									  --vector_file       grn.vectors                 \
									  --alphabet          grn.alphabet                \
									  --lr                0.0001                      \
									  --autoencoder_model grn.trained_model           \
									  --epochs            300                         \
									  --char2morph_model  grn.char2morph_debug.pt
