SHELL := /bin/bash

PYTHON=/usr/bin/python3

%.alphabet: %
	${PYTHON} alphabet.py -i $< -o $*.alphabet --description $* --log $@.log

%.tensors: %
	cat $* | ${PYTHON} corpus2tensors.py -d '>' -a $*.alphabet -o $*.tensors

%.model: %.tensors
	${PYTHON} autoencoder.py --mode train --tensor_file $*.tensors --output_file $@ --cuda_device 0 --batch_size 100 --hidden_layer_size 64 --learning_rate 0.01 --hidden_layers 1 -a $*.alphabet --epochs 500

%.vectors: %.model %.tensors
	${PYTHON} autoencoder.py --mode t2v --tensor_file $*.tensors --model_file $*.model --cuda_device 0 --batch_size 100 --output_file $@ -a $*.alphabet

%.test: %.model %.tensors %.vectors
	${PYTHON} autoencoder.py --mode v2s --tensor_file $*.tensors --model_file $*.model --vector_file $*.vectors --batch_size 100 --cuda_device 0 --output_file $@ -a alphabet

%.clean:
	rm -f $*.model $*.tensors $*.test

