SHELL := /bin/bash

VENV=/opt/python/3.7/venv/pytorch1.0_cuda10.0

%.alphabet: %.txt
	source ${VENV}/bin/activate && python3 alphabet.py --description $* --morpheme_delimiter ">" --end_of_morpheme_symbol '\u0000' --input_file $< --blacklist_char '*' --output_file $@ --log $@.log

%.trained_model: %.tensors
	source ${VENV}/bin/activate && python3 autoencoder.py --mode train -a ess.alphabet --tensor_file $*.tensors --output_file $@ --cuda_device 1 --batch_size 100 --epochs 200 --hidden_layer_size 128 --learning_rate 0.01 --hidden_layers 1

%.tensors: %.txt
	source ${VENV}/bin/activate && python3 iiksiin.py -d "^" -i $< -o $@ -c 23 --alphabet_output ess.alphabet

%.vectors: %.trained_model %.tensors
	source ${VENV}/bin/activate && python3 autoencoder.py --mode t2v --tensor_file $*.tensors --model_file $*.trained_model --cuda_device 1 --batch_size 100 --output_file $@ -a ess.alphabet.pkl

%.test: %.trained_model %.tensors %.vectors
	source ${VENV}/bin/activate && python3 autoencoder.py --mode v2s --tensor_file $*.tensors --model_file $*.trained_model --vector_file $*.vectors --batch_size 100 --output_file $@ --cuda_device 1 -a ess.alphabet.pkl
#source ${VENV}/bin/activate && ./autoencoder.py --mode tv2s --tensor_file $*.tensors --model_file $*.trained_model --cuda_device 1 --batch_size 100 --output_file $@ > $@

%.clean:
	rm -f $*.trained_model $*.tensors $*.test

ess.morpheme_info.pickle: finite_state_morphology/lexicon/lexicon.py finite_state_morphology/ess.lexc
	source ${VENV}/bin/activate && python3 finite_state_morphology/lexicon/lexicon.py --mode l2p --lexc finite_state_morphology/ess.lexc --output ess.morpheme_info.pickle

.PRECIOUS: %.tensors %.test %.trained_model
.PHONY: %.clean always

always:
	source ${VENV}/bin/activate && ./autoencoder.py --mode t2v --tensor_file grn.tensors --model_file grn.trained_model --cuda_device 1 --batch_size 100 --output_file always.vectors
