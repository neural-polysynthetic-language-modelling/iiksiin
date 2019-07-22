SHELL := /bin/bash

%.trained_model: %.tensors
	source /opt/python/3.7/venv/pytorch1.0_cuda10.0/bin/activate && ./autoencoder.py --mode train --tensor_file $< --output_file $@ --cuda_device 1 --batch_size 100 --epochs 200 --hidden_layer_size 200 --learning_rate 0.01 --hidden_layers 1


%.tensors: %.txt
	source /opt/python/3.7/venv/pytorch1.0_cuda10.0/bin/activate && python3 iiksiin.py -d ">" -i $< -o $@

%.vectors: %.trained_model %.tensors
	source /opt/python/3.7/venv/pytorch1.0_cuda10.0/bin/activate && ./autoencoder.py --mode t2v --tensor_file $*.tensors --model_file $*.trained_model --cuda_device 1 --batch_size 100 --output_file $@

%.test: %.trained_model %.tensors %.vectors
	source /opt/python/3.7/venv/pytorch1.0_cuda10.0/bin/activate && ./autoencoder.py --mode v2s --tensor_file $*.tensors --model_file $*.trained_model --vector_file $*.vectors --cuda_device 1 --batch_size 100 --output_file $@
#source /opt/python/3.7/venv/pytorch1.0_cuda10.0/bin/activate && ./autoencoder.py --mode tv2s --tensor_file $*.tensors --model_file $*.trained_model --cuda_device 1 --batch_size 100 --output_file $@ > $@

%.clean:
	rm -f $*.trained_model $*.tensors $*.test


.PRECIOUS: %.tensors %.test %.trained_model
.PHONY: %.clean always

always:
	source /opt/python/3.7/venv/pytorch1.0_cuda10.0/bin/activate && ./autoencoder.py --mode t2v --tensor_file grn.tensors --model_file grn.trained_model --cuda_device 1 --batch_size 100 --output_file always.vectors
