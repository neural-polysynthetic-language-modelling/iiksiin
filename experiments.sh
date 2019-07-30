#!/bin/bash
# run everything from the topmost repo folder #

#set virtualenv
source /opt/python/3.7/venv/pytorch1.0_cuda10.0/bin/activate` # use the same virtualenv for the whole file

###ESS, FULL BIBLE, NO OOV###
# get morpheme-delimited text #
python src/ess/no_oov/data/scripts/ess_fst_char_preprocess.py -ip src/ess/no_oov/data/raw/all.train.analyzed.ess -op src/ess/data/no_oov/preprocessed/all.train.no_oov.segmented.ess.txt
python src/ess/no_oov/data/scripts/ess_fst_char_preprocess.py -ip src/ess/no_oov/data/raw/all.test.analyzed.ess  -op src/ess/data/no_oov/preprocessed/all.test.no_oov.segmented.ess.txt
python src/ess/no_oov/data/scripts/ess_fst_char_preprocess.py -ip src/ess/no_oov/data/raw/all.valid.analyzed.ess -op src/ess/data/no_oov/preprocessed/all.valid.no_oov.segmented.ess.txt

# create alphabet and deterministic morphemes #
python src/iiksiin.py --morpheme_delimiter "^" --input_file src/ess/no_oov/data/preprocessed/all.train.no_oov.segmented.ess.txt --output_file src/ess/no_oov/all.train.no_oov.ess.tensors --alphabet_output /ess/no_oov/all.train.no_oov.ess.alphabet

# run autoencoder for the different vector sizes #
python src/autoencoder.py --mode train --tensor_file src/ess/no_oov/all.train.no_oov.ess.tensors  --output_file src/ess/no_oov/64/all.train.no_oov.ess.64.autoencoder  --cuda_device 0 --batch_size 100 --epochs 200 --hidden_layer_size 64 --learning_rate 0.01 --hidden_layers 1 &
python src/autoencoder.py --mode train --tensor_file src/ess/no_oov/all.train.no_oov.ess.tensors  --output_file src/ess/no_oov/128/all.train.no_oov.ess.128.autoencoder --cuda_device 1 --batch_size 100 --epochs 200 --hidden_layer_size 128 --learning_rate 0.01 --hidden_layers 1 &
python src/autoencoder.py --mode train --tensor_file src/ess/no_oov/all.train.no_oov.ess.tensors  --output_file src/ess/no_oov/256/all.train.no_oov.ess.256.autoencoder --cuda_device 2 --batch_size 100 --epochs 200 --hidden_layer_size 256 --learning_rate 0.01 --hidden_layers 1 &
python src/autoencoder.py --mode train --tensor_file src/ess/no_oov/all.train.no_oov.ess.tensors  --output_file src/ess/no_oov/512/all.train.no_oov.ess.512.autoencoder --cuda_device 3 --batch_size 100 --epochs 200 --hidden_layer_size 512 --learning_rate 0.01 --hidden_layers 1

# generate the vectors from the autoencoder models #
python src/autoencoder.py --mode t2v --tensor_file src/ess/no_oov/all.train.no_oov.ess.tensors  --model_file src/ess/no_oov/autoencoder/64/all.train.no_oov.ess.64.autoencoder   --output_file src/ess/no_oov/autoencoder/64/all.train.no_oov.ess.64.vectors   --cuda_device 0 --batch_size 100 &
python src/autoencoder.py --mode t2v --tensor_file src/ess/no_oov/all.train.no_oov.ess.tensors  --model_file src/ess/no_oov/autoencoder/128/all.train.no_oov.ess.128.autoencoder --output_file src/ess/no_oov/autoencoder/128/all.train.no_oov.ess.128.vectors --cuda_device 1 --batch_size 100 &
python src/autoencoder.py --mode t2v --tensor_file src/ess/no_oov/all.train.no_oov.ess.tensors  --model_file src/ess/no_oov/autoencoder/256/all.train.no_oov.ess.256.autoencoder --output_file src/ess/no_oov/autoencoder/256/all.train.no_oov.ess.256.vectors --cuda_device 2 --batch_size 100 &
python src/autoencoder.py --mode t2v --tensor_file src/ess/no_oov/all.train.no_oov.ess.tensors  --model_file src/ess/no_oov/autoencoder/512/all.train.no_oov.ess.512.autoencoder --output_file src/ess/no_oov/autoencoder/512/all.train.no_oov.ess.512.vectors --cuda_device 3 --batch_size 100

# train the seq2seq model on the 64 dim vectors #
python src/char2morph.py --train_file       src/ess/data/no_oov/preprocessed/all.train.no_oov.segmented.ess.txt  \
                         --dev_file          src/ess/data/no_oov/preprocessed/all.test.no_oov.segmented.ess.txt  \
                         --test_file         src/ess/data/no_oov/preprocessed/all.valid.no_oov.segmented.ess.txt \
                         --vector_file       grn.vectors                                                         \
                         --alphabet          grn.alphabet                                                        \
                         --lr                0.01                                                                \
                         --autoencoder_model grn.trained_model                                                   \
                         --epochs            300                                                                 \
                         --batch_size        320

mv .save/char2morph_300.pt src/ess/no_oov/char2morph.no_oov.trained.300_epochs.model.pt
