# iiksiin

> **iiksiin** *ӣксӣн* /'iːk.siːn/ — a detached piece of ice (p.147, *Badten et al, 2008*)

Deterministically constructs a sequence of morpheme tensors from a word using Tensor Product Representation

## Example usage

Input file:
```
Upévare Saúl heʼi imoirũhára>pe : “ e>nohẽ nde kyse puku ha che juka , ani ou umi tekove ahẽ ha o>ñemboharái che rehe . 
” Iñirũ katu okyhyje ha nd>o>japosé>i . upémarõ Saúl o>hekýi ikyse puku ha o>jeity hiʼári . 
```
To build tensors with the default parameters, run:
```bash
./iiksiin.sh bible.txt guarani.tensors

```

To build alphabet, you could use `alphabet.py` script. It accepts the following mandatory arguments:
* `-i` — input file containing whitespace delimited words (- for standard input).
* `-o` — output file where pickled alphabet is dumped.
* `--desciption` — description of the alphabet. Will serve as the name of the alphabet.
* `--log` — log file.

Example usage:
```bash
alphabet.py -i bible.txt -o alphabet --description "description" --log log
```

To build tensors, use `corpus2tensors.py` script. It accepts a corpus as the standard input the following mandatory arguments:
* `-a` — Python pickle file containing an Alphabet object.
* `-o` — output file where morpheme tensors are recorded.
* `-d` — in the user-provided input file, this character must appear between adjacent morphemes. This symbol must not appear in the alphabet.

Example usage:
```bash
cat bible.txt | python3 corpus2tensors.py -d '>' -a alphabet -o guarani.tensors
```


## How to use

* Start with a file called prefix.txt, where you replace prefix with something meaningful to you. This file should contain whitespace-separated words, where each word is composed of one or more morphemes\
, with morphemes separated by a morpheme boundary.

* Run `make prefix.tensors` to construct TPR representation of each morpheme.

* Run `make prefix.model` to train an autoencoder over those TPR representations.

* Run `make prefix.vectors` to extract vector representations of each morpheme from the trained autoencoder's final hidden layer

* Run `make prefix.test` to reconstruct surface strings from the vector representations.
