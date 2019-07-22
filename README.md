# iiksiin

> **iiksiin** *ӣксӣн* /'iːk.siːn/ — a detached piece of ice (p.147, *Badten et al, 2008*)

Deterministically constructs a sequence of morpheme tensors from a word using Tensor Product Representation

## Example usage

Input file:
```
Upévare Saúl heʼi imoirũhára>pe : “ e>nohẽ nde kyse puku ha che juka , ani ou umi tekove ahẽ ha o>ñemboharái che rehe . 
” Iñirũ katu okyhyje ha nd>o>japosé>i . upémarõ Saúl o>hekýi ikyse puku ha o>jeity hiʼári . 
```
Run:
```bash
cat bible.txt |python3 iiksiin.py -d '>' -a alphabet.txt -o guarani.tensors

```


## How to use

* Start with a file called prefix.txt, where you replace prefix with something meaningful to you. This file should contain whitespace-separated words, where each word is composed of one or more morphemes\
, with morphemes separated by a morpheme boundary.

* Run `make prefix.tensors` to construct TPR representation of each morpheme.

* Run `make prefix.model` to train an autoencoder over those TPR representations.

* Run `make prefix.vectors` to extract vector representations of each morpheme from the trained autoencoder's final hidden layer

* Run `make prefix.test` to reconstruct surface strings from the vector representations.
