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
