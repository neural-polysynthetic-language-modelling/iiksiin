#!/bin/bash
if [[ -n "$1" && -n "$2" ]]
then
    echo "Creating alphabet from $1..."
    python3 alphabet.py -i $1 -o alphabet --description "" --log log
    echo "Alphabet created."
    echo "Creating tensors..."
    cat $1 | python3 corpus2tensors.py -d '>' -a alphabet -o $2
    echo "Tensors are created and saved as $2"
else
    echo "You should use two arguments: for the input and output files."
    echo "Example:"
    echo "./iiksiin.sh bible.txt bible.tensors"
fi
