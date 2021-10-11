#!/usr/bin/env python3.7

import sys

delimiter="\u241E"

def extract(word):
    if word.startswith("*"):
        return word
    elif delimiter in word:
        return word.split(delimiter)[0]
    else:
        return word


for line in sys.stdin:
    words = line.strip().split()
    results = " ".join([extract(word) for word in words])
    print(results)
