#!/usr/bin/env python3.7

from iiksiin import Alphabet, Dimension
import grapheme # type: ignore
import logging
import pickle
import sys
from typing import List, Mapping, Set, Iterable, Iterator
import unicodedata

"""Constructs a character alphabet for use in the 
Tensor Product Representation for potentially multi-morphemic words.

This file was developed as part of the Neural Polysynthetic Language Modelling project
at the 2019 Frederick Jelinek Memorial Summer Workshop at École de Technologie Supérieure in Montréal, Québec, Canada.
https://www.clsp.jhu.edu/workshops/19-workshop/
"""

__author__ = "Lane Schwartz"
__copyright__ = "Copyright 2019, Lane Schwartz"
__license__ = "MPL 2.0"
__credits__ = [
    "Lane Schwartz",
    "Coleman Haley",
    "Francis Tyers",
    "JSALT 2019 NPLM team members",
]
__maintainer = "Lane Schwartz"
__email__ = "dowobeha@gmail.com"
__version__ = "0.0.1"
__status__ = "Prototype"


if sys.version_info < (3, 7):
    raise RuntimeError(f"{__file__} requires Python 3.7 or later")

    
def main(name: str, input_source: Iterable[str], output_filename: str, log_filename: str, morpheme_delimiter: str, end_of_morpheme_symbol: str, padding_symbol: str, blacklist_char: str) -> None:

    alphabet: Alphabet = Alphabet.create_from_source(name, input_source, morpheme_delimiter, end_of_morpheme_symbol, padding_symbol, blacklist_char)

    with open(log_filename, "wt") as log:
        print(f"Symbols in alphabet: {alphabet.number_of_symbols()}", file=log)
        print("-----------------------", file=log)
        print(f"0\t\t\tThe integer value 0 is reserved to represent any symbol not in the alphabet", file=log)
        for symbol in sorted(iter(alphabet)):
            message=f"{alphabet[symbol]}\t{Alphabet.unicode_info(symbol)}\t"
            if symbol==alphabet.end_of_morpheme_symbol:
                message += "End-of-morpheme symbol"
            if symbol==alphabet.padding_symbol:
                message += "Padding symbol"
            print(message, file=log)
    
    logging.info(f"Writing alphabet object as pickle to {output_filename}")
    with open(output_filename, "wb") as output:
        pickle.dump(alphabet, output)


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Construct alphabet for use in tensor product representations of morphemes."
    )
    arg_parser.add_argument(
        "--description",
        metavar="string",
        type=str,
        required=True,
        help="Description of the alphabet. Will serve as the name of the alphabet."
    )
    arg_parser.add_argument(
        "-d",
        "--morpheme_delimiter",
        metavar="string",
        type=str,
        nargs="?",
        default=">",
        help="In the user-provided input file, "
        + "this character must appear between adjacent morphemes. "
        + "This symbol must not appear in the alphabet",
    )
    arg_parser.add_argument(
        "-e",
        "--end_of_morpheme_symbol",
        metavar="character",
        type=str,
        nargs="?",
        default="\\u0000",
        help="In this output tensor representation, "
        + "this character will be appended as the final symbol in every morpheme. "
        + "This symbol must not appear in the alphabet",
    )
    arg_parser.add_argument(
        "-p",
        "--padding_symbol",
        metavar="character",
        type=str,
        nargs="?",
        default="\\u0004",
        help="This character will be used when padding is needed in a tensor. "
        + "This symbol must not appear in the alphabet",
    )
    arg_parser.add_argument(
        "-i",
        "--input_file",
        metavar="filename",
        type=str,
        required=True,
        help="Input file containing whitespace delimited words (- for standard input)",
    )
    arg_parser.add_argument(
        "--blacklist_char",
        metavar="filename",
        type=str,
        nargs="?",
        default="*",
        help="Character that marks unanalyzed words that should be ignored",
    )
    arg_parser.add_argument(
        "-o",
        "--output_file",
        metavar="filename",
        type=str,
        nargs="?",
        required=True,
        help="Output file where pickled alphabet is dumped",
    )
    arg_parser.add_argument(
        "--log",
        metavar="filename",
        type=str,
        nargs="?",
        required=True,
        help="Log file"
    )
    arg_parser.add_argument("-v", "--verbose", metavar="int", type=int, default=0)

    args = arg_parser.parse_args()

    main(
        name=args.description,
        input_source=open(args.input_file) if args.input_file != "-" else sys.stdin,
        output_filename=args.output_file,
        log_filename=args.log,
        morpheme_delimiter=str.encode(args.morpheme_delimiter).decode("unicode_escape"),
        end_of_morpheme_symbol=str.encode(args.end_of_morpheme_symbol).decode("unicode_escape"),
        padding_symbol=str.encode(args.padding_symbol).decode("unicode_escape"),
        blacklist_char=args.blacklist_char
    )
