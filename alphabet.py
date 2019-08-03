#!/usr/bin/env python3.7

import grapheme # type: ignore
import logging
import pickle
import sys
from typing import List, Mapping, Set, Iterable, Iterator
import unicodedata

"""Implements a character alphabet for use in the 
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


class Alphabet:

    END_OF_MORPHEME: str = "\u0000"
    END_OF_TRANSMISSION: str = "\u0004"

    def __init__(self, name: str, symbols: Set[str]):
        alphabet_symbols = set(symbols)
        alphabet_symbols.add(Alphabet.END_OF_MORPHEME)
        alphabet_symbols.add(Alphabet.END_OF_TRANSMISSION)
        self._symbols: Mapping[str, int] = {
            symbol: index for (index, symbol) in enumerate(sorted(alphabet_symbols), start=1)
        }
        from iiksiin import Dimension
        self.dimension: Dimension = Dimension(name, 1 + len(alphabet_symbols))
        self.name = name
        self.oov = 0
        from iiksiin import Vector, OneHotVector
        self._vector: List[Vector] = list()
        for i in range(len(self.dimension)):
            self._vector.append(OneHotVector(i, self.dimension))

    def get_vector(self, symbol: str) -> "Vector":
        index: int = self[symbol]
        return self._vector[index]

    def number_of_symbols(self) -> int:
        return len(self._symbols)
    
    def __iter__(self) -> Iterator[str]:
        return iter(self._symbols.keys())

    def __getitem__(self, symbol: str) -> int:
        if symbol in self._symbols:
            return self._symbols[symbol]
        else:
            return self.oov

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Alphabet({str(self.dimension)})"

    @staticmethod
    def char_to_code_point(c: str) -> str:
        x_hex_string: str = hex(ord(c))  # a string of the form "0x95" or "0x2025"
        hex_string: str = x_hex_string[2:]  # a string of the form "95" or "2025"
        required_zero_padding = max(0, 4 - len(hex_string))
        return (
            f"U+{required_zero_padding * '0'}{hex_string}"
        )  # a string of the form "\\u0095" or "\\u2025"

    @staticmethod
    def char_to_name(c: str) -> str:
        try:
            return unicodedata.name(c)
        except ValueError:
            return ""

    @staticmethod
    def unicode_info(s: str) -> str:
        return (
            s
            + "\t"
            + "; ".join(
                [
                    f"{Alphabet.char_to_code_point(c)} {Alphabet.char_to_name(c)}"
                    for c in s
                ]
            )
        )

    @staticmethod
    def create_from_source(name: str, source: Iterable[str], morpheme_delimiter: str, end_of_morpheme_symbol: str, blacklist_char: str) -> "Alphabet":
        alphabet_set: Set[str] = set()
        for line in source:
            for character in grapheme.graphemes(line.strip()):
                category = unicodedata.category(character)
                if category[0] != "Z" and category[0] != "C" and character != morpheme_delimiter and character != end_of_morpheme_symbol and character != blacklist_char:
                    alphabet_set.add(character)

        alphabet_set.add(end_of_morpheme_symbol)

        for symbol in alphabet_set:
            for character in symbol:
                category = unicodedata.category(character)
                if category[0] == "Z":
                    logging.warning(f"WARNING - alphabet contains whitespace character:\t{Alphabet.unicode_info(symbol)}")

                elif (category[0] == "C" and character != morpheme_delimiter and character != end_of_morpheme_symbol):
                    logging.warning(f"WARNING - alphabet contains control character:\t{Alphabet.unicode_info(symbol)}")

        return Alphabet(name, alphabet_set)

    
def main(name: str, input_source: Iterable[str], output_filename: str, log_filename: str, morpheme_delimiter: str, end_of_morpheme_symbol: str, blacklist_char: str) -> None:

    alphabet: Alphabet = Alphabet.create_from_source(name, input_source, morpheme_delimiter, end_of_morpheme_symbol, blacklist_char)

    with open(log_filename, "wt") as log:
        print(f"Symbols in alphabet: {alphabet.number_of_symbols()}", file=log)
        print("-----------------------", file=log)
        for symbol in sorted(iter(alphabet)):
            print(Alphabet.unicode_info(symbol), file=log)
    
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
        blacklist_char=args.blacklist_char
    )
