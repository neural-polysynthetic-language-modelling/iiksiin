#!/usr/bin/env python3.7

import argparse
import grapheme  # type: ignore
import logging
import sys
from typing import Dict, List, Callable, Tuple, Set, Mapping, Iterable, Iterator
import unicodedata

"""Extracts an alphabet of characters from a corpus.

This file was developed as part of the Neural Polysynthetic Language Modelling project
at the 2019 Frederick Jelinek Memorial Summer Workshop at École de Technologie Supérieure in Montréal, Québec, Canada.
https://www.clsp.jhu.edu/workshops/19-workshop/
"""

__author__ = "Lane Schwartz"
__copyright__ = "Copyright 2019, Lane Schwartz"
__license__ = "MPL 2.0"
__credits__ = [
    "Lane Schwartz",
    "JSALT 2019 NPLM team members",
]
__maintainer = "Lane Schwartz"
__email__ = "dowobeha@gmail.com"
__version__ = "0.0.1"
__status__ = "Prototype"


if sys.version_info < (3, 7):
    raise RuntimeError(f"{__file__} requires Python 3.7 or later")

def escaped_codepoints(s: str) -> str:
    return "".join([f"\\u{char_to_code_point(c)}" for c in s])
    
def char_to_code_point(c: str) -> str:
    x_hex_string: str = hex(ord(c))  # a string of the form "0x95" or "0x2025"
    hex_string: str = x_hex_string[2:]  # a string of the form "95" or "2025"
    required_zero_padding = max(0, 4 - len(hex_string))
    return (
        f"{required_zero_padding * '0'}{hex_string}"
    )  # a string of the form "\\u0095" or "\\u2025"

def char_to_name(c: str) -> str:
    try:
        return unicodedata.name(c)
    except ValueError:
        return ""

def unicode_info(s: str) -> str:
    return "; ".join([f"U+{char_to_code_point(c)} {char_to_name(c)}" for c in s])

def output(output_file, int_value, character, unicode_name, description):
    print(f"{int_value}\t{character}\t{unicode_name}\t{description}", file=output_file)

def main(
    morpheme_delimiter: str,
    end_of_morpheme_symbol: str,
    padding_symbol: str,    
    input_file,
    output_file,
    verbose: int,
    blacklist_char: str,
) -> None:
    
    if grapheme.length(end_of_morpheme_symbol) != 1:
        raise RuntimeError(
            "The end of morpheme symbol must consist of a single grapheme cluster "
            + "(see Unicode Standard Annex #29)."
        )

    alphabet_set: Set[str] = set()
    logging.info(f"Reading alphabet from input file {input_file.name}...")

    for line in input_file:
        for character in grapheme.graphemes(line.strip()):
            category = unicodedata.category(character)
            if category[0] == "Z":
                logging.debug("Input contains whitespace character:\t{unicode_info(symbol)}. This character will not be included in the alphabet.")
            elif category[0] == "C":
                logging.debug("Input contains control character:\t{unicode_info(symbol)}. This character will not be included in the alphabet.")
            elif character == morpheme_delimiter:
                logging.debug("Not including morpheme delimeter {morpheme_delimiter} in the alphabet.")
            elif character == blacklist_char:
                logging.debug("Not including character {blacklist_char} in the alphabet.")
            elif character == padding_symbol:
                raise RuntimeError(f"Input contains reserved padding character {padding_symbol}, but this character must not occur in the corpus.")
            elif character == end_of_morpheme_symbol:
                raise RuntimeError(f"Input contains reserved end of morpheme character {end_of_morpheme_symbol}, but this character must not occur in the corpus.")
            else:
                alphabet_set.add(character)

    # Zero is reserved for OOV
    output(output_file=output_file,
           int_value=0,
           character="",
           unicode_name="",
           description="Integer value 0 is reserved to represent out-of-vocabulary characters in a tensor product representation")

    # We reserve another character to represent the end of morpheme in a tensor product representation
    output(output_file=output_file,
           int_value=1,
           character=escaped_codepoints(end_of_morpheme_symbol),
           unicode_name=unicode_info(end_of_morpheme_symbol),
           description="Integer value 1 is reserved to represent the end of a morpheme in a tensor product representation")
    
    # We reserve another character to represent the padding after the end of morpheme in a tensor product representation
    output(output_file=output_file,
           int_value=2,
           character=escaped_codepoints(padding_symbol),
           unicode_name=unicode_info(padding_symbol),
           description="Integer value 2 is reserved to represent padding beyond the end of a morpheme in a tensor product representation")

    # Remaining actual characters
    for i, symbol in enumerate(sorted(alphabet_set), start=3):
        output(output_file=output_file,
               int_value=i,
               character=symbol,
               unicode_name=unicode_info(symbol),
               description="")

        
if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Construct tensor product representations of each morpheme."
    )
    arg_parser.add_argument(
        "-e",
        "--end_of_morpheme_symbol",
        metavar="character",
        type=str,
        nargs="?",
        default="\\u0000",
        help="In the output tensor representation, "
        + "this character will be appended as the final symbol in every morpheme. "
        + "This symbol must not appear in the alphabet",
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
        "-p",
        "--padding_symbol",
        metavar="string",
        type=str,
        nargs="?",
        default="\\u0004",
        help="In the output tensor representation, "
        + "this character will be appended after the end-of-morpheme symbol."
        + "This symbol must not appear in the alphabet",
    )
    arg_parser.add_argument(
        "-i",
        "--input_file",
        metavar="filename",
        type=str,
        nargs="?",
        default="-",
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
        default="-",
        help="Output file"
    )
    arg_parser.add_argument("-v", "--verbose", metavar="int", type=int, default=0)

    args = arg_parser.parse_args()

    input_file=sys.stdin if args.input_file=="-" else open(args.input_file, mode="rt", encoding="utf8")
    output_file=sys.stdout if args.output_file=="-" else open(args.output_file, mode="wt", encoding="utf8")
    
    main(
        end_of_morpheme_symbol=str.encode(args.end_of_morpheme_symbol).decode("unicode_escape"),
        padding_symbol=str.encode(args.padding_symbol).decode("unicode_escape"),
        morpheme_delimiter=str.encode(args.morpheme_delimiter).decode("unicode_escape"),
        input_file=input_file,
        output_file=output_file,
        verbose=args.verbose,
        blacklist_char=args.blacklist_char
    )
