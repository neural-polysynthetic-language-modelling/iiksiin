#!/usr/bin/env python3.7

import grapheme  # type: ignore
import gzip
import logging
import math
import torch     # type: ignore
import sys
from typing import Dict, List, Callable, Tuple, Set, Mapping, Iterable, Iterator
import unicodedata

"""Implements Tensor Product Representation for potentially multi-morphemic words.

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

def extract_surface_form(alphabet: Mapping[str, torch.Tensor],
                         morpheme_tensor: torch.Tensor,
                         roles: List[torch.Tensor],
                         end_of_morpheme_symbol: str,
                         max_chars_per_morpheme: int = 20) -> str:

    result: List[str] = list()

    for character_position in range(max_chars_per_morpheme):  # type: int

        equation_in_einstein_notation: str = "...j,j->..."  #
        """This means that the final dimension of data and the only dimension of role are the same size.
           We will be summing over that dimension."""

        vector_for_current_character: torch.Tensor = torch.einsum(
            equation_in_einstein_notation, [morpheme_tensor, role[character_position]]
        )

        best_character = None
        best_distance = float("inf")

        for character in alphabet:
            i: int = alphabet[character]
            gold_character_vector = torch.zeros(vector_for_current_character.nelement())
            gold_character_vector[i] = 1.0

            summation: float = 0.0
            for x in range(vector_for_current_character.nelement()):
                summation += (gold_character_vector[x] - vector_for_current_character[x])**2
            distance = math.sqrt(summation)

            if distance < best_distance:
                best_character = character
                best_distance = distance

        if best_character == end_of_morpheme_symbol:
            break
        else:
            result.append(best_character)

    return "".join(result)


def main(
    max_characters: int,
    max_morphemes: int,
    alphabet_file: str,
    end_of_morpheme_symbol: str,
    morpheme_delimiter: str,
    input_file: str,
    output_file: str,
    verbose: int,
    blacklist_char: str,
) -> None:

    import pickle

    if grapheme.length(end_of_morpheme_symbol) != 1:
        raise RuntimeError(
            "The end of morpheme symbol must consist of a single grapheme cluster "
            + "(see Unicode Standard Annex #29)."
        )

    alphabet_set: Set[str] = set()
    logging.info(f"Reading alphabet from input file {input_file}...")

    with open(input_file) as input_source:
        for line in input_source:
            for character in grapheme.graphemes(line.strip()):
                category = unicodedata.category(character)
                if category[0] != "Z" and  category[0] != "C" and character != morpheme_delimiter and character != end_of_morpheme_symbol:
                    alphabet_set.add(character)

    alphabet_set.add(end_of_morpheme_symbol)

    for symbol in sorted(alphabet_set):
        for character in symbol:
            category = unicodedata.category(character)
            if category[0] == "Z": # and character != " ":
                logging.warning(f"WARNING - alphabet contains whitespace character:\t{Alphabet.unicode_info(symbol)}")

            elif (
                category[0] == "C"
                and character != morpheme_delimiter
                and character != end_of_morpheme_symbol
            ):
                logging.warning(f"WARNING - alphabet contains control character:\t{Alphabet.unicode_info(symbol)}")

    logging.info(f"Symbols in alphabet: {len(alphabet_set)}")
    if verbose > 0:
        print("-----------------------", file=sys.stderr)
        for symbol in sorted(alphabet_set):
            print(Alphabet.unicode_info(symbol), file=sys.stderr)

    with (sys.stdin if input_file == "-" else open(input_file)) as input_source:

        with gzip.open(output_file, "wb") as output, open(alphabet_file, "wb") as alphabet_output:

            alphabet: Alphabet = Alphabet("alphabet", alphabet_set)
            characters_dimension: Dimension = Dimension("characters", max_characters)
            morphemes_dimension: Dimension = Dimension("morphemes", max_morphemes)

            tpr: TensorProductRepresentation = TensorProductRepresentation(
                alphabet=alphabet,
                characters_dimension=characters_dimension
            )

            result: Dict[str, torch.Tensor] = {}
            skipped_morphemes: Set[str] = set()
            for number, line in enumerate(input_source):
                logging.debug(f"Processing line {number}\t{line.strip()}")
                for word in line.strip().split():
                    if blacklist_char in word:
                        logging.info(f"Skipping unanalyzed word {word}")
                    elif word not in result:
                        for character in grapheme.graphemes(word):
                            if character not in alphabet_set and character != morpheme_delimiter and character != end_of_morpheme_symbol:
                                logging.warning(f"WARNING - not in alphabet:\t{Alphabet.unicode_info(character)}")

                        morphemes = word.split(morpheme_delimiter)
                        for morpheme in morphemes:
                            if len(morpheme) == 0:
                                logging.debug(f"Line {number} - skipping morpheme of length 0 in word {word}")
                            elif len(morpheme) == max_characters:
                                logging.warning(f"Line {number} - skipping morpheme {morpheme} of {word} because its length {len(morpheme)} equals max length {max_characters}, and there is no space to insert the required end of morpheme symbol")
                            elif len(morpheme) > max_characters:
                                logging.warning(f"Line {number} - skipping morpheme {morpheme} of {word} because its length {len(morpheme)} exceeds max length {max_characters}")
                            else:
                                try:
                                    tensor: Tensor = tpr.process_morpheme(morpheme)
                                    result[morpheme] = tensor.data
                                except IndexError:
                                    logging.warning(f"Line {number} - unable to process morpheme {morpheme} (length {len(morpheme)}) of {word}")
                                    skipped_morphemes.add(morpheme)

            logging.info(f"Writing binary file for alphabet of size {len(alphabet)} to disk at {output}...")
            pickle.dump(alphabet, alphabet_output)
            logging.info(f"...done writing alphabet")                                    
                                    
            logging.info(f"Writing binary file containing {len(result)} morphemes to disk at {output}...")
            pickle.dump((result, alphabet._symbols), output)
            logging.info(f"...done writing binary file to disk at {output}")

            logging.info(f"Failed to process {len(skipped_morphemes)} morphemes:\n"+"\n".join(skipped_morphemes))

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Construct tensor product representations of each morpheme."
    )
    arg_parser.add_argument(
        "-c",
        "--max_characters",
        metavar="N",
        type=int,
        nargs="?",
        default=20,
        help="Maximum number of characters allowed per morpheme.",
    )
    arg_parser.add_argument(
        "-m",
        "--max_morphemes",
        metavar="N",
        type=int,
        nargs="?",
        default=10,
        help="Maximum number of morphemes allowed per word.",
    )
    arg_parser.add_argument(
        "-a",
        "--alphabet",
        metavar="filename",
        type=str,
        nargs="?",
        required=True,
        help="Output file containing alphabet of characters "
#        + "(Unicode escapes of the form \\u2017 are allowed.",
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
        "-d",
        "--morpheme_delimiter",
        metavar="string",
        type=str,
        nargs="?",
        default="\\u001F",
        help="In the user-provided input file, "
        + "this character must appear between adjacent morphemes. "
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
        required=True,
        help="Output file where morpheme tensors are recorded",
    )
    arg_parser.add_argument("-v", "--verbose", metavar="int", type=int, default=0)

    args = arg_parser.parse_args()

    main(
        max_characters=args.max_characters,
        max_morphemes=args.max_morphemes,
        alphabet_file=(args.alphabet if args.alphabet else ""),
        end_of_morpheme_symbol=str.encode(args.end_of_morpheme_symbol).decode(
            "unicode_escape"
        ),
        morpheme_delimiter=str.encode(args.morpheme_delimiter).decode("unicode_escape"),
        input_file=args.input_file,
        output_file=args.output_file,
        verbose=args.verbose,
        blacklist_char=args.blacklist_char
    )
