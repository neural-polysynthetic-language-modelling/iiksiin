#!/usr/bin/env python3.7

import grapheme  # type: ignore
import gzip
import torch     # type: ignore
import sys
from typing import Dict, List, Callable, Tuple, Set, Mapping, Iterable, Iterator
import unicodedata
from iiksiin import *

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

def main(
    max_characters: int,
    alphabet_file: str,
    input_file: str,
    output_file: str,
) -> None:

    import pickle

    success = True
    
    with gzip.open(input_file, "rb") as input_source, open(alphabet_file, "rb") as alphabet_file, open(output_file, "wt") as output:

        alphabet: Alphabet = pickle.load(alphabet_file)

        tensor_dict: Dict[str, torch.Tensor] = pickle.load(input_source)

        for morpheme in sorted(tensor_dict.keys()):
            
            tensor: torch.Tensor = tensor_dict[morpheme]
            
            reconstructed_surface_form = TensorProductRepresentation.extract_surface_form(alphabet=alphabet,
                                                                                          morpheme_tensor=tensor,
                                                                                          max_chars_per_morpheme=max_characters)

            if morpheme!=reconstructed_surface_form:
                success = False
            
            print(f"{morpheme}\t{reconstructed_surface_form}\t{morpheme==reconstructed_surface_form}", file=output)
            output.flush()

        return success

    
if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Reconstruct strings from tensor product representations of each morpheme."
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
        "-a",
        "--alphabet",
        metavar="filename",
        type=str,
        required=True,
        help="Python pickle file containing an Alphabet object"
    )
    arg_parser.add_argument(
        "-i",
        "--input_file",
        metavar="filename",
        type=str,
        nargs="?",
        default="-",
        help="Input file containing pickled dictionary mapping morpheme strings to tensors"
    )
    arg_parser.add_argument(
        "-o",
        "--output_file",
        metavar="filename",
        type=str,
        nargs="?",
        required=True,
        help="Output file"
    )

    args = arg_parser.parse_args()

    status: Bool = main(
        max_characters=args.max_characters,
        alphabet_file=args.alphabet,
        input_file=args.input_file,
        output_file=args.output_file,
    )

    if status==True:
        sys.exit(0)
    else:
        sys.exit(1)
