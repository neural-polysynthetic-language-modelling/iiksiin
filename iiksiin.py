from alphabet import Alphabet
import grapheme  # type: ignore
import gzip
import logging
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


class Dimension:
    def __init__(self, name: str, size: int):
        self.name = name
        self._length = size

    def __str__(self) -> str:
        return f"{self._length}"

    def __len__(self) -> int:
        return self._length

    def __repr__(self) -> str:
        return f"Dimension({self.name}, {self._length})"


class Shape:
    def __init__(self, *dimensions: Dimension):
        self.dimensions: Tuple[Dimension, ...] = dimensions
        self._name2dimension: Mapping[str, Dimension] = {
            dimension.name: dimensions[index] for (index, dimension) in enumerate(dimensions)
        }

    def __len__(self) -> int:
        """Gets the number of dimensions."""
        return len(self.dimensions)

    def __getitem__(self, name: str) -> int:
        """Gets the size of the named dimension."""
        return len(self._name2dimension[name])

    def __iter__(self) -> Iterator[Dimension]:
        return iter(self.dimensions)

    def __str__(self) -> str:
        return f"({','.join([str(len(dimension)) for dimension in self.dimensions])})"

    def __repr__(self) -> str:
        return f"Shape({', '.join([str(dimension) for dimension in self.dimensions])})"


class Tensor:
    def __init__(self, shape: Shape, data: torch.Tensor):
        self.shape = shape
        self.data = data

    @staticmethod
    def zeros(shape: Shape) -> "Tensor":
        return Tensor(
            shape=shape, data=torch.zeros(size=[len(dimension) for dimension in shape])
        )

    def __len__(self) -> int:
        return len(self.shape)

    def tensor_product(self, other: "Vector") -> "Tensor":
        """Perform tensor product of this tensor with the other vector.

           Assume that the j^th dimension of this tensor is the final dimension of this tensor.
           Assume that the k^th dimension is the only dimension of the other vector that vector.

           The resulting tensor will have the dimensionality of this tensor plus the dimensionality of the other vector.
           The resulting tensor's final two dimensions will be the j^th and k^th dimensions.

           Let d, ..., i refer to the first through penultimate dimensions of this tensor.

           The value result[d][...][i][j][k] = self[d][...][i][j] * other[k]
        """
        resulting_dimensions: Iterable[
            Dimension
        ] = self.shape.dimensions + other.shape.dimensions
        resulting_shape: Shape = Shape(
            *resulting_dimensions
        )  # Using * to convert list to varargs

        equation_in_einstein_notation: str = "...j,k->...jk"
        torch_tensor: torch.Tensor = torch.einsum(
            equation_in_einstein_notation, [self.data, other.data]
        )
        return Tensor(shape=resulting_shape, data=torch_tensor)

    def __iadd__(self, other: "Tensor") -> "Tensor":
        self.data += other.data
        return self


class Vector(Tensor):
    def __init__(self, dimension: Dimension, data: torch.Tensor):
        super().__init__(shape=Shape(dimension), data=data)
        self.dimension: Dimension = dimension


class OneHotVector(Vector):
    def __init__(self, index: int, dimension: Dimension):
        super().__init__(dimension=dimension, data=torch.zeros(size=[len(dimension)]))
        if 0 <= index < len(dimension):
            self._index = index
            self.data[index] = 1
        else:
            raise IndexError(
                f'The provided index {index} is invalid for "{repr(dimension)}"'
            )

    def __str__(self) -> str:
        return f"One hot at index {self._index} of {repr(self.dimension)}"

    def __repr__(self) -> str:
        return f"OneHotVector({self._index}, {self.dimension})"




class Roles:
    def __init__(
        self,
        dimension: Dimension,
        get_role_vectors: Callable[[Dimension], List[Vector]],
    ):
        self._one_hot_role_vectors: List[Vector] = get_role_vectors(dimension)
        self.dimension: Dimension = dimension

    def __getitem__(self, index) -> Vector:
        return self._one_hot_role_vectors[index]

    def __iter__(self) -> Iterable[Vector]:
        return iter(self._one_hot_role_vectors)

    def __len__(self) -> int:
        return len(self._one_hot_role_vectors)
    
    @staticmethod
    def get_one_hot_role_vectors(dimension: Dimension) -> List[Vector]:
        return [OneHotVector(index, dimension) for index in range(len(dimension))]


class TensorProductRepresentation:
    def __init__(
        self,
        alphabet: Alphabet,
        characters_dimension: Dimension
#        morphemes_dimension: Dimension,
    ):
        self.alphabet: Alphabet = alphabet
        self.character_roles: Roles = Roles(
            characters_dimension, get_role_vectors=Roles.get_one_hot_role_vectors
        )
#        self.morpheme_roles: Roles = Roles(
#            morphemes_dimension, get_role_vectors=Roles.get_one_hot_role_vectors
#        )

    @staticmethod
    def extract_surface_form(alphabet: Mapping[str, int],
                             morpheme_tensor: torch.Tensor,
                             max_chars_per_morpheme: int = 20) -> str:

        import math

        result: List[str] = list()

        for character_position in range(max_chars_per_morpheme):  # type: int
            character_position_role = torch.zeros(max_chars_per_morpheme).cpu()
            character_position_role[character_position] = 1.0
        # for character_position in character_roles:  # Type: Vector

            equation_in_einstein_notation: str = "...j,j->..."  #
            """This means that the final dimension of data and the only dimension of role are the same size.
               We will be summing over that dimension."""

#            print(f"morpheme_tensor.shape is {morpheme_tensor.shape}")
#            print(f"character_position_role.shape is {character_position_role.shape}")

            vector_for_current_character: torch.Tensor = torch.einsum(
                equation_in_einstein_notation, [morpheme_tensor.cpu(), character_position_role.cpu()]
            )
#            print(f"vector_for_current_character.shape is {vector_for_current_character.shape}")
#            print(f"vector for position {character_position}:\t{vector_for_current_character}")
            best_character = None
            best_distance = float("inf")

            for character in alphabet:
                i: int = alphabet[character]
                gold_character_vector = torch.zeros(vector_for_current_character.nelement()).cpu()
                gold_character_vector[i] = 1.0
                #print(f"gold_character_vector.shape is {gold_character_vector.shape}")
                # print(gold_character_vector.shape)
                # print(vector_for_current_character.shape)
                # character_vector: Vector = alphabet.get_vector(character)
                #distance = gold_character_vector.dot(vector_for_current_character).item()
                summation: float = 0.0
                for x in range(vector_for_current_character.nelement()):
                    summation += (gold_character_vector[x] - vector_for_current_character[x])**2
                distance = math.sqrt(summation)

                if distance < best_distance:
                    best_character = character
                    best_distance = distance
                  #  print(f"best_character is now {Alphabet.unicode_info(best_character)} with distance {best_distance}")

            if best_character == Alphabet.END_OF_MORPHEME:
#                result.append("\u2400")
                break
#            elif best_character == Alphabet.END_OF_TRANSMISSION:
#                result.append("\u2404")
            else:
                result.append(best_character)
            #print(f"best character at position {character_position} is {Alphabet.unicode_info(best_character)}", file=sys.stderr)

        return "".join(result)

    def process_morpheme(self, morpheme: Iterable[str]) -> Tensor:
        return TensorProductRepresentation.process_characters_in_morpheme(
            characters=list(morpheme) + [Alphabet.END_OF_MORPHEME],
            alphabet=self.alphabet,
            character_roles=self.character_roles
        )

    # def process_morphemes(self, morphemes: Iterable[Iterable[str]]) -> Tensor:
    #     return TensorProductRepresentation.process_morphemes_in_word(
    #         morphemes=morphemes,
    #         alphabet=self.alphabet,
    #         character_roles=self.character_roles,
    #         morpheme_roles=self.morpheme_roles,
    #     )

    @staticmethod
    def process_characters_in_morpheme(
        characters: Iterable[str], alphabet: Alphabet, character_roles: Roles
    ) -> Tensor:

        result: Tensor = Tensor.zeros(
            shape=Shape(alphabet.dimension, character_roles.dimension)
        )

        # Process characters in the actual morpheme
        for index, char in enumerate(characters):
            char_vector: Vector = alphabet.get_vector(char)  # OneHotVector(alphabet[char], alphabet.dimension)
            role_vector: Vector = character_roles[index]
            result += char_vector.tensor_product(role_vector)

        # Treat anything after the morpheme as being filled by Alphabet.END_OF_TRANSMISSION
        char_vector = alphabet.get_vector(Alphabet.END_OF_TRANSMISSION)
        for index in range(index+1, len(character_roles)):
            role_vector: Vector = character_roles[index]
            result += char_vector.tensor_product(role_vector)

        return result

    # @staticmethod
    # def process_morphemes_in_word(
    #     morphemes: Iterable[Iterable[str]],
    #     alphabet: Alphabet,
    #     morpheme_roles: Roles,
    #     character_roles: Roles,
    # ) -> Tensor:
    #
    #     result: Tensor = Tensor.zeros(
    #         shape=Shape(
    #             alphabet.dimension, character_roles.dimension, morpheme_roles.dimension
    #         )
    #     )
    #
    #     for index, morpheme in enumerate(morphemes):
    #         morpheme_role: Vector = morpheme_roles[index]
    #         morpheme_vector: Vector = TensorProductRepresentation.process_characters_in_morpheme(
    #             characters=morpheme, alphabet=alphabet, character_roles=character_roles
    #         )
    #
    #         morph_tensor: Tensor = morpheme_vector.tensor_product(morpheme_role)
    #         result += morph_tensor
    #
    #     return result


def main(
    max_characters: int,
    max_morphemes: int,
    alphabet_file: str,
    end_of_morpheme_symbol: str,
    morpheme_delimiter: str,
    input_file: str,
    output_file: str,
    alphabet_out: str,
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
    if (
        alphabet_file
    ):  # If user provided a file containing alphabet symbols, attempt to use it
        try:
            with open(alphabet_file, "r") as alphabet_source:
                for line in alphabet_source:
                    for character in grapheme.graphemes(line.strip()):
                        alphabet_set.add(character)

        except OSError as err:
            logging.error(f"ERROR - failed to read alphabet file {alphabet_file}:\t{err}")
            sys.exit(-1)

    with (sys.stdin if input_file == "-" else open(input_file)) as input_source:

        with gzip.open(output_file, "wb") as output:

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

            logging.info(f"Writing binary file containing {len(result)} morphemes to disk at {output}...")
            with open(alphabet_out, 'wb') as afile:
                pickle.dump(alphabet, afile)
            pickle.dump((result, alphabet._symbols), output)
            logging.info(f"...done writing binary file to disk at {output}", file=sys.stderr)

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
        help="File containing alphabet of characters "
        + "(Unicode escapes of the form \\u2017 are allowed.",
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
    arg_parser.add_argument(
        "--alphabet_output",
        type=str,
        nargs="?",
        required=True,
        help="output file for alphabet object"
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
        alphabet_out=args.alphabet_output,
        verbose=args.verbose,
        blacklist_char=args.blacklist_char
    )
