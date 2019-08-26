import grapheme  # type: ignore
import gzip
import logging
import torch     # type: ignore
import sys
from typing import Any, Dict, List, Callable, Tuple, Set, Mapping, Iterable, Iterator
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

class Alphabet:

    def __init__(self, name: str, symbols: Set[str], end_of_morpheme_symbol: str = "\u0000", padding_symbol: str = "\u0004"):
        constructor_errors: List[str] = list()
        if end_of_morpheme_symbol in symbols:
            constructor_errors.append(f"The end-of-morpheme symbol ({Alphabet.unicode_info(end_of_morpheme_symbol)}) must not be in the provided set of symbols")
        if padding_symbol in symbols:
            constructor_errors.append(f"The padding symbol ({Alphabet.unicode_info(padding_symbol)})  must not be in the provided set of symbols.")
        if end_of_morpheme_symbol==padding_symbol:
            constructor_errors.append(f"The end-of-morpheme symbol ({Alphabet.unicode_info(end_of_morpheme_symbol)}) and the padding symbol ({Alphabet.unicode_info(padding_symbol)}) must not be the same.")
        if constructor_errors:
            raise ValueError(" ".join(constructor_errors))

        self._symbols: Mapping[str, int] = dict()

        # The value self.oov (in the context of self._symbols) is reserved for all out-of-alphabet symbols
        self.oov = 0
        
        self.end_of_morpheme_symbol = end_of_morpheme_symbol
        self._symbols[self.end_of_morpheme_symbol] = 1
        
        self.padding_symbol = padding_symbol
        self._symbols[self.padding_symbol] = 2

        for (index, symbol) in enumerate(sorted(symbols), start=3):
            self._symbols[symbol] = index
        
        self.dimension: Dimension = Dimension(name, 1 + len(self._symbols))
        self.name = name

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

    def __contains__(self, item: Any) -> bool:
        if item in self._symbols:
            return True
        else:
            return False
        
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
    def create_from_source(name: str, source: Iterable[str], morpheme_delimiter: str, end_of_morpheme_symbol: str, padding_symbol: str, blacklist_char: str) -> "Alphabet":
        alphabet_set: Set[str] = set()
        for line in source: # type: str
            for character in grapheme.graphemes(line.strip()): # type: str
                category: str = unicodedata.category(character)
                if category[0] != "Z" and category[0] != "C" and character != morpheme_delimiter and character != end_of_morpheme_symbol and character != blacklist_char:
                    alphabet_set.add(character)

        for symbol in alphabet_set: # type: str
            for character in symbol: # type: str
                category: str = unicodedata.category(character)
                if category[0] == "Z":
                    logging.warning(f"WARNING - alphabet contains whitespace character:\t{Alphabet.unicode_info(symbol)}")

                elif (category[0] == "C" and character != morpheme_delimiter and character != end_of_morpheme_symbol):
                    logging.warning(f"WARNING - alphabet contains control character:\t{Alphabet.unicode_info(symbol)}")

        return Alphabet(name=name,
                        symbols=alphabet_set,
                        end_of_morpheme_symbol=end_of_morpheme_symbol,
                        padding_symbol=padding_symbol)


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
    def extract_surface_form(alphabet: Alphabet,
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

            if best_character == alphabet.end_of_morpheme_symbol:
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
            characters=list(morpheme) + [self.alphabet.end_of_morpheme_symbol],
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

        # Treat anything after the morpheme as being filled by alphabet.padding_symbol
        char_vector = alphabet.get_vector(alphabet.padding_symbol)
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
    verbose: int,
    blacklist_char: str,
    validate_tensors: bool
) -> None:

    import pickle

    if grapheme.length(end_of_morpheme_symbol) != 1:
        raise RuntimeError(
            "The end of morpheme symbol must consist of a single grapheme cluster "
            + "(see Unicode Standard Annex #29)."
        )

    
    with open(alphabet_file, "rb") as f:
        alphabet: Alphabet = pickle.load(f)

    with (sys.stdin if input_file == "-" else open(input_file)) as input_source:

        with gzip.open(output_file, "wb") as output:

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
                            if character not in alphabet and character != morpheme_delimiter and character != end_of_morpheme_symbol:
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
                                    if validate_tensors:
                                        reconstructed_surface_form = TensorProductRepresentation.extract_surface_form(alphabet=tpr.alphabet, morpheme_tensor=tensor.data, max_chars_per_morpheme=len(tpr.character_roles))
                                        assert(reconstructed_surface_form == morpheme)
                                    result[morpheme] = tensor.data
                                except (IndexError, AssertionError) as e:
                                    if isinstance(e, IndexError):
                                        logging.warning(f"Line {number} - unable to process morpheme {morpheme} (length {len(morpheme)}) of {word}" + "\t" + str(e))
                                    elif isinstance(e, AssertionError):
                                        logging.warning(f"Line {number} - unable to reconstruct morpheme {morpheme} (length {len(morpheme)}) of {word} from tensor representation")
                                    
                                    skipped_morphemes.add(morpheme)
                                    raise e

            logging.info(f"Writing binary file containing {len(result)} morphemes to disk at {output}...")
            pickle.dump(result, output)
            logging.info(f"...done writing binary file to disk at {output}", file=sys.stderr)

            logging.info(f"Failed to process {len(skipped_morphemes)} morphemes:\n"+"\n".join(skipped_morphemes))

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Construct tensor product representations of each morpheme."
    )
    arg_parser.add_argument(
        "-x",
        "--validate_tensors",
        metavar="BOOL",
        type=bool,
        nargs="?",
        default=True,
        help="Should the program attempt to reconstruct each morpheme from its tensor representation"
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
        required=True,
        help="Python pickle file containing an Alphabet object"
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
        required=True,
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
        alphabet_file=args.alphabet,
        end_of_morpheme_symbol=str.encode(args.end_of_morpheme_symbol).decode(
            "unicode_escape"
        ),
        morpheme_delimiter=str.encode(args.morpheme_delimiter).decode("unicode_escape"),
        input_file=args.input_file,
        output_file=args.output_file,
        verbose=args.verbose,
        blacklist_char=args.blacklist_char,
        validate_tensors=args.validate_tensors
    )
