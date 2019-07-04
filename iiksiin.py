import grapheme
import torch
import sys
from typing import Dict, List, Callable, Set, Mapping, Iterable

"""Implements Tensor Product Representation for potentially multi-morphemic words.

This file was developed as part of the Neural Polysynthetic Language Modelling project
at the 2019 Frederick Jelinek Memorial Summer Workshop at École de Technologie Supérieure in Montréal, Québec, Canada.
https://www.clsp.jhu.edu/workshops/19-workshop/
"""

__author__ = "Lane Schwartz"
__copyright__ = "Copyright 2019, Lane Schwartz"
__license__ = "MPL 2.0"
__credits__ = ["Lane Schwartz", "Coleman Haley", "Francis Tyers", "JSALT 2019 NPLM team members"]
__maintainer = "Lane Schwartz"
__email__ = "dowobeha@gmail.com"
__version__ = "0.0.1"
__status__ = "Prototype"


if sys.version_info < (3, 7):
    raise RuntimeError(f"{__file__} requires Python 3.7 or later")


class Dimension:

    def __init__(self, name: str, size: int):
        self._name = name
        self._length = size

    def __str__(self) -> str:
        return f"{self._length}"

    def __len__(self):
        return self._length

    def __repr__(self):
        return f"Dimension({self._name}, {self._length})"


class Shape:

    def __init__(self, *dimensions: Dimension):
        self.dimensions: List[Dimension] = dimensions
        self._name2dimension: Mapping[str, Dimension] = \
            {name: dimensions[index] for (index, name) in enumerate(dimensions)}

    def __len__(self) -> int:
        """Gets the number of dimensions."""
        return len(self.dimensions)

    def __getitem__(self, name: str) -> int:
        """Gets the size of the named dimension."""
        return len(self._name2dimension[name])

    def __iter__(self) -> Iterable[Dimension]:
        return iter(self.dimensions)

    def __str__(self):
        return f"({','.join([str(len(dimension)) for dimension in self.dimensions])})"

    def __repr__(self):
        return f"Shape({', '.join([str(dimension) for dimension in self.dimensions])})"


class Tensor:

    def __init__(self, shape: Shape, data: torch.Tensor):
        self.shape = shape
        self.data = data

    @staticmethod
    def zeros(shape: Shape):
        return Tensor(shape=shape, data=torch.zeros(size=[len(dimension) for dimension in shape]))

    def __len__(self):
        return len(self.shape)

    def tensor_product(self, other: 'Vector') -> 'Tensor':
        """Perform tensor product of this tensor with the other vector.

           Assume that the j^th dimension of this tensor is the final dimension of this tensor.
           Assume that the k^th dimension is the only dimension of the other vector that vector.

           The resulting tensor will have the dimensionality of this tensor plus the dimensionality of the other vector.
           The resulting tensor's final two dimensions will be the j^th and k^th dimensions.

           Let d, ..., i refer to the first through penultimate dimensions of this tensor.

           The value result[d][...][i][j][k] = self[d][...][i][j] * other[k]
        """
        resulting_dimensions: List[Dimension] = self.shape.dimensions + other.shape.dimensions
        resulting_shape: Shape = Shape(*resulting_dimensions)  # Using * to convert list to varargs

        equation_in_einstein_notation: str = "...j, k -> ...jk"
        torch_tensor: torch.Tensor = torch.einsum(equation_in_einstein_notation, [self.data, other.data])
        return Tensor(shape=resulting_shape, data=torch_tensor)

    def __iadd__(self, other: 'Tensor') -> 'Tensor':
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
            raise IndexError(f"The provided index {index} is invalid for \"{repr(dimension)}\"")

    def __str__(self):
        return f"One hot at index {self._index} of {repr(self.dimension)}"

    def __repr__(self):
        return f"OneHotVector({self._index}, {self.dimension})"


class Alphabet:

    END_OF_MORPHEME = "\u0000"

    def __init__(self, name: str, symbols: Set[str]):
        self._symbols: Mapping[str, int] = {symbol: index for (index, symbol) in enumerate(symbols, start=1)}
        self.dimension: Dimension = Dimension(name, 1+len(symbols))
        self.name = name
        self.oov = 0

    def __getitem__(self, symbol: str) -> int:
        if symbol in self._symbols:
            return self._symbols[symbol]
        else:
            return self.oov

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Alphabet({str(self.dimension)})"


class Roles:

    def __init__(self, dimension: Dimension, get_role_vectors: Callable[[Dimension], List[Vector]]):
        self._one_hot_role_vectors: List[Vector] = get_role_vectors(dimension)
        self.dimension: Dimension = dimension

    def __getitem__(self, index) -> Vector:
        return self._one_hot_role_vectors[index]

    def __iter__(self) -> Iterable[Vector]:
        return iter(self._one_hot_role_vectors)

    @staticmethod
    def get_one_hot_role_vectors(dimension: Dimension) -> List[Vector]:
        return [OneHotVector(index, dimension) for index in range(len(dimension))]


class TensorProductRepresentation:

    def __init__(self, alphabet: Alphabet, characters_dimension: Dimension, morphemes_dimension: Dimension):
        self.alphabet: Alphabet = alphabet
        self.character_roles: Roles = Roles(characters_dimension, get_role_vectors=Roles.get_one_hot_role_vectors)
        self.morpheme_roles: Roles = Roles(morphemes_dimension, get_role_vectors=Roles.get_one_hot_role_vectors)

    def process_morphemes(self, morphemes: Iterable[Iterable[str]]):
        return TensorProductRepresentation.process_morphemes_in_word(
            morphemes=morphemes,
            alphabet=self.alphabet,
            character_roles=self.character_roles,
            morpheme_roles=self.morpheme_roles)

    @staticmethod
    def process_characters_in_morpheme(characters: Iterable[str],
                                       alphabet: Alphabet,
                                       character_roles: Roles) -> Tensor:

        result: Tensor = Tensor.zeros(shape=Shape(alphabet.dimension, character_roles.dimension))

        for index, char in enumerate(characters):
            char_vector: Vector = OneHotVector(alphabet[char], alphabet.dimension)
            role_vector: Vector = character_roles[index]

            result += char_vector.tensor_product(role_vector)

        return result

    @staticmethod
    def process_morphemes_in_word(morphemes: Iterable[Iterable[str]],
                                  alphabet: Alphabet,
                                  morpheme_roles: Roles,
                                  character_roles: Roles) -> Tensor:

        result: Tensor = Tensor.zeros(shape=Shape(alphabet.dimension,
                                                  character_roles.dimension,
                                                  morpheme_roles.dimension))

        for index, morpheme in enumerate(morphemes):
            morpheme_role: Vector = morpheme_roles[index]
            morpheme_vector: Vector = TensorProductRepresentation.process_characters_in_morpheme(
                                                                    characters=morpheme,
                                                                    alphabet=alphabet,
                                                                    character_roles=character_roles)

            morph_tensor: Tensor = morpheme_vector.tensor_product(morpheme_role)
            result += morph_tensor

        return result


def main(max_characters: int,
         max_morphemes: int,
         alphabet_symbols: str,
         end_of_morpheme_symbol: str,
         morpheme_delimiter: str,
         input_file: str,
         output_file: str) -> None:

    import pickle

    if grapheme.length(end_of_morpheme_symbol) != 1:
        raise RuntimeError("The end of morpheme symbol must consist of a single grapheme cluster " +
                           "(see Unicode Standard Annex #29).")

    with (sys.stdin if input_file == "-" else open(input_file)) as input_source:

        with open(output_file, "wb") as output:

            alphabet: Alphabet = Alphabet("alphabet", set(grapheme.graphemes(end_of_morpheme_symbol +
                                                                             alphabet_symbols)))
            characters_dimension: Dimension = Dimension("characters", max_characters)
            morphemes_dimension: Dimension = Dimension("morphemes", max_morphemes)

            tpr: TensorProductRepresentation = TensorProductRepresentation(alphabet=alphabet,
                                                                           characters_dimension=characters_dimension,
                                                                           morphemes_dimension=morphemes_dimension)

            result: Dict[str, torch.Tensor] = {}

            for line in input_source:
                for word in line.split():
                    if word not in result:
                        morphemes = word.split(morpheme_delimiter)
                        tensor: Tensor = tpr.process_morphemes(morphemes)
                        result[word] = tensor.data

            pickle.dump(result, output)


if __name__ == "__main__":

    import argparse
    import string

    parser = argparse.ArgumentParser(description='Construct tensor product representations of each morpheme.')
    parser.add_argument('-c', '--max_characters',
                        metavar='N', type=int, nargs='?', default=100,
                        help='maximum number of characters allowed per morpheme')
    parser.add_argument('-m', '--max_morphemes',
                        metavar='N', type=int, nargs='?', default=20,
                        help='maximum number of morphemes allowed per word')
    parser.add_argument('-a', '--alphabet',
                        metavar='ABC', type=str, nargs='?',
                        default=string.ascii_letters +
                                string.digits +
                                string.punctuation +
                                "\\u2018\\u2019\\u201C\\u201D" +  # single and double quotation marks
                                " ",
                        help="alphabet of characters (Unicode escapes of the form \\u2017 are allowed")
    parser.add_argument('-e', '--end_of_morpheme_symbol',
                        metavar='character', type=str, nargs='?',
                        default="\\u0000",
                        help='Character that should appear as the final symbol in every morpheme' +
                             'This symbol must not appear in the alphabet')
    parser.add_argument('-d', '--morpheme_delimiter',
                        metavar='string', type=str, nargs='?',
                        default="\\u001F",
                        help='Character that should appear between adjacent morphemes. ' +
                             'This symbol must not appear in the alphabet')
    parser.add_argument('-i', '--input_file',
                        metavar='filename', type=str, nargs='?',
                        default="-",
                        help='Input file containing whitespace delimited words (- for standard input)')
    parser.add_argument('-o', '--output_file',
                        metavar='filename', type=str, nargs='?',
                        required=True,
                        help='Output file where morpheme tensors are recorded')

    args = parser.parse_args()

    main(max_characters=args.max_characters,
         max_morphemes=args.max_morphemes,
         alphabet_symbols=args.alphabet.decode('unicode_escape'),
         end_of_morpheme_symbol=args.end_of_morpheme_symbol.decode('unicode_escape'),
         morpheme_delimiter=args.morpheme_delimiter.decode('unicode_escape'),
         input_file=args.input_file,
         output_file=args.output_file)
