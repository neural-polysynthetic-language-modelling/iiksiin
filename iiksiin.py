import torch
import sys
from typing import List, Callable, Set, Mapping, Iterable

if sys.version_info < (3, 7):
    raise RuntimeError(f"{__file__} requires Python 3.7 or later")


class Dimension:

    def __init__(self, name: str, size: int):
        self._name = name
        self._length = size

    def __str__(self) -> str:
        return f"{self._length} ({self._name})"

    def __len__(self):
        return self._length

    def __repr__(self):
        return f"Dimension({self._name}, {self._length})"


class Shape:

    def __init__(self, *dimensions: Dimension):
        self.dimensions: [Dimension] = dimensions
        self._name2dimension: Mapping[str,Dimension] = {name: dimensions[index] for (index, name) in enumerate(dimensions)}

    def __len__(self) -> int:
        """Gets the number of dimensions."""
        return len(self.dimensions)

    def __getitem__(self, name: str) -> int:
        """Gets the size of the named dimension."""
        return len(self._name2dimension[name])

    def __iter__(self) -> Iterable[Dimension]:
        return iter(self.dimensions)

    def __repr__(self):
        return f"Shape({', '.join([str(dimension) for dimension in self.dimensions])})"


class Tensor:

    def __init__(self, shape: Shape):
        self.shape = shape
        self._data = torch.zeros(size=[len(dimension) for dimension in self.shape])

    def __len__(self):
        return len(self.shape)


class Vector(Tensor):

    def __init__(self, dimension: Dimension):
        super().__init__(shape=Shape(dimension))
        self.dimension: Dimension = dimension


class OneHotVector(Vector):
    def __init__(self, index: int, dimension: Dimension):
        super().__init__(dimension)
        if 0 <= index < len(dimension):
            self._index = index
            self._data[index] = 1
        else:
            raise IndexError(f"The provided index {index} is invalid for \"{repr(dimension)}\"")

    def __str__(self):
        return f"One hot at index {self._index} of {repr(self.dimension)}"

    def __repr__(self):
        return f"OneHotVector({self._index}, {self.dimension})"


class Alphabet:

    def __init__(self, name: str, symbols: Set[str]):
        self._symbols: Mapping[str, int] = {symbol: index for (index, symbol) in enumerate(symbols)}
        self.dimension: Dimension = Dimension(name, len(symbols))
        self.name = name

    def __getitem__(self, symbol: str) -> int:
        return self._symbols[symbol]

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Alphabet({str(self.dimension)})"


class Roles:

    def __init__(self, dimension: Dimension, get_role_vectors: Callable[[Dimension], List[Vector]]):
        self._one_hot_role_vectors: List[Vector] = get_role_vectors(dimension)

    def __getitem__(self, index) -> Vector:
        return self._one_hot_role_vectors[index]

    def __iter__(self) -> Iterable[Vector]:
        return iter(self._one_hot_role_vectors)


def get_one_hot_role_vectors(dimension: Dimension) -> List[Vector]:
    return [OneHotVector(index, dimension) for index in range(len(dimension))]


if __name__ == "__main__":

    alphabet_dimension: Dimension = Dimension("alphabet", 26)
    characters_dimension: Dimension = Dimension("characters", 100)
    morphemes_dimension: Dimension = Dimension("morphemes", 20)

    shape: Shape = Shape(alphabet_dimension, characters_dimension, morphemes_dimension)
    print(shape)
    r1 = OneHotVector(17, alphabet_dimension)
    print(r1)
    characterRoles: List[Vector] = Roles(characters_dimension, get_one_hot_role_vectors)

    print(f"Hi there, {len(alphabet_dimension)}")

    print(repr(characterRoles[77]))
