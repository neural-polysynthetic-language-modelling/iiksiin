import torch # type: ignore
from typing import Dict, List
from iiksiin import Alphabet, TensorProductRepresentation


class TrueTensorRetreiver:
    def __init__(self, *, alphabet: Dict[str, int], device: torch.device):
        self.alphabet: Alphabet = alphabet

        self.c = 1 + len(self.alphabet)
        """The size of a character vector."""

        self.a = 1 + len(self.alphabet)
        """Size of alphabet when augmented to include OOV"""
        
        alpha_tensor = []
        for character in self.alphabet.keys(): # type: str
            i: int = self.alphabet[character]
            gold_character_vector = torch.zeros(self.c) # shape: [c] where c is the size of a character vector.
                                                        #
                                                        # In our case c = 1 + len(alphabet) because we are using 1-hot vectors to represent characters.
                                                        # The +1 is because we reserve one spot for out-of-vocabulary characters.
            gold_character_vector[i] = 1.0
            alpha_tensor.append(gold_character_vector)
        oov = torch.zeros(len(self.alphabet) + 1)
        oov[0] = 1.0
        alpha_tensor.insert(0, oov)
        self.alpha_tensor: torch.Tensor = torch.stack(alpha_tensor).to(device) # shape: [a,c] where a is 1+len(alphabet) and
                                                                               #              where c is the dimensionality of each character vector
                                                                               #
                                                                               # In our case, because we happen to be using 1-hot character vectors,
                                                                               #    the value of c is also 1+len(alphabet), but in the general case a and c wouldn't necessarily be equal.

    def retreive(self, batch: torch.Tensor): # shape [b,c,m] where b is the batch size and
                                             #               where c is the size of each character vector and
                                             #               where m is the maximum number of characters allowed per morpheme in our tensor product representation
        distances = torch.einsum(
            "bcm,ac-> bam",
            batch.view(batch.size(0), len(self.alphabet) + 1, -1),
            self.alpha_tensor,
        )
        indices = torch.argmax(distances, dim=1)
        return torch.zeros_like(distances).scatter_(1, indices.unsqueeze(1).expand(-1, distances.size(1), -1), torch.ones_like(distances))
