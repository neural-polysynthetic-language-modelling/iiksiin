import torch # type: ignore
from typing import Dict, List
from iiksiin import Alphabet, TensorProductRepresentation


class TrueTensorRetreiver:
    def __init__(self, *, alphabet: Dict[str, int], device: torch.device):
        self.alphabet: Dict[str, int] = alphabet

        self.symbols: Dict[int, str] = {i:s for s,i in self.alphabet.items()}
        self.symbols[0] = "?"
        
        self.c = 1 + len(self.alphabet)
        """The size of a character vector."""

        self.a = 1 + len(self.alphabet)
        """Size of alphabet when augmented to include OOV"""
        
        alpha_tensor = []
        for character in self.alphabet.keys(): # type: str
            
            gold_character_vector = torch.zeros(self.c) # shape: [c] where c is the size of a character vector.
            i: int = self.alphabet[character]           #
            gold_character_vector[i] = 1.0              # In our case c = 1 + len(alphabet) because we are using 1-hot vectors to represent characters.
                                                        # The +1 is because we reserve one spot for out-of-vocabulary characters.
            
            alpha_tensor.append(gold_character_vector)  # Keep a list of character vectors

        oov = torch.zeros(self.c)                       # Create a vector for OOV characters
        oov[0] = 1.0                                    #
        alpha_tensor.insert(0, oov)                     # Add it to the front of the list
        
        self.alpha_tensor: torch.Tensor = torch.stack(alpha_tensor).to(device) # shape: [a,c] where a is 1+len(alphabet) and
                                                                               #              where c is the dimensionality of each character vector
                                                                               #
                                                                               # In our case, because we happen to be using 1-hot character vectors,
                                                                               #    the value of c is also 1+len(alphabet), but in the general case a and c wouldn't necessarily be equal.

    def argmax_characters(self, batch: torch.Tensor):
        """For each morpheme in the batch, this method calculates the most likely character at each character position.

           The argument batch is a PyTorch tensor of shape [n,b,c*m].

           n is the number of morphemes in the sequence.
           b is the batch size.
           c is the size of each character vector.
           m is the maximum number of characters allowed per morpheme in the tensor product representation.

           The result returned is a tensor of shape [b,m].

           Let 0 >= w > n so that w represents a particular morpheme in the sequence and 
               0 >= x > b so that x represents a particular element in the batch and 
               0 >= y > m so that y represents a particular character position in the morpheme.

           If z = indices[w,x,y], 
              then z represents an integer representation of a character in the alphabet
              so that for element x in the batch, z is the most likely character at character position y.
        """

        n = batch.size(0)
        b = batch.size(1)
        distances = torch.einsum(            # The result of the einsum operation has shape [n,b,a,m]
            "nbcm,ac-> nbam",
            batch.view(n, b, self.c, -1),    # View the batch data to have shape [n,b,c,m] instead of shape [n, b, c*m]
            self.alpha_tensor,               #       self.alpha_tensor has shape [a,c]
        )

        indices = torch.argmax(distances, dim=2) # The result of the argmax is shape [n,b,m]

        # print(f"batch {batch.size()}\tdistances {distances.size()}\tindices {indices.size()}")

        return distances, indices
                                                                               
    def retreive(self, batch: torch.Tensor): # shape: [b,c*m] where b is the batch size and
                                             #                where c is the size of each character vector and
                                             #                where m is the maximum number of characters allowed per morpheme in our tensor product representation
        distances, indices = self.argmax_characters(batch)
                                                 

#        result = torch.zeros_like(distances).scatter_(1, indices.unsqueeze(2).expand(-1,-1, distances.size(2), -1), torch.ones_like(distances))
        alpha_expanded = self.alpha_tensor.unsqueeze(-1).unsqueeze(0).unsqueeze(0).expand(indices.size(0), indices.size(1), -1, -1, indices.size(2))
        indices_expanded = indices.unsqueeze(-2).unsqueeze(-2).expand(-1, -1, -1, self.alpha_tensor.size(1), -1)

        result = torch.gather(alpha_expanded, 2, indices_expanded).squeeze(2)
        # Let result be a tensor of shape [n,b,a,m], initially all zeros.
        # Let 0 >= w > n and 0 >= x > b and 0 >= y > m and let 0 >= z > a. Then set result[w,x,z,y] = 1, meaning that in batch x, in morpheme w, character z occurs at character position y.
        #
        # TODO: result ideally should be shape [n,b,c,m], but for the moment is actually of shape [n,b,a,m]. It just happens that c==a in our case.
        
        return result # Result is a tensor product representation


    def extract_string(self, batch: torch.Tensor, morpheme_delimiter: str):
        """Given a tensor called batch of shape [n,b,c*m], returns a list of length b containing the string representation of each TPR in the batch.
           n is the number of morphemes in a sequence.
           b is the batch size.
           c is the size of each character vector.
           m is the maximum number of characters allowed per morpheme in the tensor product representation.
        """

        _, indices = self.argmax_characters(batch)  # indices has shape [n,b,m]

        n = indices.size(0)
        b = indices.size(1)
        m = indices.size(2)
        print(f"indices {indices.size()}\tbatch {batch.size()}")
        print(indices[0,0])
        result=list()
        for batch_index in range(b):
            word=list()
            for morpheme_index in range(n):
                morpheme = list()
                for char_position in range(m):
                    alphabet_index = indices[morpheme_index, batch_index, char_position].item()
                    morpheme.append(self.symbols[alphabet_index])
                word.append("".join(morpheme))
            result.append(morpheme_delimiter.join(word))
                
        return result
    
            
        
        
