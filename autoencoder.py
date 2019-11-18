#!/usr/bin/env python3.7

from iiksiin import Alphabet, TensorProductRepresentation
import logging
import torch  # type: ignore
import torch.nn  # type: ignore
import torch.optim  # type: ignore
from torch import sigmoid  # type: ignore
from torch.nn.functional import relu, cross_entropy  # type: ignore

# import torch.nn.modules  # type: ignore
from typing import Mapping, Dict, List, Tuple, Iterable, Iterator
import sys

"""Implements an autoencoder to embed Tensor Product Representation tensors into smaller vectors.

This file was developed as part of the Neural Polysynthetic Language Modelling project
at the 2019 Frederick Jelinek Memorial Summer Workshop at École de Technologie Supérieure in Montréal, Québec, Canada.
https://www.clsp.jhu.edu/workshops/19-workshop/
"""

__author__ = "Lane Schwartz"
__copyright__ = "Copyright 2019, Lane Schwartz"
__license__ = "MPL 2.0"
__credits__ = ["Lane Schwartz", "JSALT 2019 NPLM team members"]
__maintainer = "Lane Schwartz"
__email__ = "dowobeha@gmail.com"
__version__ = "0.0.1"
__status__ = "Prototype"

if sys.version_info < (3, 7):
    raise RuntimeError(f"{__file__} requires Python 3.7 or later")


class BatchInfo:
    """Store information necessary to identify a morpheme given a batch number and an index within that batch."""

    def __init__(self, morphemes: Iterable[str], items_per_batch: int):
        sizes_dict: Dict[int, List[str]] = dict()
        for morpheme in morphemes:  # type: str
            length = len(morpheme)  # type: int
            if length not in sizes_dict:
                sizes_dict[length] = list()
            sizes_dict[length].append(morpheme)

        self._batches: List[List[str]] = list()
        current_batch: List[str] = list()
        for length in sorted(sizes_dict.keys()):
            for morpheme in sizes_dict[length]:
                if len(current_batch) == items_per_batch:
                    self._batches.append(current_batch)
                    current_batch = list()
                current_batch.append(morpheme)
        if len(current_batch) > 0:
            self._batches.append(current_batch)

    def __getitem__(self, index) -> List[str]:
        return self._batches[index]

    def __iter__(self) -> Iterator[List[str]]:
        return iter(self._batches)


class Tensors:
    def __init__(self, tensor_dict: Mapping[str, torch.Tensor], alphabet: Alphabet):
        self.tensor_dict: Mapping[str, torch.Tensor] = tensor_dict
        self.alphabet: Alphabet = alphabet
        self.morph_size: int = next(
            iter(self.tensor_dict.values())
        ).numel()  # Get the number of elements in a Tensor
        self.input_dimension_size: int = next(iter(self.tensor_dict.values())).view(
            -1
        ).shape[0]

    def shape(self):
        #        print(type(self.tensor_dict))
        #        print(type(self.tensor_dict.values()))
        #        print(type(iter(self.tensor_dict.values())))
        #        print(type(next(iter(self.tensor_dict.values()))))
        return next(iter(self.tensor_dict.values())).shape

    @staticmethod
    def load_from_pickle_file(tensor_filename: str, alphabet_filename: str) -> "Tensors":
        import pickle
        import gzip

        with gzip.open(tensor_filename, 'rb') as tensor_file, open(alphabet_filename, 'rb') as alphabet_file:
#            result: Tuple[Dict[str, torch.Tensor], Alphabet] = pickle.load(
#                f, encoding="utf8"
#            )
#            tensor_dict: Dict[str, torch.Tensor] = result[0]
#            alphabet: Alphabet = result[1]
            tensor_dict: Dict[str, torch.Tensor] = pickle.load(tensor_file, encoding='utf8')
            alphabet: Alphabet = pickle.load(alphabet_file, encoding='utf8')
            # (tensor_dict, alphabet:Alphabet) = pickle.load(f, encoding='utf8')
            return Tensors(tensor_dict, alphabet)

    def get_batch_info(self, items_per_batch) -> BatchInfo:
        return BatchInfo(self.tensor_dict.keys(), items_per_batch)

    def get_batches(
        self, items_per_batch, device_number
    ) -> Iterable[Tuple[int, torch.Tensor]]:

        batch_info: BatchInfo = self.get_batch_info(items_per_batch)
        batch_number: int = -1
        for batch_of_morphemes in batch_info:  # type: List[str]
            batch_number = batch_number + 1
            tensor: torch.Tensor = torch.zeros(
                items_per_batch, self.input_dimension_size
            )
            for numbered_morpheme in enumerate(
                batch_of_morphemes
            ):  # type: Tuple[int, str]
                n: int = numbered_morpheme[0]
                morpheme: str = numbered_morpheme[1]
                tensor[n] = self.tensor_dict[morpheme].view(-1)

            if 0 <= device_number < torch.cuda.device_count():
                yield (batch_number, tensor.cuda(device_number))
            else:
                yield (batch_number, tensor.cpu())


class Autoencoder(torch.nn.Module):
    def __init__(
        self, input_dimension_size: int, hidden_layer_size: int, num_hidden_layers: int
    ):
        super().__init__()
        self.input_dimension_size: int = input_dimension_size
        self.hidden_layer_size: int = hidden_layer_size
        self.hidden_layers: torch.nn.ModuleList = torch.nn.ModuleList()
        for n in range(num_hidden_layers):  # type: int
            if n == 0:
                self.hidden_layers.append(
                    torch.nn.Linear(
                        self.input_dimension_size, self.hidden_layer_size, bias=True
                    )
                )
            else:
                self.hidden_layers.append(
                    torch.nn.Linear(
                        self.hidden_layer_size, self.hidden_layer_size, bias=True
                    )
                )
        self.output_layer: torch.nn.Module = torch.nn.Linear(
            self.hidden_layer_size, self.input_dimension_size, bias=True
        )

    def forward(self, tensor_at_input_layer: torch.Tensor) -> torch.Tensor:
        tensor_at_final_hidden_layer: torch.Tensor = self._apply_hidden_layers(
            tensor_at_input_layer
        )
        tensor_at_output_layer: torch.Tensor = self._apply_output_layer(
            tensor_at_final_hidden_layer
        )
        return tensor_at_output_layer

    def _apply_hidden_layers(self, tensor_at_input_layer: torch.Tensor) -> torch.Tensor:
        tensor_at_previous_layer: torch.nn.Module = tensor_at_input_layer

        for hidden in self.hidden_layers:  # type: torch.nn.Module
            tensor_at_current_layer: torch.Tensor = relu(
                hidden(tensor_at_previous_layer)
            )
            tensor_at_previous_layer = tensor_at_current_layer

        return tensor_at_current_layer

    def _apply_output_layer(self, tensor_at_hidden_layer: torch.Tensor) -> torch.Tensor:
        # return
        # sigmoid(
        return self.output_layer(tensor_at_hidden_layer)  # .cuda(device=cuda_device))
        # )

    #    def run_v2t(self, data, max_items_per_batch: int, cuda_device: int):
    #
    #        self.eval()
    #
    #        if cuda_device < 0:
    #            self.cpu()
    #        else:
    #            self.cuda(device=cuda_device)
    #
    #        results = dict()
    #        batch_info: BatchInfo = data.get_batch_info(max_items_per_batch)

    def run_t2v(
        self, data, max_items_per_batch: int, cuda_device: int
    ) -> Dict[str, List[float]]:

        self.eval()

        if cuda_device < 0:
            self.cpu()
        else:
            self.cuda(device=cuda_device)

        results: Dict[str, torch.Tensor] = dict()
        batch_info: BatchInfo = data.get_batch_info(max_items_per_batch)
        for numbered_batch in data.get_batches(
            items_per_batch=max_items_per_batch, device_number=cuda_device
        ):  # type: Tuple[int, torch.Tensor]

            batch_number: int = numbered_batch[0]
            data_on_device: torch.Tensor = numbered_batch[1]
            logging.info(f"Batch number {batch_number}...")
            batch_of_results: torch.Tensor = self._apply_hidden_layers(data_on_device)

            morphemes: List[str] = batch_info[batch_number]
            number_of_results: int = batch_of_results.shape[0]
            for n in range(min(number_of_results, len(morphemes))):  # type: int
                morpheme: str = morphemes[n]
                tensor: torch.Tensor = batch_of_results[n]
                # print(tensor)
                # print(tensor.shape)
                # print(tensor.tolist())
                # sys.exit(0)
                results[morpheme] = tensor.tolist()
                # print(results[morpheme])
                # print(results[morpheme].shape)
                # sys.exit(0)

        return results

    def run_training(
        self,
        data: Tensors,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        max_items_per_batch: int,
        save_frequency: int,
        cuda_device: int,
    ) -> None:

        self.train()

        if cuda_device < 0:
            self.cpu()
        else:
            self.cuda(device=cuda_device)

        for epoch in range(1, num_epochs + 1):  # type: int
            optimizer.zero_grad()

            total_loss: float = 0.0

            for numbered_batch in data.get_batches(
                items_per_batch=max_items_per_batch, device_number=cuda_device
            ):  # type: Tuple[int, torch.Tensor]

                #  batch_number: int = numbered_batch[0]
                data_on_device: torch.Tensor = numbered_batch[1]

                prediction: torch.Tensor = self(data_on_device)

                # Compute Loss
                loss: torch.Tensor = criterion(prediction.squeeze(), data_on_device)
                total_loss += loss.item()

                loss.backward()

            logging.info(
                f"Epoch {str(epoch).zfill(len(str(num_epochs)))}\ttrain loss: {loss.item()}"
            )

            if epoch % save_frequency == 0 or epoch == num_epochs:
                logging.info(
                    f"Saving model to model_at_epoch_{str(epoch).zfill(len(str(num_epochs)))}.pt"
                )
                self.cpu()
                torch.save(
                    self, f"model_at_epoch_{str(epoch).zfill(len(str(num_epochs)))}.pt"
                )
                if cuda_device < 0:
                    self.cpu()
                else:
                    self.cuda(device=cuda_device)

            # Backward pass
            optimizer.step()


def program_arguments():
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Autoencode tensor product representations of each morpheme."
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
        "--epochs",
        type=int,
        default=200,
        help="Number of epochs to run during training.",
    )
    arg_parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    arg_parser.add_argument(
        "--tensor_file",
        type=str,
        help="Path to pickle file containing dictionary of morpheme tensors. "
        + "In training mode and t2v mode, this file will be used as input.",
    )
    arg_parser.add_argument(
        "--vector_file",
        type=str,
        help="Path to pickle file containing dictionary of morpheme vectors. "
        + "In v2t and v2s mode, this file will be used as input.",
    )
    arg_parser.add_argument(
        "--hidden_layer_size", type=int, default="50", help="Size of each hidden layer"
    )
    arg_parser.add_argument(
        "--hidden_layers", type=int, default="2", help="Number of hidden layers"
    )
    arg_parser.add_argument(
        "--learning_rate", type=float, default="0.01", help="Learning rate"
    )
    arg_parser.add_argument(
        "--save_frequency",
        metavar="N",
        type=int,
        default="100",
        help="Save model after every N epochs",
    )
    arg_parser.add_argument(
        "-v",
        "--verbose",
        metavar="LEVEL",
        type=str,
        default="INFO",
        help="Verbosity level",
    )
    arg_parser.add_argument(
        "--cuda_device",
        type=int,
        required=True,
        help="Number specifying which cuda device should be used. A negative number means run on CPU.",
    )
    arg_parser.add_argument(
        "--model_file",
        metavar="FILE",
        type=str,
        help="Previously trained autoencoder model file",
    )
    arg_parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Mode: "
        + "train (train autoencoder), "
        + "t2v (convert tensors to vectors using previously trained autoencoder model), "
        + "v2t (convert vectors to tensors using previously trained autoencoder model), "
        + "s2v (convert strings to vectors using previously trained autoencoder model)",
    )
    arg_parser.add_argument(
        "--output_file",
        metavar="FILE",
        type=str,
        required=True,
        help="Path where final result will be saved",
    )

    return arg_parser



class UnbindingLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        alphabet: Alphabet,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
    ):
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        # super(UnbindingLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.weight = weight
        self.alphabet: Mapping[str, int] = alphabet
        alpha_tensor = []
        for character in self.alphabet:
            i = self.alphabet[character]
            gold_character_vector = torch.zeros(len(self.alphabet))
            gold_character_vector[i] = 1.0
            alpha_tensor.append(gold_character_vector)
        oov = torch.zeros(len(self.alphabet))
        oov[0] = 1.0
        alpha_tensor.insert(0, oov)
        alpha_tensor = torch.stack(alpha_tensor)
        self.register_buffer("alpha_tensor", alpha_tensor)

    def forward(self, input, target):
        distances = torch.einsum(
            "bcm,ac-> bam",
            input.view(input.size(0), len(self.alphabet), -1),
            self.alpha_tensor,
        )
        return cross_entropy(
            distances,
            target.view(target.size(0), len(self.alphabet), -1).argmax(dim=1),
            weight=self.weight,
            ignore_index=self.ignore_index,
        )


def main():

    arg_parser = program_arguments()
    args = arg_parser.parse_args()

    logging.basicConfig(
        level=args.verbose,
        stream=sys.stderr,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s\t%(message)s",
    )

    if args.mode == "train" and args.tensor_file:

        logging.info(
            f"Training autoencoder using tensors in {args.tensor_file} as training data"
        )

        data: Tensors = Tensors.load_from_pickle_file(args.tensor_file, args.alphabet)

        model: Autoencoder = Autoencoder(
            input_dimension_size=data.input_dimension_size,
            hidden_layer_size=args.hidden_layer_size,
            num_hidden_layers=args.hidden_layers,
        )
        
        device = "cpu" if args.cuda_device == -1 else f"cuda:{args.cuda_device}"
        criterion: torch.nn.Module = UnbindingLoss(alphabet=data.alphabet).to(
            device
        )
        optimizer: torch.optim.Optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate
        )

        model.run_training(
            data,
            criterion,
            optimizer,
            args.epochs,
            args.batch_size,
            args.save_frequency,
            args.cuda_device,
        )

        torch.save(model, args.output_file)

    elif args.mode == "t2s" and args.tensor_file:

        import gzip
        import pickle

        logging.info(f"Extracting surface strings from tensors in {args.tensor_file}")

        data: Tensors = Tensors.load_from_pickle_file(args.tensor_file)
        for (
            key_value_tuple
        ) in data.tensor_dict.items():  # type: Tuple[str, torch.Tensor]
            expected: str = key_value_tuple[0]
            tensor: torch.Tensor = key_value_tuple[1]
            actual: str = TensorProductRepresentation.extract_surface_form(
                alphabet=data.alphabet, morpheme_tensor=tensor
            )
    #            print(f"{expected==actual}\t{expected}\t{actual}")

    elif args.mode == "tv2s" and args.tensor_file:

        import gzip
        import pickle

        logging.info(
            f"Constructing vectors from tensors in {args.tensor_file} "
            f"using previously trained model {args.model_file}"
        )

        data: Tensors = Tensors.load_from_pickle_file(args.tensor_file)

        model: Autoencoder = torch.load(args.model_file)

        results: Dict[str, torch.Tensor] = model.run_t2v(  # TODO: Fix type signature
            data, args.batch_size, args.cuda_device
        )

        for key_value_tuple in results.items():
            #            print(key_value_tuple)
            expected: str = key_value_tuple[0]
            #            print(f'Expected string is "{expected}":')
            #            for c in expected:
            #                print(Alphabet.unicode_info(c))
            tensor_shape = data.tensor_dict[expected].shape

            #            print(tensor_dict[expected].shape)
            #            sys.exit(0)
            vector: torch.Tensor = key_value_tuple[1].cpu()

            if args.cuda_device < 0:
                tensor: torch.Tensor = model._apply_output_layer(vector).reshape(
                    tensor_shape
                ).cpu()
            else:
                tensor: torch.Tensor = model._apply_output_layer(vector).reshape(
                    tensor_shape
                ).cuda(device=args.cuda_device)

            #            print(f"Actual tensor:\n{tensor}")

            actual: str = TensorProductRepresentation.extract_surface_form(
                alphabet=data.alphabet, morpheme_tensor=tensor
            )

            #            print(
            #                f"Expected {expected} tensor:\n{data.tensor_dict[expected]}\n{data.tensor_dict[expected].nonzero()}\tActual {tensor}"
            #            )

            #            actual = actual.replace(Alphabet.END_OF_MORPHEME,'')
            #            actual = actual.replace(Alphabet.END_OF_TRANSMISSION, '')
            report = f"{expected==actual}\t{expected}\t{actual}"
            logging.info(report)
            print(report)
            sys.stdout.flush()

    elif args.mode == "v2s" and args.tensor_file:

        import gzip
        import pickle

        logging.info(f"Extracting tensor shape from {args.tensor_file}")
        tensors = Tensors.load_from_pickle_file(args.tensor_file, args.alphabet)
        tensor_shape = tensors.shape()

        logging.info(f"Loading vectors from {args.vector_file}")
        with open(args.vector_file, "rb") as f:
            vectors: Dict[str, List[float]] = pickle.load(f, encoding="utf8")

        logging.info(
            f"Loading previously trained autoencoder model from {args.model_file}"
        )
        model: Autoencoder = torch.load(args.model_file).to(torch.device('cpu'))

        logging.info(
            f"Processing {len(vectors)} vectors through autoencoder output layer..."
        )
        status = 0
        with open(args.output_file, "wt") as output_file:
            for key_value_tuple in vectors.items():
                status += 1
                expected: str = key_value_tuple[0]
                vector: List[float] = key_value_tuple[1]

                if args.cuda_device < 0:
                    tensor: torch.Tensor = model._apply_output_layer(
                        torch.tensor(vector)
                    ).reshape(tensor_shape).cpu()
                else:
                    tensor: torch.Tensor = model._apply_output_layer(
                        torch.tensor(vector)
                    ).reshape(tensor_shape).cuda(device=args.cuda_device)

                actual: str = TensorProductRepresentation.extract_surface_form(
                    alphabet=tensors.alphabet, morpheme_tensor=tensor
                )
                print(f"{expected==actual}\t{expected}\t{actual}", file=output_file)
                output_file.flush()
                logging.info(f"Completed morpheme {status} of {len(vectors)}")

    elif args.mode == "t2v" and args.model_file and args.tensor_file:

        import gzip
        import pickle

        logging.info(
            f"Constructing vectors from tensors in {args.tensor_file} "
            f"using previously trained model {args.model_file}"
        )

        data: Tensors = Tensors.load_from_pickle_file(args.tensor_file, args.alphabet)

        model: Autoencoder = torch.load(args.model_file)

        results: Dict[str, List[float]] = model.run_t2v(
            data, args.batch_size, args.cuda_device
        )

        logging.info(
            f"Dictionary of {len(results)} morphemes to vectors takes {sys.getsizeof(results)} bytes"
        )

        with open(args.output_file, "wb") as output:
            logging.info(
                f"Saving dictionary of {len(results)} morphemes to vectors in {args.output_file}"
            )
            pickle.dump(results, output)

    else:

        arg_parser.print_usage(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
