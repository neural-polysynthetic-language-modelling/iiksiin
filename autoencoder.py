#!/usr/bin/env python3.7

import logging
import torch
import torch.nn
from torch import sigmoid
from torch.nn.functional import relu
import sys

"""Implements an autoencoder to embed Tensor Product Representation tensors into smaller vectors.

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


class Tensors:

    def __init__(self, tensor_dict, alphabet):
        self.tensor_dict = tensor_dict
        self.alphabet = alphabet
        self.morph_size = next(iter(self.tensor_dict.values())).numel()
        self.input_dimension_size = next(iter(self.tensor_dict.values())).view(-1).shape[0]

    @staticmethod
    def load_from_pickle_file(tensor_file: str):
        import pickle
        import gzip
        with gzip.open(tensor_file) as f:
            tensor_dict, alphabet = pickle.load(f, encoding='utf8')
            return Tensors(tensor_dict, alphabet)

    def get_batch_info(self, items_per_batch):
        return BatchInfo(self.tensor_dict.keys(), items_per_batch)
        
    def get_batches(self, items_per_batch, device_number):

        batch_info = self.get_batch_info(items_per_batch)
        
        for batch_of_morphemes in batch_info:
            tensor = torch.zeros(items_per_batch, self.input_dimension_size)
            for n, morpheme in enumerate(batch_of_morphemes):
                tensor[n] = self.tensor_dict[morpheme].view(-1)

            if 0 <= device_number < torch.cuda.device_count():
                yield tensor.cuda(device_number)
            else:
                yield tensor.cpu()

                
class BatchInfo:
    """Store information necessary to identify a morpheme given a batch number and an index within that batch."""
    
    def __init__(self, morphemes, items_per_batch):
        sizes_dict = dict()
        for morpheme in morphemes:
            length = len(morpheme)
            if length not in sizes_dict:
                sizes_dict[length] = list()
            sizes_dict[length].append(morpheme)

        self._batches = list()
        current_batch = list()
        for length in sorted(sizes_dict.keys()):
            for morpheme in sizes_dict[length]:
                if len(current_batch) == items_per_batch:
                    self._batches.append(current_batch)
                    current_batch = list()
                current_batch.append(morpheme)
        if len(current_batch) > 0:
            self._batches.append(current_batch)

    def __getitem__(self, index):
        return self._batches[index]

    def __iter__(self):
        return iter(self._batches)
    

class Autoencoder(torch.nn.Module):

    def __init__(self, input_dimension_size: int, hidden_layer_size: int, num_hidden_layers: int):
        super().__init__()
        self.input_dimension_size: int = input_dimension_size
        self.hidden_layer_size: int = hidden_layer_size
        self.hidden_layers = torch.nn.ModuleList()
        for n in range(num_hidden_layers):
            if n == 0:
                self.hidden_layers.append(torch.nn.Linear(self.input_dimension_size, self.hidden_layer_size))
            else:
                self.hidden_layers.append(torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size))
        self.output_layer = torch.nn.Linear(self.hidden_layer_size, self.input_dimension_size)
        
    def forward(self, input_layer):
        final_hidden_layer = self._apply_hidden_layers(input_layer)
        output_layer = self._apply_output_layer(final_hidden_layer)
        return output_layer

    def _apply_hidden_layers(self, input_layer):
        previous_layer = input_layer

        for hidden in self.hidden_layers:
            current_layer = relu(hidden(previous_layer))
            previous_layer = current_layer

        return current_layer

    def _apply_output_layer(self, hidden_layer):
        return sigmoid(self.output_layer(hidden_layer))

    def run_t2v(self, data, max_items_per_batch: int, cuda_device: int):

        self.eval()

        if cuda_device < 0:
            self.cpu()
        else:
            self.cuda(device=cuda_device)

        results = dict()
        batch_info = data.get_batch_info(max_items_per_batch)
        for batch_number, data_on_device in enumerate(data.get_batches(items_per_batch=max_items_per_batch,
                                                                       device_number=cuda_device)):

            batch_of_results = self._apply_hidden_layers(data_on_device)

            morphemes = batch_info[batch_number]
            number_of_results = batch_of_results.shape[0]
            for n in range(min(number_of_results, len(morphemes))):
                morpheme = morphemes[n]
                tensor = batch_of_results[n]
                results[morpheme] = tensor

        return results
    
    def run_training(
            self,
            data: Tensors,
            criterion,
            optimizer,
            num_epochs: int,
            max_items_per_batch: int,
            save_frequency: int,
            cuda_device: int
    ):
        self.train()

        if cuda_device < 0:
            self.cpu()
        else:
            self.cuda(device=cuda_device)
        
        for epoch in range(1, num_epochs+1):
            optimizer.zero_grad()

            total_loss = 0
            
            for data_on_device in data.get_batches(items_per_batch=max_items_per_batch, device_number=cuda_device):

                prediction = self(data_on_device)

                # Compute Loss
                loss = criterion(prediction.squeeze(), data_on_device)
                total_loss += loss.item()

                loss.backward()

                
            logging.info(f"Epoch {str(epoch).zfill(len(str(num_epochs)))}\ttrain loss: {loss.item()}")
            
            if epoch % save_frequency == 0 or epoch == num_epochs:
                logging.info(f"Saving model to model_at_epoch_{str(epoch).zfill(len(str(num_epochs)))}.pt")
                torch.save(self, f"model_at_epoch_{str(epoch).zfill(len(str(num_epochs)))}.pt")

            # Backward pass
            optimizer.step()


def program_arguments():
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Autoencode tensor product representations of each morpheme."
    )
    arg_parser.add_argument(
        "--epochs",
        type=int,
         default=200,
        help="Number of epochs to run during training.",
    )
    arg_parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size",
    )
    arg_parser.add_argument(
        "--tensor_file",
        type=str,
        help="Path to pickle file containing dictionary of morpheme tensors. " +
             "In training mode and t2v mode, this file will be used as input.",
    )
    arg_parser.add_argument(
        "--hidden_layer_size",
        type=int,
        default="50",
        help="Size of each hidden layer",
    )
    arg_parser.add_argument(
        "--hidden_layers",
        type=int,
        default="2",
        help="Number of hidden layers",
    )
    arg_parser.add_argument(
        "--learning_rate",
        type=float,
        default="0.01",
        help="Learning rate",
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
        help="Verbosity level"
    )
    arg_parser.add_argument(
        "--cuda_device",
        type=int,
        required=True,
        help="Number specifying which cuda device should be used. A negative number means run on CPU."
    )
    arg_parser.add_argument(
        "--model_file",
        metavar="FILE",
        type=str,
        help="Previously trained autoencoder model file"
    )
    arg_parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Mode: train (train autoencoder), t2v (convert tensors to vectors using previously trained autoencoder model), v2t (convert vectors to tensors using previously trained autoencoder model), s2v (convert strings to vectors using previously trained autoencoder model)"
    )
    arg_parser.add_argument(
        "--output_file",
        metavar="FILE",
        type=str,
        required=True,
        help="Path where final result will be saved"
    )
    
    return arg_parser


def main():
    
    arg_parser = program_arguments()
    args = arg_parser.parse_args()

    logging.basicConfig(level=args.verbose, stream=sys.stderr,
                        datefmt="%Y-%m-%d %H:%M:%S", format="%(asctime)s\t%(message)s")
    
    if args.mode == "train" and args.tensor_file:

        logging.info(f"Training autoencoder using tensors in {args.tensor_file} as training data")

        data = Tensors.load_from_pickle_file(args.tensor_file)

        model = Autoencoder(input_dimension_size=data.input_dimension_size,
                            hidden_layer_size=args.hidden_layer_size,
                            num_hidden_layers=args.hidden_layers)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

        model.run_training(data,
                           criterion,
                           optimizer,
                           args.epochs,
                           args.batch_size,
                           args.save_frequency,
                           args.cuda_device)

        torch.save(model, args.output_file)
        
    elif args.mode == "t2v" and args.model_file and args.tensor_file:

        import gzip
        import pickle
        
        logging.info(f"Constructing vectors from tensors in {args.tensor_file} "
                     "using previously trained model {args.model_file}")

        data = Tensors.load_from_pickle_file(args.tensor_file)

        model = torch.load(args.model_file)

        results = model.run_t2v(data, args.batch_size, args.cuda_device)

        with gzip.open(args.output_file, "wb") as output:
            logging.info(f"Saving gzipped dictionary of morphemes to vectors in {args.output_file}")
            pickle.dump(results, output)
        
    else:

        arg_parser.print_usage(file=sys.stderr)
        sys.exit(1)

        
if __name__ == "__main__":
    main()
