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

    def get_batches(self, items_per_batch):

        sizes_dict = dict()
        for morpheme in self.tensor_dict.keys():
            length = len(morpheme)
            if length not in sizes_dict:
                sizes_dict[length] = list()
            sizes_dict[length].append(morpheme)

        batches_of_morphemes = list()
        batch_of_morphemes = list()
        for length in sorted(sizes_dict.keys()):
            for morpheme in sizes_dict[length]:
                if len(batch_of_morphemes) == items_per_batch:
                    batches_of_morphemes.append(batch_of_morphemes)
                    batch_of_morphemes = list()
                batch_of_morphemes.append(morpheme)
        if len(batch_of_morphemes) > 0:
            batches_of_morphemes.append(batch_of_morphemes)

        batches_of_tensors = [[self.tensor_dict[morpheme].view(-1) for morpheme in batch_of_morphemes] for
                              batch_of_morphemes in batches_of_morphemes]

        return batches_of_morphemes, batches_of_tensors


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
#        self.hidden_layers.extend([torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size)] * )
#        self.create_hidden_layer1 = torch.nn.Linear(self.input_dimension_size, self.hidden_layer_size)
#        self.create_hidden_layer2 = torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
#        self.apply_hidden_layer_activation_function = torch.nn.ReLU()
#        self.create_output_layer = torch.nn.Linear(self.hidden_layer_size, self.input_dimension_size)
#        self.apply_output_layer_sigmoid_function = torch.nn.Sigmoid()
        
    def forward(self, input_layer):
        final_hidden_layer = self.apply_hidden_layers(input_layer)
        output_layer = self.apply_output_layer(final_hidden_layer)
        return output_layer

    def apply_hidden_layers(self, input_layer):
        previous_layer = input_layer

        for hidden in self.hidden_layers:
            current_layer = relu(hidden(previous_layer))
            previous_layer = current_layer

        return current_layer

    def apply_output_layer(self, hidden_layer):
        return sigmoid(self.output_layer(hidden_layer))
#        unactivated_output_layer = self.create_output_layer(hidden_layer)
#        output_layer = self.apply_output_layer_sigmoid_function(unactivated_output_layer)
#        return output_layer

    def run_training(
            self,
            data: Tensors,
            criterion,
            optimizer,
            num_epochs: int,
            max_items_per_batch: int,
            save_frequency: int
    ):
        from datetime import datetime

        print(f"Loading data...", file=sys.stderr, end="")
        sys.stderr.flush()
        batches_of_morphemes, batches_of_tensors = data.get_batches(items_per_batch=max_items_per_batch)
        print(f" {len(batches_of_morphemes)} batches loaded", file=sys.stderr)
        sys.stderr.flush()

        self.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            total_loss = 0
            for batch_number, list_of_tensors in enumerate(batches_of_tensors):
                training_data = torch.zeros(max_items_per_batch, data.input_dimension_size)
                for tensor_number, tensor in enumerate(list_of_tensors):
                    training_data[tensor_number] = tensor.data

                training_data = training_data.cuda()
                prediction = self(training_data)

                # Compute Loss
                loss = criterion(prediction.squeeze(), training_data)
                total_loss += loss.item()

                loss.backward()

            print(f"{datetime.now()}\tEpoch {str(epoch).zfill(len(str(num_epochs)))}\ttrain loss: {loss.item()}",
                  file=sys.stderr)

            sys.stderr.flush()

            if epoch % save_frequency == 0:
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
        help="Path to pickle file containing dictionary of morpheme tensors.",
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

    return arg_parser


def main():

    arg_parser = program_arguments()
    args = arg_parser.parse_args()

    if args.tensor_file:

        logging.info("Starting program...")

        sys.stderr.flush()

        data = Tensors.load_from_pickle_file(args.tensor_file)

        model = Autoencoder(input_dimension_size=data.input_dimension_size,
                            hidden_layer_size=args.hidden_layer_size,
                            num_hidden_layers=args.hidden_layers).cuda()

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

        model.run_training(data, criterion, optimizer, args.epochs, args.batch_size, args.save_frequency)

    else:

        arg_parser.print_usage(file=sys.stderr)
        sys.exit(1)

        
if __name__ == "__main__":
    main()
