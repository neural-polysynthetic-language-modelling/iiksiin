#!/usr/bin/env python3.7

import torch
import torch.nn
import sys
from sys import stderr
from typing import Dict


class FeedForwardNetwork(torch.nn.Module):

    def __init__(self, input_dimension_size: int, hidden_layer_size: int):
        super().__init__()
        self.input_dimension_size: int = input_dimension_size
        self.hidden_layer_size: int = hidden_layer_size
        self.create_hidden_layer1 = torch.nn.Linear(self.input_dimension_size, self.hidden_layer_size)
        self.create_hidden_layer2 = torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.apply_hidden_layer_activation_function = torch.nn.ReLU()
        self.create_output_layer = torch.nn.Linear(self.hidden_layer_size, self.input_dimension_size)
        self.apply_output_layer_sigmoid_function = torch.nn.Sigmoid()
        
    def forward(self, input_layer):
        final_hidden_layer = self.calculate_hidden_layer(input_layer)
        output_layer = self.calculate_output_layer(final_hidden_layer)
        return output_layer

    def calculate_hidden_layer(self, input_layer):
        unactivated_hidden_layer1 = self.create_hidden_layer1(input_layer)
        hidden_layer1 = self.apply_hidden_layer_activation_function(unactivated_hidden_layer1)
        unactivated_hidden_layer2 = self.create_hidden_layer2(hidden_layer1)
        hidden_layer2 = self.apply_hidden_layer_activation_function(unactivated_hidden_layer2)
        return hidden_layer2

    def calculate_output_layer(self, hidden_layer):
        unactivated_output_layer = self.create_output_layer(hidden_layer)
        output_layer = self.apply_output_layer_sigmoid_function(unactivated_output_layer)
        return output_layer


class Data:

    def __init__(self, tensor_file: str):
        import pickle
        import gzip
        with gzip.open(tensor_file) as f:
            self.tensor_dict, self.alphabet = pickle.load(f, encoding='utf8')
        
        self.morph_size = next(iter(self.tensor_dict.values())).numel()
        self.input_dimension_size = next(iter(self.tensor_dict.values())).view(-1).shape[0]
        
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

        batches_of_tensors = [[self.tensor_dict[morpheme].view(-1) for morpheme in batch_of_morphemes] for batch_of_morphemes in batches_of_morphemes]
        
        return batches_of_morphemes, batches_of_tensors


def main():
    from datetime import datetime

    print(f"Starting program...", file=sys.stderr)
    sys.stderr.flush()

    epochs = 100000
    max_items_per_batch=100

    tensor_file="tensors.pickle"

    data = Data(tensor_file)

    print(f"Loading data...", file=sys.stderr, end="")
    sys.stderr.flush()
    batches_of_morphemes, batches_of_tensors = data.get_batches(items_per_batch=max_items_per_batch)
    print(f" {len(batches_of_morphemes)} batches loaded", file=sys.stderr)
    sys.stderr.flush()

    model = FeedForwardNetwork(input_dimension_size=data.input_dimension_size,
                               hidden_layer_size=50).cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        total_loss = 0
        for batch_number, list_of_tensors in enumerate(batches_of_tensors):
            training_data = torch.zeros(max_items_per_batch, data.input_dimension_size)
            for tensor_number, tensor in enumerate(list_of_tensors):
                training_data[tensor_number] = tensor.data

            training_data = training_data.cuda()
            prediction = model(training_data)

            # Compute Loss
            loss = criterion(prediction.squeeze(), training_data)
            total_loss += loss.item()

            loss.backward()

        print(f"{datetime.now()}\tEpoch {str(epoch).zfill(len(str(epochs)))}\ttrain loss: {loss.item()}", file=sys.stderr)

        sys.stderr.flush()

        if epoch % 100 == 0:
            torch.save(model, f"model_at_epoch_{str(epoch).zfill(len(str(epochs)))}.pt")

        # Backward pass
        optimizer.step()


if __name__ == "__main__":
    main()
