# based on https://github.com/keon/seq2seq

import os
import math
import argparse
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from morphnet.model import Encoder, Decoder, Seq2Seq
from iiksiin import Alphabet, Dimension, OneHotVector, Shape
import sys
import pickle
from autoencoder import UnbindingLoss, TrueTensorRetreiver


def parse_arguments():
    p = argparse.ArgumentParser(description="Hyperparams")
    p.add_argument(
        "--train_file",
        type=str,
        help="morph-segmented version of training data",
        required=True,
    )
    p.add_argument(
        "--dev_file",
        type=str,
        help="morph-segmented version of validation data",
        required=True,
    )
    p.add_argument(
        "--test_file",
        type=str,
        help="morph-segmented version of test data",
        required=True,
    )
    p.add_argument(
        "--vector_file", type=str, help="pickled dict of morph vectors", required=True
    )
    p.add_argument(
        "--alphabet_file",
        type=str,
        help="Alphabet defining padding characters and an stoi",
        required=True,
    )
    p.add_argument(
        "--autoencoder_model",
        type=str,
        help="path to dumped torch file of autoencoder params.",
        required=True,
    )
    p.add_argument(
        "-m",
        "--morph_delim",
        type=str,
        default=">",
        help="character delimiting morph boundaries (default '>')",
    )
    p.add_argument(
        "-e", "--epochs", type=int, default=100, help="number of epochs for train"
    )
    p.add_argument(
        "-b", "--batch_size", type=int, default=32, help="initial learning rate"
    )
    p.add_argument(
        "--max_num_morphs",
        type=int,
        default=10,
        help="maximum number of morphemes in a word",
    )
    p.add_argument(
        "--lr", type=float, default=0.0001, help="in case of gradient explosion"
    )
    p.add_argument(
        "--grad_clip", type=float, default=10.0, help="in case of gradient explosion"
    )
    return p.parse_args()


class Char2MorphDataset(Dataset):
    def __init__(
        self,
        morph_file,
        vector_dict,
        alphabet,
        morph_delim,
        max_num_morphs,
        transform=None,
    ):
        self.morph_delim = morph_delim
        self.max_num_morphs = max_num_morphs
        self.vector_dict = vector_dict
        self.alphabet = alphabet
        self.transform = transform

        self._init_dataset(vector_dict, morph_file)

    def _init_dataset(self, vector_dict, morph_file):
        with open(morph_file) as mfile:
            segmented_corpus = mfile.readlines()

        self.morph_size = len(next(iter(self.vector_dict.values())))

        self.corpus = []
        self.length = 0
        for line in segmented_corpus:
            words = line.split()
            for word in words:
                analyzable = True
                for morph in word.split(self.morph_delim):
                    if morph not in self.vector_dict:
                        analyzable = False
                        break
                if not analyzable:
                    break
                self.corpus.append(word)
                self.length += 1
        self.corpus.sort(reverse=False, key=len)

    def __len__(self):
        return self.length

    def _bind_morph_to_position(self, morph, index):
        position = torch.zeros(self.max_num_morphs)
        position[index] = 1
        return torch.einsum("...j,k->...jk", (morph, position))

    def _get_vector(self, morph):
        if morph in self.vector_dict:
            return torch.FloatTensor(self.vector_dict[morph])
        else:
            return torch.zeros(next(iter(self.vector_dict.values())).size())

    def _morph_tensors(self, line):
        morphs = [word.split(self.morph_delim) for word in line.split()]
        return torch.stack(
            [
                # self._bind_morph_to_position(self._get_vector(morph), i)
                self._get_vector(morph)
                for word in morphs
                for i, morph in enumerate(word)
            ]
        )

    def _charstoi(self, chars):
        return torch.LongTensor([self.alphabet[c] for c in chars])

    def _input_tensors(self, line):
        chars = line.replace(self.morph_delim, "").replace("\n", "")
        return self._charstoi(chars)

    def __getitem__(self, idx):
        sample = {
            "chars": self._input_tensors(self.corpus[idx]),
            "morphs": self._morph_tensors(self.corpus[idx]),
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


def char2morph_collate(batch):
    chars = [item["chars"] for item in batch]
    morphs = [item["morphs"] for item in batch]
    chars = pad_sequence(chars)
    morphs = pad_sequence(morphs)
    return {"chars": chars, "morphs": morphs}


def load_data(
    *,
    train_file,
    dev_file,
    test_file,
    vector_file,
    batch_size,
    morph_delim,
    alphabet_file,
    max_num_morphs,
):
    with open(vector_file, "rb") as vfile:
        vector_dict = pickle.load(vfile, encoding="utf8")
    with open(alphabet_file, "rb") as afile:
        alphabet = pickle.load(afile, encoding="utf8")

    train_dataset = Char2MorphDataset(
        train_file,
        vector_dict,
        alphabet,
        morph_delim=morph_delim,
        max_num_morphs=max_num_morphs,
    )
    dev_dataset = Char2MorphDataset(
        dev_file,
        vector_dict,
        alphabet,
        morph_delim=morph_delim,
        max_num_morphs=max_num_morphs,
    )
    test_dataset = Char2MorphDataset(
        test_file,
        vector_dict,
        alphabet,
        morph_delim=morph_delim,
        max_num_morphs=max_num_morphs,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=char2morph_collate
    )
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=batch_size, collate_fn=char2morph_collate
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=char2morph_collate
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        train_dataset.alphabet,
        train_dataset.morph_size,
    )


def train(
    e,
    model,
    autoencoder,
    optimizer,
    train_iter,
    tensor_size,
    grad_clip,
    alphabet,
    device,
):
    model.train()
    pad = alphabet[alphabet.__class__.END_OF_TRANSMISSION]
    total_loss = 0
    criterion = UnbindingLoss(alphabet=alphabet).to(device)
    retreiver = TrueTensorRetreiver(alphabet=alphabet)

    for b, batch in enumerate(train_iter):
        src = batch["chars"]
        trg = batch["morphs"]
        # TODO: Remove volatile=True and replace with with torch.no_grad():
        optimizer.zero_grad()
        src_tensor = Variable(src.data.to(device), volatile=True)
        tgt_tensor = Variable(trg.data.to(device), volatile=True)
        del src
        del trg

        output = model(src_tensor, tgt_tensor, teacher_forcing_ratio=0.0)
        tensor = autoencoder._apply_output_layer(output)
        correct = autoencoder._apply_output_layer(tgt_tensor)
        correct = retreiver.retreive(correct)
        # fix
        loss = criterion(output, correct)
        loss.backward()
        optimizer.step()
        total_loss += (
            loss.data
        )  # TODO: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
        del output
        del loss
        #     print(f"batch {b} complete.", file=sys.stderr)
        sys.stderr.flush()
        if b % 100 == 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f][pp:%5.2f]" % (b, total_loss, math.exp(total_loss)))
            sys.stdout.flush()
            total_loss = 0


def evaluate(model, autoencoder, val_iter, tensor_size, alphabet, device):
    model.eval()
    pad = alphabet[alphabet.__class__.END_OF_TRANSMISSION]
    total_loss = 0
    for b, batch in enumerate(val_iter):
        src = batch["chars"]
        trg = batch["morphs"]
        src, trg = src.to(device), trg.to(device)
        output = model(src, trg)

        # fix
        loss = F.mse_loss(output, trg)
        total_loss += loss.data

        del src
        del trg
        del loss
        del output
    return total_loss / len(val_iter)


def segment_corpus(model, train_iter, val_iter, test_iter, alphabet, device):
    model.eval()
    pad = alphabet[alphabet.__class__.END_OF_TRANSMISSION]
    for b, batch in enumerate(train_iter):
        src = batch["chars"]
        trg = batch["morphs"]
        src, trg = src.to(device), trg.to(device)
        output = model(src, trg)

        del src
        del trg


def debug():
    import sys

    flush()
    sys.exit(-1)


def flush():
    sys.stderr.flush
    sys.stdout.flush()


def get_freer_gpu():
    import os

    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >cuda")
    memory_available = [int(x.split()[2]) for x in open("cuda", "r").readlines()]
    import numpy as np

    return np.argmax(memory_available)


def main():

    import sys

    args = parse_arguments()
    hidden_size = 512
    embed_size = 256
    device = torch.device("cpu")
    device = torch.device(
        f"cuda:{get_freer_gpu()}" if torch.cuda.is_available() else "cpu"
    )

    print("[!] preparing dataset...", file=sys.stderr)
    train_iter, dev_iter, test_iter, alphabet, morph_size = load_data(
        train_file=args.train_file,
        dev_file=args.dev_file,
        test_file=args.test_file,
        vector_file=args.vector_file,
        alphabet_file=args.alphabet_file,
        batch_size=args.batch_size,
        morph_delim=args.morph_delim,
        max_num_morphs=args.max_num_morphs,
    )
    print(
        f"[TRAIN]:{len(train_iter)} (dataset:{len(train_iter.dataset)})\t"
        + f"[TEST]:{len(test_iter)} (dataset:{len(test_iter.dataset)})",
        file=sys.stderr,
    )

    print("[!] Instantiating models...", file=sys.stderr)
    encoder = Encoder(len(alphabet), embed_size, hidden_size, n_layers=2, dropout=0.5)
    decoder = Decoder(morph_size, hidden_size, morph_size, n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder, device).to(device)
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq, file=sys.stderr)
    autoencoder = torch.load(args.autoencoder_model)
    best_dev_loss = None
    for e in range(1, args.epochs + 1):
        print("HELOOOOOOOO")
        print(next(seq2seq.parameters()))
        train(
            e,
            seq2seq,
            autoencoder,
            optimizer,
            train_iter,
            morph_size,
            args.grad_clip,
            alphabet,
            device,
        )
        dev_loss = evaluate(
            seq2seq, autoencoder, dev_iter, morph_size, alphabet, device
        )
        print(
            "[Epoch:%d] dev_loss:%5.3f | dev_pp:%5.2fS"
            % (e, dev_loss, math.exp(dev_loss)),
            file=sys.stderr,
        )
        flush()
        # Save the model if the dev loss is the best we've seen so far.
        if not best_dev_loss or dev_loss < best_dev_loss:
            print("[!] saving model...", file=sys.stderr)
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(seq2seq.state_dict(), "./.save/char2morph_%d.pt" % (e))
            best_dev_loss = dev_loss
    test_loss = evaluate(seq2seq, test_iter, morph_size, alphabet, device)
    print("[TEST] loss:%5.2f" % test_loss, file=sys.stderr)
    flush()


if __name__ == "__main__":
    main()
