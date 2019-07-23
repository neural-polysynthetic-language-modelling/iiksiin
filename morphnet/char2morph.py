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
from model import Encoder, Decoder, Seq2Seq
import sys
import pickle
import gzip


def parse_arguments():
    p = argparse.ArgumentParser(description="Hyperparams")
    p.add_argument(
        "-e", "--epochs", type=int, default=100, help="number of epochs for train"
    )
    p.add_argument(
        "-b", "--batch_size", type=int, default=32, help="initial learning rate"
    )
    p.add_argument(
        "--train_data",
        type=str,
        help="morph-segmented version of training data",
        required=True,
    )
    p.add_argument(
        "--dev_data",
        type=str,
        help="morph-segmented version of validation data",
        required=True,
    )
    p.add_argument(
        "--test_data",
        type=str,
        help="morph-segmented version of test data",
        required=True,
    )
    p.add_argument("--tensor_file", type=str, help="pickled dict of morph tensors")
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
        tensor_file,
        morph_delim=">",
        max_num_morphs=10,
        transform=None,
    ):
        self.morph_delim = morph_delim
        self.max_num_morphs = max_num_morphs
        self.transform = transform
        self._init_dataset(tensor_file, morph_file)

    def _init_dataset(self, tensor_file, morph_file):
        with open(morph_file) as mfile:
            segmented_corpus = mfile.readlines()

        with gzip.open(tensor_file) as tfile:
            self.tensor_dict, self.alphabet = pickle.load(tfile, encoding="utf8")
        self.morph_size = next(iter(self.tensor_dict.values())).numel()

        self.corpus = []
        self.length = 0
        for line in segmented_corpus:
            words = line.split()
            for word in words:
                analyzable = True
                for morph in word.split(self.morph_delim):
                    if morph not in self.tensor_dict:
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

    def _get_tensor(self, morph):
        if morph in self.tensor_dict:
            return self.tensor_dict[morph].data
        else:
            return torch.zeros(next(iter(self.tensor_dict.values())).size())

    def _morph_tensors(self, line):
        morphs = [word.split(">") for word in line.split()]
        return torch.stack(
            [
                self._bind_morph_to_position(self._get_tensor(morph), i)
                for word in morphs
                for i, morph in enumerate(word)
            ]
        )

    def _charstoi(self, chars):
        return torch.LongTensor([self.alphabet[c] for c in chars])

    def _input_tensors(self, line):
        chars = line.replace(self.morph_delim, "").replace("\n", "")

        # tensor = torch.zeros(len(chars), len(self.alphabet))
        # ones = torch.ones(len(chars), len(self.alphabet))
        # tensor.scatter_(1, self._charstoi(chars).unsqueeze(1).expand(len(chars),len(self.alphabet)), ones)
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
    # print(morphs[0].size())
    # Need to switch to enforce_sorted if we wanna use ONNX
    chars = pad_sequence(chars)  # , enforce_sorted=False)
    morphs = pad_sequence(morphs)  # , enforce_sorted=False)
    # print(morphs[:,0].size())
    # print(morphs.size())
    # print(morphs[31,0])
    return {"chars": chars, "morphs": morphs}


def load_data(batch_size, tensor_file, *, train_file, dev_file, test_file):
    train_dataset = Char2MorphDataset(train_file, tensor_file)
    test_dataset = Char2MorphDataset(test_file, tensor_file)
    dev_dataset = Char2MorphDataset(dev_file, tensor_file)
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


def train(e, model, optimizer, train_iter, tensor_size, grad_clip, alphabet, device):
    model.train()
    pad = alphabet["\u0004"]
    total_loss = 0

    #    src_tensor = None
    #    tgt_tensor = None

    for b, batch in enumerate(train_iter):
        src = batch["chars"]
        trg = batch["morphs"]

        # print(f"Starting batch {b} with src ({src.shape}) and tgt ({trg.shape})...", end="\n", file=sys.stderr)

        #        del src_tensor
        #        del tgt_tensor

        #        src_tensor = torch.tensor(src.data).to(device)
        #        tgt_tensor = torch.tensor(trg.data).to(device)

        #        src_tensor=src if src_tensor is None else src_tensor.
        #        if src_tensor is None:
        #            src_tensor = src.to(device)
        #            src_tensor.to(device)
        #        else:
        #            print(f"Before, src_tensor.is_cuda == {src_tensor.is_cuda}", file=sys.stderr)
        #            src_tensor.data = src.data.to(device)
        #            print(f"After, src_tensor.is_cuda == {src_tensor.is_cuda}", file=sys.stderr)           #
        #
        #        if tgt_tensor is None:
        #            tgt_tensor = trg.to(device)
        #            tgt_tensor.to(device)
        #        else:
        #            tgt_tensor.data = trg.data.to(device)

        # TODO: Remove volatile=True and replace with with torch.no_grad():
        src_tensor = Variable(src.data.to(device), volatile=True)
        tgt_tensor = Variable(trg.data.to(device), volatile=True)
        del src
        del trg

        output = model(src_tensor, tgt_tensor, teacher_forcing_ratio=0.0)

        # fix
        loss = F.mse_loss(output.view(-1), tgt_tensor.contiguous().view(-1))
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


def evaluate(model, val_iter, tensor_size, alphabet, device):
    model.eval()
    pad = alphabet["\u0004"]
    total_loss = 0
    for b, batch in enumerate(val_iter):
        src = batch["chars"]  # not sure if this is the right interface for batch
        trg = batch["morphs"]
        src, trg = src.to(device), trg.to(device)
        output = model(src, trg)

        # fix
        loss = F.mse_loss(output.view(-1), trg.contiguous().view(-1), ignore_index=pad)
        total_loss += loss.data[0]

        del src
        del trg
        del loss
        del output
    return total_loss / val_iter


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
        args.batch_size,
        args.tensor_file,
        train_file=args.train_file,
        dev_file=args.dev_file,
        test_file=args.test_file,
    )
    print(
        f"[TRAIN]:{len(train_iter)} (dataset:{len(train_iter.dataset)})\t"
        + f"[TEST]:{len(test_iter)} (dataset:{len(test_iter.dataset)})",
        file=sys.stderr,
    )

    #    print(
    #        "[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
    #        % (
    ##            len(train_iter),
    #            len(train_iter.dataset),
    #            len(test_iter),
    #            len(test_iter.dataset),
    #        )
    #    )
    # print("[DE_vocab]:%d [en_vocab]:%d" % (de_size, en_size))

    print("[!] Instantiating models...", file=sys.stderr)
    encoder = Encoder(len(alphabet), embed_size, hidden_size, n_layers=2, dropout=0.5)
    decoder = Decoder(
        morph_size * args.max_num_morphs,
        hidden_size,
        morph_size * args.max_num_morphs,
        n_layers=1,
        dropout=0.5,
    )
    seq2seq = Seq2Seq(encoder, decoder, device).to(device)
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq, file=sys.stderr)

    best_dev_loss = None
    for e in range(1, args.epochs + 1):
        train(
            e,
            seq2seq,
            optimizer,
            train_iter,
            morph_size,
            args.grad_clip,
            alphabet,
            device,
        )
        dev_loss = evaluate(seq2seq, dev_iter, morph_size, alphabet, device)
        print(
            "[Epoch:%d] dev_loss:%5.3f | dev_pp:%5.2fS"
            % (e, dev_loss, math.exp(dev_loss)),
            file=sys.stderr,
        )
        flush()
        # Save the model if the devidation loss is the best we've seen so far.
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
