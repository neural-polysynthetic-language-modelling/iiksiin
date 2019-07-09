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
from model import Encoder, Decoder, Seq2Seq
import pickle


def parse_arguments():
    p = argparse.ArgumentParser(description="Hyperparams")
    p.add_argument("-epochs", type=int, default=100, help="number of epochs for train")
    p.add_argument("-batch_size", type=int, default=32, help="initial learning rate")
    p.add_argument(
        "-corpus_dir", type=str, help="seperated morpheme version of text corpus"
    )
    p.add_argument("-tensor_file", type=str, help="pickled dict of morph tensors")
    p.add_argument(
        "-lr", type=float, default=0.0001, help="in case of gradient explosion"
    )
    return p.parse_args()


class Char2MorphDataset(Dataset):
    def __init__(self, char_file, morph_file, morph_delim=">", transform=None):
        self.transform = transform
        self._init_dataset(tensor_file, morph_file)

    def _init_dataset(self, tensor_file, morph_file):
        with open(morph_file) as morph_file:
            self.segmented_corpus = morph.readlines()

        with open(tensor_file) as tfile:
            self.tensor_dict, self.alphabet = pickle.load(tfile)
        self.morph_size = self.tensor_dict.devues()[0].numel()

        self.length = 0
        for line in self.segmented_corpus:
            self.length += 1

    def __len__(self):
        return self.length

    def _bind_morph_to_position(morph, index):
        position = torch.zeros(self.max_num_morphs)
        position[index] = 1
        return torch.einsum("...j, k -> ...jk", morph, index)

    def _morph_tensors(self, line):
        morphs = line.split(" ").split(self.morph_delim)
        return torch.stack(
            [
                self._bind_morph_to_position(self.tensor_dict[morph].data, i)
                for i, morph in enumerate(morphs)
            ]
        )

    def _charstoi(self, chars):
        return torch.LongTensor([self.alphabet[c] for c in chars])

    def _input_tensors(self, line):
        chars = line.replace(self.morph_delim, "").split(" ")

        tensor = torch.zeros(len(chars), len(self.alphabet))
        ones = torch.ones(len(chars), len(self.alphabet))
        tensor.scatter_(1, self._charstoi(schars), ones)
        return tensor

    def __getitem__(self, idx):
        sample = {
            "chars": self._input_tensors(self.segmented_corpus[idx]),
            "morphs": self._morph_tensors(self.segmented_corpus[idx]),
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


def load_data(batch_size, char_file, morph_file, tensor_file):
    train_dataset = Char2MorphDataset(morph_dir + "/train.txt", tensor_file)
    test_dataset = Char2MorphDataset(morph_dir + "/test.txt", tensor_file)
    dev_dataset = Char2MorphDataset(morph_dir + "/dev.txt", tensor_file)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        train_dataset.alphabet,
        train_dataset.morph_size,
    )


def train(e, model, optimizer, train_iter, tensor_size, grad_clip, alphabet):
    model.train()
    pad = alphabet["\u0004"]
    total_loss = 0

    for b, batch in enumerate(train_iter):
        print(batch)
        src, len_src = batch.chars  # not sure if this is the right interface for batch
        trg, len_trg = batch.morphs
        src = Variable(src.data.cuda(), volatile=True)
        src = Variable(tgt.data.cuda(), volatile=True)
        output = model(src, trg, teacher_forcing_ratio=0.0)

        # fix
        loss = F.mse_loss(output.view(-1), trg.contiguous().view(-1), ignore_index=pad)
        total_loss += loss.data[0]
    return total_loss / len(val_iter)


def evaluate(e, model, optimizer, val_iter, tensor_size, alphabet):
    model.eval()
    pad = alphabet["\u0004"]
    total_loss = 0
    for b, batch in enumerate(train_iter):
        src, len_src = batch.chars  # not sure if this is the right interface for batch
        trg, len_trg = batch.morphs
        src, trg = src.cuda(), trg.cuda()
        src = Variable()
        optimizer.zero_grad()
        output = model(src, trg)

        # fix
        loss = F.mse_loss(output.view(-1), trg.contiguous().view(-1), ignore_index=pad)
        loss.backward()
        clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.data[0]

        if b % 100 == 0 and b != 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f][pp:%5.2f]" % (b, total_loss, math.exp(total_loss)))
            total_loss = 0


def main():
    args = parse_arguments()
    hidden_size = 512
    embed_size = 256
    assert torch.cuda.is_available()

    print("[!] preparing dataset...")
    train_iter, dev_iter, test_iter, alphabet, morph_size = load_dataset(
        args.batch_size, args.corpus_dir, args.tensor_file
    )
    print(
        "[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
        % (
            len(train_iter),
            len(train_iter.dataset),
            len(test_iter),
            len(test_iter.dataset),
        )
    )
    # print("[DE_vocab]:%d [en_vocab]:%d" % (de_size, en_size))

    print("[!] Instantiating models...")
    encoder = Encoder(char_size, embed_size, hidden_size, n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, morph_size, n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).cuda()
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    best_dev_loss = None
    for e in range(1, args.epochs + 1):
        train(e, seq2seq, optimizer, train_iter, en_size, args.grad_clip, alphabet)
        dev_loss = evaluate(seq2seq, dev_iter, en_size, DE, EN)
        print(
            "[Epoch:%d] dev_loss:%5.3f | dev_pp:%5.2fS"
            % (e, dev_loss, math.exp(dev_loss))
        )

        # Save the model if the devidation loss is the best we've seen so far.
        if not best_dev_loss or dev_loss < best_dev_loss:
            print("[!] saving model...")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(seq2seq.state_dict(), "./.save/char2morph_%d.pt" % (e))
            best_dev_loss = dev_loss
    test_loss = evaluate(seq2seq, test_iter, en_size, DE, EN)
    print("[TEST] loss:%5.2f" % test_loss)
