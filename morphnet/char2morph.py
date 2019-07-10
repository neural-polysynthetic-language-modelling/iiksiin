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

import pickle
import gzip


def parse_arguments():
    p = argparse.ArgumentParser(description="Hyperparams")
    p.add_argument("-epochs", type=int, default=100, help="number of epochs for train")
    p.add_argument("-batch_size", type=int, default=32, help="initial learning rate")
    p.add_argument(
        "-corpus_dir", type=str, help="seperated morpheme version of text corpus"
    )
    p.add_argument("-tensor_file", type=str, help="pickled dict of morph tensors")
    p.add_argument("-max_num_morphs", type=int, default=10, help="maximum number of morphemes in a word")
    p.add_argument(
        "-lr", type=float, default=0.0001, help="in case of gradient explosion"
    )
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='in case of gradient explosion')
    return p.parse_args()


class Char2MorphDataset(Dataset):
    def __init__(self, morph_file, tensor_file, morph_delim=">", max_num_morphs=10, transform=None):
        self.morph_delim=morph_delim
        self.max_num_morphs=max_num_morphs
        self.transform = transform
        self._init_dataset(tensor_file, morph_file)

    def _init_dataset(self, tensor_file, morph_file):
        with open(morph_file) as mfile:
            self.segmented_corpus = mfile.readlines()

        with gzip.open(tensor_file) as tfile:
            self.tensor_dict, self.alphabet = pickle.load(tfile, encoding='utf8')
        self.morph_size = next(iter(self.tensor_dict.values())).numel()

        self.length = 0
        for line in self.segmented_corpus:
            self.length += 1

    def __len__(self):
        return self.length

    def _bind_morph_to_position(self, morph, index):
        position = torch.zeros(self.max_num_morphs)
        position[index] = 1
        return torch.einsum("...j,k->...jk", (morph, position))

    def _morph_tensors(self, line):
        morphs = [word.split('>') for word in line.split()]
        return torch.stack(
                [self._bind_morph_to_position(self.tensor_dict[morph].data, i) for word in morphs for i, morph in enumerate(word)]
        )

    def _charstoi(self, chars):
        return torch.LongTensor([self.alphabet[c] for c in chars])

    def _input_tensors(self, line):
        chars = line.replace(self.morph_delim, "")[:-1]

        # tensor = torch.zeros(len(chars), len(self.alphabet))
        # ones = torch.ones(len(chars), len(self.alphabet))
        # tensor.scatter_(1, self._charstoi(chars).unsqueeze(1).expand(len(chars),len(self.alphabet)), ones)
        return self._charstoi(chars)

    def __getitem__(self, idx):
        sample = {
            "chars": self._input_tensors(self.segmented_corpus[idx]),
            "morphs": self._morph_tensors(self.segmented_corpus[idx]),
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

def char2morph_collate(batch):
    chars = [item['chars'] for item in batch]
    morphs = [item['morphs'] for item in batch]
    # Need to switch to enforce_sorted if we wanna use ONNX
    chars = pad_sequence(chars)#, enforce_sorted=False)
    morphs = pad_sequence(morphs)#, enforce_sorted=False)
    return {'chars': chars, 'morphs': morphs}

def load_data(batch_size, morph_dir, tensor_file):
    train_dataset = Char2MorphDataset(morph_dir + "/train.txt", tensor_file)
    test_dataset = Char2MorphDataset(morph_dir + "/test.txt", tensor_file)
    dev_dataset = Char2MorphDataset(morph_dir + "/dev.txt", tensor_file)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=char2morph_collate)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, collate_fn=char2morph_collate)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=char2morph_collate)
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
        src = batch['chars']
        trg = batch['morphs']
        src = Variable(src.data.cuda(), volatile=True)
        trg = Variable(trg.data.cuda(), volatile=True)
        output = model(src, trg, teacher_forcing_ratio=0.0)

        # fix
        loss = F.mse_loss(output.view(-1), trg.contiguous().view(-1))
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
    train_iter, dev_iter, test_iter, alphabet, morph_size = load_data(
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
    encoder = Encoder(len(alphabet), embed_size, hidden_size, n_layers=2, dropout=0.5)
    decoder = Decoder(morph_size*args.max_num_morphs, hidden_size, morph_size*args.max_num_morphs, n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).cuda()
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    best_dev_loss = None
    for e in range(1, args.epochs + 1):
        train(e, seq2seq, optimizer, train_iter, morph_size, args.grad_clip, alphabet)
        dev_loss = evaluate(seq2seq, test_iter, morph_size, alphabet)
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
    test_loss = evaluate(seq2seq, test_iter, morph_size, alphabet)
    print("[TEST] loss:%5.2f" % test_loss)


if __name__ == "__main__":
    main()
