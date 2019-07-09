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
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='initial learning rate')
    p.add_argument('-txt_file', type=str,
                   help='corpus used to create morph tensors')
    p.add_argument('-morph_file', type=str,
                    help='dash-split morpheme version of text corpus')
    p.add_argument('-tensor_file', type=str,
                    help='pickled dict of morph tensors')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='in case of gradient explosion')
    return p.parse_args()

class Char2MorphDataset(Dataset):
    def __init__(self, char_file, morph_file, morph_delim='-', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.char_file = char_file
        self.morph_file = morph_file
        with open(tensor_file) as tfile:
            self.tensor_dict = pickle.load(tfile)
        self.morph_delim = morph_delim
        self.alphabet = ???
        self.transform = transform
        self._init_dataset()

    def _init_dataset(self):
        self.length = 0
        for line in open(char_file):
            self.length += 1

    def __len__(self):
        return self.length

    def _morph_tensors(self, morphs):
        return stack([self.tensor_dict[morph] for morph in morphs])

    def _charstoi(self, chars):
        return torch.LongTensor([self.alphabet.stoi(c) for c in chars])

    def _input_tensors(self, chars):
        tensor = torch.zeros(len(chars), len(self.alphabet))
        ones = torch.ones(len(chars), len(self.alphabet))
        tensor.scatter_(1, self._charstoi(schars), ones)
        return tensor

    def __getitem__(self, idx):
        with open(char_file) as cfile, open(morph_file) as mfile:
            for index, chars, morphs in enumerate(zip(cfile, mfile)):
                if index == idx:
                    sample = {'chars': _input_tensors(chars), 'morphs': self._morph_tensors(morphs.split(morph_delim))}
        if self.transform:
            sample = self.transform(sample)
        return sample

def load_data(batch_size, char_file, morph_file, tensor_file):
    dataset = Char2MorphDataset(char_file, morph_file, tensor_file)
    validation_split = .1
    test_split = .2
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train = int(np.floor((validation_split + test_split) * dataset_size))
    test = int(np.floor(test_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[train:], indices[test:train], indices[:test]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             sampler=train_sampler)
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             sampler=val_sampler)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             sampler=test_sampler)
    return train_dataloader, val_dataloader, test_dataloader

def train(e, model, optimizer, train_iter, tensor_size, grad_clip):
    model.train()
    total_loss = 0
    pad = chars.alphabet.stoi['<pad>']

    for b, batch in enumerate(train_iter):
        print(batch)
        src, len_src = batch.chars # not sure if this is the right interface for batch
        trg, len_trg = batch.morphs
        src, trg = src.cuda(), trg.cuda()
        optimizer.zero_grad()
        output = model(src, trg)

        # fix
        loss = F.mse_loss(output[1:].view(-1, vocab_size),
                          trg[1:].contiguous().view(-1),
                          ignore_index=pad)
        loss.backward()
        clip_grad_norm(model.parameters(), grad_clip
        optimizer.step()
        total_loss += loss.data[0]

        if b % 100 == 0 and b != 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                  (b, total_loss, math.exp(total_loss)))
            total_loss = 0

def main():
    args = parse_arguments()
    hidden_size = 512
    embed_size = 256
    assert torch.cuda.is_available()

    print("[!] preparing dataset...")
    train_iter, val_iter, test_iter, chars, morphs = load_dataset(args.batch_size,
                                                                  args.txt_file,
                                                                  args.morph_file,
                                                                  args.tensor_file)
    char_size, morph_size = len(chars.dataset.alphabet), len(morphs.vocab)
    print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
          % (len(train_iter), len(train_iter.dataset),
             len(test_iter), len(test_iter.dataset)))
    # print("[DE_vocab]:%d [en_vocab]:%d" % (de_size, en_size))

    print("[!] Instantiating models...")
    encoder = Encoder(char_size, embed_size, hidden_size,
                      n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, morph_size,
                      n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).cuda()
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    best_val_loss = None
    for e in range(1, args.epochs+1):
        train(e, seq2seq, optimizer, train_iter,
              en_size, args.grad_clip, , EN)
        val_loss = evaluate(seq2seq, val_iter, en_size, DE, EN)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
              % (e, val_loss, math.exp(val_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model...")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(seq2seq.state_dict(), './.save/seq2seq_%d.pt' % (e))
            best_val_loss = val_loss
    test_loss = evaluate(seq2seq, test_iter, en_size, DE, EN)
    print("[TEST] loss:%5.2f" % test_loss)
