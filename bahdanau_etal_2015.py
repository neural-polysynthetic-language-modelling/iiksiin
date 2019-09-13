from typing import Tuple

import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, *,
                 mode: str,
                 input_size: int,
                 hidden_size: int,
                 num_hidden_layers: int,
                 use_bias: bool = True,
                 dropout: float = 0.0,
                 bidirectional: bool = True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = nn.RNNBase(mode=mode,
                              input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_hidden_layers,
                              bias=use_bias,
                              batch_first=True,
                              dropout=dropout,
                              bidirectional=bidirectional)

        self.num_directions = 2 if bidirectional else 1

    def forward(self, *,
                batch_size: int,
                max_seq_len: int,
                input_tensor: torch.Tensor,
                previous_hidden_state: torch.Tensor = None) -> torch.Tensor:

        assert batch_size > 0
        assert max_seq_len > 0

        assert input_tensor.shape == torch.Size([batch_size, max_seq_len, self.rnn.input_size])
        rnn_outputs: Tuple[torch.Tensor, torch.Tensor] = self.rnn(input_tensor, previous_hidden_state)

        rnn_final_hidden_layers: torch.Tensor = rnn_outputs[0]
        assert rnn_final_hidden_layers.shape == torch.Size([batch_size,
                                                            max_seq_len,
                                                            self.num_directions * self.rnn.hidden_size])

        return rnn_final_hidden_layers


class Attention(nn.Module):

    def __init__(self, *,
                 attention_hidden_size: int,
                 encoder_hidden_size: int,
                 decoder_hidden_size: int,
                 bidirectional_encoder: bool,
                 attention_activation_function: nn.Module):
        super().__init__()

        self.attention_hidden_size: int = attention_hidden_size
        self.encoder_hidden_size: int = encoder_hidden_size
        self.decoder_hidden_size: int = decoder_hidden_size
        self.num_encoder_directions: int = 2 if bidirectional_encoder else 1
        self.activation_function = attention_activation_function

        self.context_size = self.encoder_hidden_size * self.num_encoder_directions

        self.W = nn.Linear(self.decoder_hidden_size, self.attention_hidden_size)

        self.U = nn.Linear(self.context_size, self.attention_hidden_size)

        self.v = nn.Parameter(torch.rand(self.attention_hidden_size))

    def a(self, *,
          batch_size: int,
          max_seq_len: int,
          previous_decoder_hidden_state: torch.Tensor,
          encoder_final_hidden_layers: torch.Tensor) -> torch.Tensor:

        assert batch_size > 0
        assert max_seq_len > 0

        assert previous_decoder_hidden_state.shape == torch.Size([batch_size, self.decoder_hidden_size])

        assert encoder_final_hidden_layers.shape == torch.Size([batch_size,
                                                                max_seq_len,
                                                                self.context_size])

        tmp_1: torch.Tensor = self.W(previous_decoder_hidden_state)
        assert tmp_1.shape == torch.Size([batch_size, self.attention_hidden_size])

        tmp_2: torch.Tensor = tmp_1.reshape(batch_size, 1, self.attention_hidden_size)
        assert tmp_2.shape == torch.Size([batch_size, 1, self.attention_hidden_size])

        tmp_3: torch.Tensor = self.U(encoder_final_hidden_layers)
        assert tmp_3.shape == torch.Size([batch_size, max_seq_len, self.attention_hidden_size])

        tmp_4: torch.Tensor = torch.add(tmp_2, tmp_3)
        assert tmp_4.shape == torch.Size([batch_size, max_seq_len, self.attention_hidden_size])

        tmp_5: torch.Tensor = self.activation_function(tmp_4)
        assert tmp_5.shape == torch.Size([batch_size, max_seq_len, self.attention_hidden_size])

        energy: torch.Tensor = torch.matmul(self.v, tmp_5)
        assert energy.shape == torch.Size([batch_size, max_seq_len])

        return energy

    def calculate_alpha(self, *,
                        batch_size: int,
                        max_seq_len: int,
                        previous_decoder_hidden_state: torch.Tensor,
                        encoder_final_hidden_layers: torch.Tensor) -> torch.Tensor:

        e: torch.Tensor = self.a(batch_size=batch_size,
                                 max_seq_len=max_seq_len,
                                 previous_decoder_hidden_state=previous_decoder_hidden_state,
                                 encoder_final_hidden_layers=encoder_final_hidden_layers)

        assert e.shape == torch.Size([batch_size, max_seq_len])

        exp_e: torch.Tensor = torch.exp(e)
        assert exp_e.shape == torch.Size([batch_size, max_seq_len])

        exp_e_summation: torch.Tensor = torch.sum(exp_e, dim=1)
        assert exp_e_summation.shape == torch.Size([batch_size])

        denominator: torch.Tensor = exp_e_summation.reshape(batch_size, 1)
        assert denominator.shape == torch.Size([batch_size, 1])

        alpha: torch.Tensor = torch.div(exp_e, denominator)
        assert alpha.shape == torch.Size([batch_size, max_seq_len])

        return alpha

    def forward(self, *,
                batch_size: int,
                max_seq_len: int,
                previous_decoder_hidden_state: torch.Tensor,
                encoder_final_hidden_layers: torch.Tensor) -> torch.Tensor:

        alpha: torch.Tensor = self.calculate_alpha(batch_size=batch_size,
                                                   max_seq_len=max_seq_len,
                                                   previous_decoder_hidden_state=previous_decoder_hidden_state,
                                                   encoder_final_hidden_layers=encoder_final_hidden_layers)

        assert alpha.shape == torch.Size([batch_size, max_seq_len])

        alpha_reshaped: torch.Tensor = alpha.reshape(batch_size, 1, max_seq_len)
        assert alpha_reshaped.shape == torch.Size([batch_size, 1, max_seq_len])

        assert encoder_final_hidden_layers.shape == torch.Size([batch_size,
                                                                max_seq_len,
                                                                self.context_size])

        c: torch.Tensor = torch.bmm(alpha_reshaped, encoder_final_hidden_layers)
        assert c.shape == torch.Size([batch_size, self.context_size])

        return c


class DecoderWithAttention(nn.Module):

    def __init__(self, *,
                 attention: Attention,
                 mode: str,
                 output_size: int,
                 hidden_size: int,
                 num_hidden_layers: int,
                 use_bias: bool = True,
                 dropout: float = 0.0):
        super().__init__()

        self.attention: Attention = attention

        self.output_layer_size = output_size
        self.hidden_layer_size = hidden_size
        self.input_size = self.output_layer_size + attention.context_size

        self.decoder = nn.RNNBase(mode=mode,
                                  input_size=self.input_size,
                                  hidden_size=self.hidden_layer_size,
                                  num_layers=num_hidden_layers,
                                  bias=use_bias,
                                  batch_first=True,
                                  dropout=dropout,
                                  bidirectional=False)

        self.output_layer = nn.Linear(self.hidden_layer_size, self.output_layer_size)

    def forward(self, *,
                batch_size: int,
                encoder_max_seq_len: int,
                encoder_final_hidden_layers: torch.Tensor,
                decoder_previous_output: torch.Tensor,
                decoder_previous_hidden_state: torch.Tensor = None) -> torch.Tensor:

        assert decoder_previous_output.shape == torch.Size([batch_size, self.output_layer_size])

        context: torch.Tensor = self.attention.forward(batch_size=batch_size,
                                                       max_seq_len=encoder_max_seq_len,
                                                       previous_decoder_hidden_state=decoder_previous_hidden_state,
                                                       encoder_final_hidden_layers=encoder_final_hidden_layers)

        assert context.shape == torch.Size([batch_size, self.attention.context_size])

        decoder_input: torch.Tensor = torch.cat(tensors=(decoder_previous_output, context), dim=1)
        assert decoder_input.shape == torch.Size([batch_size, self.input_size])

        rnn_outputs: Tuple[torch.Tensor, torch.Tensor] = self.rnn(decoder_input, decoder_previous_hidden_state)

        rnn_final_hidden_layers: torch.Tensor = rnn_outputs[0]
        assert rnn_final_hidden_layers.shape == torch.Size([batch_size, 1, self.rnn.hidden_size])

        rnn_final_hidden_layers = torch.squeeze(rnn_final_hidden_layers, dim=1)
        assert rnn_final_hidden_layers.shape == torch.Size([batch_size, self.rnn.hidden_size])

        decoder_output: torch.Tensor = self.output_layer(rnn_final_hidden_layers)
        assert decoder_output.shape == torch.Size([batch_size, self.output_layer_size])

        return decoder_output


class Seq2Seq(nn.Module):

    def __init__(self, *, encoder: Encoder, decoder: DecoderWithAttention):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, *,
                batch_size: int,
                max_input_seq_len: int,
                input_tensor: torch.Tensor,
                max_output_seq_len: int,
                output_start_of_sequence_tensor: torch.Tensor) -> torch.Tensor:

        assert batch_size > 0
        assert max_input_seq_len > 0

        assert output_start_of_sequence_tensor.shape == torch.Size([batch_size, self.decoder.output_layer_size])

        assert input_tensor.shape == torch.Size([batch_size, max_input_seq_len, self.encoder.input_size])
        encoder_outputs: torch.Tensor = self.encoder(input_tensor)

        assert encoder_outputs.shape == torch.Size([batch_size,
                                                    max_input_seq_len,
                                                    self.encoder.num_directions * self.encoder.hidden_size])

        decoder_previous_output: torch.Tensor = output_start_of_sequence_tensor
        decoder_output: torch.Tensor = torch.zeros(max_output_seq_len, batch_size, self.decoder.output_layer_size)
        assert decoder_output.shape == torch.Size([max_output_seq_len, batch_size, self.decoder.output_layer_size])

        for t in range(max_output_seq_len):  # type: int

            decoder_output[t] = self.decoder(batch_size=batch_size,
                                             encoder_max_seq_len=max_input_seq_len,
                                             encoder_final_hidden_layers=encoder_outputs,
                                             decoder_previous_output=decoder_previous_output)

            decoder_previous_output = decoder_output[t]

        decoder_output = decoder_output.permute(1, 0, 2)
        assert decoder_output.shape == torch.Size([batch_size, max_output_seq_len, self.decoder.output_layer_size])

        return decoder_output


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # assert input_tensor.shape == torch.Size([batch_size, max_seq_len, self.embedding_layer.num_embeddings])
    # embedding_tensor: torch.Tensor = self.embedding_layer(input_tensor)
    #
    # assert embedding_tensor.shape == torch.Size([batch_size, max_seq_len, self.embedding_layer.embedding_dim])
    # rnn_outputs: Tuple[torch.Tensor, torch.Tensor] = self.rnn(embedding_tensor, previous_hidden_state)

    # self.embedding_layer = nn.Embedding(num_embeddings=input_size,
    #                                    embedding_dim=embedding_size)
    pass
