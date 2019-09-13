from typing import Tuple

import torch           # type: ignore
import torch.nn as nn  # type: ignore


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

        print(self.v.shape)
        print(tmp_5.shape)

        energy: torch.Tensor = torch.matmul(tmp_5, self.v)
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
        assert c.shape == torch.Size([batch_size, 1, self.context_size])
        c = c.squeeze(dim=1)
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
        self.num_hidden_layers = num_hidden_layers
        self.input_size = self.output_layer_size + attention.context_size

        self.rnn = nn.RNNBase(mode=mode,
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
                decoder_previous_hidden_state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

        if not decoder_previous_hidden_state:
            decoder_previous_hidden_state = torch.zeros(self.num_hidden_layers, batch_size, self.rnn.hidden_size)
        assert decoder_previous_hidden_state.shape == torch.Size([self.num_hidden_layers, batch_size, self.rnn.hidden_size])

        assert decoder_previous_output.shape == torch.Size([batch_size, self.output_layer_size])
        decoder_previous_output = decoder_previous_output.unsqueeze(dim=1)
        assert decoder_previous_output.shape == torch.Size([batch_size, 1, self.output_layer_size])

        context: torch.Tensor = self.attention.forward(batch_size=batch_size,
                                                       max_seq_len=encoder_max_seq_len,
                                                       previous_decoder_hidden_state=decoder_previous_hidden_state,
                                                       encoder_final_hidden_layers=encoder_final_hidden_layers)

        assert context.shape == torch.Size([batch_size, self.attention.context_size])
        context = context.unsqueeze(dim=1)
        assert context.shape == torch.Size([batch_size, 1, self.attention.context_size])

        decoder_input: torch.Tensor = torch.cat(tensors=(decoder_previous_output, context), dim=2)
        assert decoder_input.shape == torch.Size([batch_size, 1, self.input_size])

        assert decoder_previous_hidden_state.shape == torch.Size([self.num_hidden_layers,
                                                                  batch_size,
                                                                  self.rnn.hidden_size])
        rnn_outputs: Tuple[torch.Tensor, torch.Tensor] = self.rnn(decoder_input, decoder_previous_hidden_state)

        rnn_final_hidden_layers: torch.Tensor = rnn_outputs[0]
        assert rnn_final_hidden_layers.shape == torch.Size([batch_size, 1, self.rnn.hidden_size])

        rnn_final_hidden_layers = torch.squeeze(rnn_final_hidden_layers, dim=1)
        assert rnn_final_hidden_layers.shape == torch.Size([batch_size, self.rnn.hidden_size])

        decoder_output: torch.Tensor = self.output_layer(rnn_final_hidden_layers)
        assert decoder_output.shape == torch.Size([batch_size, self.output_layer_size])

        return decoder_output, rnn_final_hidden_layers


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
        encoder_outputs: torch.Tensor = self.encoder(batch_size=batch_size,
                                                     max_seq_len=max_input_seq_len,
                                                     input_tensor=input_tensor)

        assert encoder_outputs.shape == torch.Size([batch_size,
                                                    max_input_seq_len,
                                                    self.encoder.num_directions * self.encoder.hidden_size])

        decoder_previous_output: torch.Tensor = output_start_of_sequence_tensor
        decoder_output: torch.Tensor = torch.zeros(max_output_seq_len, batch_size, self.decoder.output_layer_size)
        assert decoder_output.shape == torch.Size([max_output_seq_len, batch_size, self.decoder.output_layer_size])

        for t in range(max_output_seq_len):  # type: int

            result: Tuple[torch.Tensor, torch.Tensor] = self.decoder(batch_size=batch_size,
                                                                     encoder_max_seq_len=max_input_seq_len,
                                                                     encoder_final_hidden_layers=encoder_outputs,
                                                                     decoder_previous_output=decoder_previous_output)

            assert result[0].shape == torch.Size([batch_size, self.decoder.output_layer_size])
            assert result[1].shape == torch.Size([batch_size, self.decoder.hidden_size])

            decoder_output[t] = result[0]
            decoder_previous_output = result[1]

        decoder_output = decoder_output.permute(1, 0, 2)
        assert decoder_output.shape == torch.Size([batch_size, max_output_seq_len, self.decoder.output_layer_size])

        return decoder_output


def seq2seq():

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder: Encoder = Encoder(mode="RNN_RELU", input_size=80, hidden_size=128, num_hidden_layers=3)

    attention: Attention = Attention(attention_hidden_size=64,
                                     encoder_hidden_size=encoder.hidden_size,
                                     decoder_hidden_size=150,
                                     bidirectional_encoder=(encoder.num_directions==2),
                                     attention_activation_function=nn.ReLU())

    decoder: DecoderWithAttention = DecoderWithAttention(attention=attention,
                                                         mode="RNN_RELU",
                                                         output_size=80,
                                                         hidden_size=attention.decoder_hidden_size,
                                                         num_hidden_layers=4)

    encoder_decoder: Seq2Seq = Seq2Seq(encoder=encoder, decoder=decoder)

    batch_size: int = 17
    input_seq_len: int = 1
    input_tensor: torch.Tensor = torch.zeros(batch_size, input_seq_len, encoder.input_size)
    input_tensor[0][0][35] = 1
    #input_tensor[0][1][27] = 1

    start_of_output_sequence: torch.Tensor = torch.zeros(batch_size, decoder.output_layer_size)

    # result: torch.Tensor = encoder(batch_size=batch_size,
    #                               max_seq_len=input_seq_len,
    #                               input_tensor=input_tensor)

    result: torch.Tensor = encoder_decoder(batch_size=batch_size,
                                           max_input_seq_len=1,
                                           max_output_seq_len=1,
                                           input_tensor=input_tensor,
                                           output_start_of_sequence_tensor=start_of_output_sequence)

    print(result.shape)


if __name__ == "__main__":

    seq2seq()