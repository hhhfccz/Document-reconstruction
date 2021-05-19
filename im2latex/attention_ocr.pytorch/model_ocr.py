# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2021/3/18
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm
from torch.nn.parameter import Parameter

from utils import weights_init


def ConvBlock(in_channels, out_channels, relu=True, bn=False, pool=False, win=2):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
    if relu:
        layers.append(nn.CELU(inplace=True))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if pool:
        layers.append(nn.MaxPool2d(win, 2))
    return nn.Sequential(*layers)


class BidirectionalLSTM(nn.Module):
    __slots__ = ["rnn", "embedding"]

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)
        del t_rec, h
        # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.CELU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.CELU(inplace=True)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout,
                                 self.conv2, self.chomp2, self.relu2, self.dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class EncoderTCN(object):
    """docstring for EncoderTCN"""

    def __init__(self, arg):
        super(EncoderTCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=0.3)


class EncoderCRNN(nn.Module):
    """resnet and lstm, crnn"""
    __slots__ = ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "res1", "res2", "res3", "rnn"]

    def __init__(self, imgH, nc, nh):
        super(EncoderCRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.conv1 = ConvBlock(nc, 64)
        self.conv2 = ConvBlock(64, 128, pool=True, win=1)
        self.res1 = nn.Sequential(ConvBlock(128, 128, relu=False), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 256, pool=True)
        self.res2 = nn.Sequential(ConvBlock(256, 256, relu=False), ConvBlock(256, 256))

        self.conv5 = ConvBlock(256, 512, pool=True)
        self.conv6 = ConvBlock(512, 512, pool=True, win=1)
        self.res3 = nn.Sequential(ConvBlock(512, 512, bn=True, relu=False), ConvBlock(512, 512))

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nh),
        )

    def forward(self, xb):
        # conv features
        # print(xb)
        out = self.conv1(xb)
        del xb
        # print(out.shape)
        out = self.conv2(out)
        # print(out.shape)
        out = self.res1(out) + out
        # print(out.shape)
        out = self.conv3(out)
        # print(out.shape)
        out = self.conv4(out)
        # print(out.shape)
        out = self.res2(out) + out
        # print(out.shape)
        out = self.conv5(out)
        # print(out.shape)
        out = self.conv6(out)
        # print(out.shape)
        out = self.res3(out) + out
        # print(out.shape)

        b, c, h, w = out.size()
        assert h == 1, "the height of conv must be 1, but get " + str(h)
        conv = out.squeeze(2)
        del out
        conv = conv.permute(2, 0, 1)
        # print(conv.shape) # this should be [w, b, c]

        # rnn features calculate
        encoder_outputs = self.rnn(conv)
        # print(encoder_outputs.shape)

        return encoder_outputs


class DecoderAttention(nn.Module):
    __slots__ = ["hidden_size", "dropout", "embedding", "attn_combine", "gru", "out", "vat"]

    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(DecoderAttention, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)

        self.vat = nn.Linear(hidden_size, 1)

    def forward(self, input_embed, hidden, encoder_outputs):
        # print(input_embed.shape)
        embedded = self.embedding(input_embed.long())
        embedded = self.dropout(embedded)
        # print(embedded.shape, hidden.shape)

        batch_size = encoder_outputs.shape[1]
        # print(batch_size)
        # print(hidden.shape, encoder_outputs.shape)
        alpha = hidden + encoder_outputs
        # print(alpha.shape)
        alpha = alpha.view(-1, alpha.shape[-1])
        # print(alpha.shape)
        attn_weights = self.vat(torch.tanh(alpha))
        del alpha
        # print(attn_weights.shape)
        attn_weights = attn_weights.view(-1, 1, batch_size).permute((2, 1, 0))
        del batch_size
        # print(attn_weights.shape)
        attn_weights = F.softmax(attn_weights, dim=2)
        # print(attn_weights.shape)

        attn_applied = torch.matmul(attn_weights, encoder_outputs.permute((1, 0, 2)))
        # print(embedded.shape, attn_applied.view(-1, self.hidden_size).shape)
        output = torch.cat((embedded.view(-1, self.hidden_size), attn_applied.squeeze(1).view(-1, self.hidden_size)), 1)
        del embedded
        # print(output.shape)

        output = self.attn_combine(output).unsqueeze(0)
        # print(output.shape)

        output = F.relu(output)
        # print(output.shape)
        # print(hidden.shape)
        output, hidden = self.gru(output, hidden)
        # print(output.shape)

        output = F.log_softmax(self.out(output[0]), dim=1)
        # print(output.shape)
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        # don't use self.batch_size, see train.val()
        result = torch.zeros(1, batch_size, self.hidden_size)
        return result


class Decoder(nn.Module):
    """
        decoder from image features
    """
    __slots__ = ["hidden_size", "decoder"]

    def __init__(self, nh=256, nclass=13, dropout_p=0.1, batch_size=4):
        super(Decoder, self).__init__()
        self.hidden_size = nh
        self.decoder = DecoderAttention(nh, nclass, dropout_p)

    def forward(self, input, hidden, encoder_outputs):
        return self.decoder(input, hidden, encoder_outputs)

    def initHidden(self, batch_size):
        result = torch.zeros(1, batch_size, self.hidden_size)
        return result


if __name__ == '__main__':
    encoder = EncoderCRNN(imgH=32, nc=10, nh=256)
    decoder = Decoder(nh=256, nclass=100, dropout_p=0.1)

    encoder.apply(weights_init)
    decoder.apply(weights_init)
    # For prediction of an indefinite long sequence
    print(encoder)
    print(decoder)
