import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from utils import weights_init


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)
        # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class encoderV1(nn.Module):
    '''
        CNN+BiLSTM做特征提取
    '''

    def __init__(self, imgH, nc, nh):
        super(encoderV1, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.conv1 = ConvBlock(nc, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 256, pool=True)
        self.res2 = nn.Sequential(ConvBlock(256, 256), ConvBlock(256, 256))

        self.conv5 = ConvBlock(256, 512, pool=True)
        self.conv6 = ConvBlock(512, 512, pool=True)
        self.res3 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        # self.cnn = nn.Sequential(
        #               nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), 
        #               # 64x16x50
        #               nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), 
        #               # 128x8x25
        #               nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), 
        #               # 256x8x25
        #               nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)), 
        #               # 256x4x25
        #               nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), 
        #               # 512x4x25
        #               nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)), 
        #               # 512x2x25
        #               nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),
        #               # 512x1x25
        #               )
        # previous cnn network

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nh)
        )

    def forward(self, xb):
        # conv features
        out = self.conv1(xb)
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
        conv = conv.permute(2, 0, 1)
        # [w, b, c]

        # rnn features calculate
        encoder_outputs = self.rnn(conv)

        return encoder_outputs


class DecoderRNN(nn.Module):
    """
        采用RNN进行解码
    """

    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))

        return result


class Attentiondecoder(nn.Module):
    """
        采用attention注意力机制，进行解码
    """

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=71):
        super(Attentiondecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # calculate the attention weight and weight * encoder_output feature
        embedded = self.embedding(input)
        # 前一次的输出进行词嵌入
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1
        )
        # 上一次的输出和隐藏状态求出权重, 主要使用一个linear layer从512维到71维，所以只能处理固定宽度的序列
        attn_applied = torch.matmul(attn_weights.unsqueeze(1),
                                    encoder_outputs.permute((1, 0, 2))
                                    )
        # 矩阵乘法，bmm（8×1×56，8×56×256）=8×1×256

        output = torch.cat((embedded, attn_applied.squeeze(1)), 1)
        # 上一次的输出和attention feature做一个融合，再加一个linear layer
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        # just as sequence to sequence decoder

        output = F.log_softmax(self.out(output[0]), dim=1)
        # use log_softmax for nllloss
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))

        return result


class AttentiondecoderV2(nn.Module):
    """
        采用seq to seq模型，修改注意力权重的计算方式
    """

    def __init__(self, hidden_size, output_size, dropout_p=0.1, batch_size=4):
        super(AttentiondecoderV2, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.batch_size = batch_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # test
        self.vat = nn.Linear(hidden_size, 1)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        # 前一次的输出进行词嵌入
        embedded = self.dropout(embedded)

        # test
        batch_size = encoder_outputs.shape[1]
        alpha = hidden + encoder_outputs
        # 特征融合采用+/concat其实都可以
        alpha = alpha.view(-1, alpha.shape[-1])
        attn_weights = self.vat(torch.tanh(alpha))
        # 将encoder_output:batch*seq*features,将features的维度降为1
        attn_weights = attn_weights.view(-1, 1, self.batch_size).permute((2, 1, 0))
        attn_weights = F.softmax(attn_weights, dim=2)

        # attn_weights = F.softmax(
        #     self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1
        # )        
        # 上一次的输出和隐藏状态求出权重

        attn_applied = torch.matmul(attn_weights,
                                    encoder_outputs.permute((1, 0, 2))
                                    )
        # 矩阵乘法，bmm（8×1×56，8×56×256）=8×1×256
        # print(embedded.shape, attn_applied.shape)
        output = torch.cat((embedded.view(1, 1, 256), attn_applied), 0)
        # print(output.shape)

        # 上一次的输出和attention feature，做一个线性+GRU
        output = self.attn_combine(output).unsqueeze(0)
        # print(output.shape)

        output = F.relu(output)
        # print(output.shape)
        # print(hidden.shape)
        output, hidden = self.gru(output.view(-1, 1, 256), hidden.view(-1, 1, 256))
        # print(output.shape)

        output = F.log_softmax(self.out(output[0]), dim=1)
        # print(output.shape)
        # 最后输出一个概率
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))

        return result


class decoder(nn.Module):
    '''
        decoder from image features
    '''

    def __init__(self, nh=256, nclass=13, dropout_p=0.1, max_length=71):
        super(decoder, self).__init__()
        self.hidden_size = nh
        self.decoder = Attentiondecoder(nh, nclass, dropout_p, max_length)

    def forward(self, input, hidden, encoder_outputs):
        return self.decoder(input, hidden, encoder_outputs)

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result


class decoderV2(nn.Module):
    '''
        decoder from image features
    '''

    def __init__(self, nh=256, nclass=13, dropout_p=0.1, batch_size=4):
        super(decoderV2, self).__init__()
        self.hidden_size = nh
        self.decoder = AttentiondecoderV2(nh, nclass, dropout_p, batch_size)

    def forward(self, input, hidden, encoder_outputs):
        return self.decoder(input, hidden, encoder_outputs)

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result


if __name__ == '__main__':
    encoder = CNN(imgH=32, nc=10, nh=256)
    # decoder = decoder(nh=256, nclass=100, dropout_p=0.1, max_length=max_width) 
    # max_length:w/4,为encoder特征提取之后宽度方向上的序列长度
    decoder = decoderV2(nh=256, nclass=100, dropout_p=0.1)
    encoder.apply(weights_init)
    decoder.apply(weights_init)
    # For prediction of an indefinite long sequence
    print(encoder)
    print(decoder)
