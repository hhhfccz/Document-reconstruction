import time
import numpy as np

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import dataloader

from utils import *
from dataset import *
from model_ocr import *

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=280, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for Critic, default=1e-3')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--encoder', type=str, default='', help="path to encoder (to continue training)")
parser.add_argument('--decoder', type=str, default='', help='path to decoder (to continue training)')
parser.add_argument('--experiment',
                    default='/home/hhhfccz/im2latex/attention_ocr.pytorch/expr/attention_ocr',
                    help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=10, help='Interval to be displayed')
parser.add_argument('--valInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=2, help='Interval to be displayed')
parser.add_argument('--adam', default=True, action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', default=True, action='store_true',
                    help='whether to sample the dataset with random sampler')
parser.add_argument('--max_width', type=int, default=71, help='the width of the featuremap out from cnn')
opt = parser.parse_args()
print("------init------")

# if opt.experiment is None:
#     opt.experiment = 'expr'
# os.system('mkdir -p {0}'.format(opt.experiment))

opt.manualSeed = 118
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.enabled = True
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = Im2latex_Dataset(split="train", transform=None)
assert train_dataset
if not opt.random_sample:
    sampler = RandomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None

train_loader = dataloader.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=False, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio)
)

test_dataset = Im2latex_Dataset(split="validate", transform=ResizeNormalize(opt.imgH, opt.imgW))

nclass = len(get_chars("train")) + 3
nc = 1

encoder = encoderV1(opt.imgH, nc, opt.nh)
decoder = decoderV2(opt.nh, nclass, dropout_p=0.1, batch_size=opt.batchSize)
# For prediction of an indefinite long sequence
encoder.apply(weights_init)
decoder.apply(weights_init)
# continue training or use the pretrained model to initial the parameters of the encoder and decoder
if opt.encoder:
    print('loading pretrained encoder model from %s' % opt.encoder)
    encoder.load_state_dict(torch.load(opt.encoder))
if opt.decoder:
    print('loading pretrained encoder model from %s' % opt.decoder)
    decoder.load_state_dict(torch.load(opt.decoder))
print(encoder)
print(decoder)

# loss averager
loss_avg = Averager()

# setup optimizer
if opt.adam:
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=opt.lr,
                                   betas=(opt.beta1, 0.999)
                                   )
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.lr,
                                   betas=(opt.beta1, 0.999)
                                   )
elif opt.adadelta:
    optimizer = optim.Adadelta(encoder.parameters(), lr=opt.lr)
else:
    encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=opt.lr)
    decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=opt.lr)


def get_decoder_input(texts, i):
    decoder_input = []
    for text in texts:
        decoder_input.append(text[i])
    decoder_input = torch.from_numpy(np.array(decoder_input))
    return decoder_input


def trainBatch(train_iter, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion=torch.nn.CrossEntropyLoss()):
    if opt.cuda:
        encoder.cuda()
        decoder.cuda()
        criterion = criterion.cuda()

    data = train_iter.__next__()
    img, texts = data
    if opt.cuda:
        img = img.cuda()
        decoder_input = get_decoder_input(texts, 0).cuda()
        decoder_hidden = decoder.initHidden().cuda()
        encoder_outputs = encoder(img).cuda()
    else:
        decoder_input = get_decoder_input(texts, 0)
        decoder_hidden = decoder.initHidden()
        encoder_outputs = encoder(img)
    # print(decoder_hidden.shape)

    loss = 0.0
    for i in range(1, len(texts[0])):
        decoder_outputs, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

        if opt.cuda:
            decoder_inputs = get_decoder_input(texts, i).cuda()
        else:
            decoder_inputs = get_decoder_input(texts, i)
        # print(decoder_outputs.shape, decoder_inputs.shape)

        loss += criterion(decoder_outputs, decoder_inputs)

    encoder.zero_grad()
    decoder.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss


if __name__ == '__main__':
    for epoch in range(opt.niter):
        train_iter = iter(train_loader)
        i = 0
        t0 = time.time()
        while i < len(train_loader) - 1:
            for e, d in zip(encoder.parameters(), decoder.parameters()):
                e.requires_grad = True
                d.requires_grad = True
            encoder.train()
            decoder.train()
            # print("Batch: " + str(i))
            cost = trainBatch(train_iter, encoder, decoder, encoder_optimizer, decoder_optimizer)
            loss_avg.add(cost)
            i += 1

            if i % opt.displayInterval == 0:
                print('[%d/%d][%d/%d] CrossEntropyLoss: %f' %
                      (epoch, opt.niter, i, len(train_loader), loss_avg.val()), end=' '
                      )
                loss_avg.reset()
                t1 = time.time()
                print('Time: %s' % str(t1 - t0))
                t0 = time.time()

        # do checkpointing
        if epoch % opt.saveInterval == 0:
            # val(opt, encoder, decoder, 1, dataset=test_dataset)
            torch.save(
                encoder.state_dict(), '{0}/encoder_epoch_{1}.pth.tar'.format(opt.experiment, epoch)
            )
            torch.save(
                decoder.state_dict(), '{0}/decoder_epoch_{1}.pth.tar'.format(opt.experiment, epoch)
            )
        torch.cuda.empty_cache()
