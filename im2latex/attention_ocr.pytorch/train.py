import os
import sys
import time
import json
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import dataloader

from utils import *
from dataset import *
from model_ocr import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class data_prefetcher():

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        img = self.img
        text = self.text
        if img is not None:
            img.record_stream(torch.cuda.current_stream())
            text.record_stream(torch.cuda.current_stream())
        self.preload()
        return img, text

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return

        with torch.cuda.stream(self.stream):
            self.img, self.texts = self.next_input
            self.text = torch.zeros(len(self.texts), len(self.texts[0]), dtype=torch.long)
            j = 0
            for txt in self.texts:
                self.text[j] = txt
                j += 1

            self.img = self.img.cuda(non_blocking=True)
            self.text = self.text.cuda(non_blocking=True)


def enlarge_decoder_output(x, max_enlargement=5):
    """
    use the derivative in tanh
    """
    x /= 1e3
    return (max_enlargement - 1) * (1 - pow((1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x)), 2)) + 1


def get_decoder_input(texts, i):
    decoder_input = []
    for text in texts:
        decoder_input.append(text[i])
    decoder_input = torch.from_numpy(np.array(decoder_input))
    return decoder_input


def trainBatch(opt, img, text, encoder, decoder, encoder_optimizer, decoder_optimizer, times, epoch, criterion=torch.nn.NLLLoss()):
    if not opt.cuda:
        decoder_input = get_decoder_input(text, 0)
        decoder_hidden = decoder.initHidden(opt.batchSize)
        encoder_output = encoder(img)
    else:
        encoder.cuda()
        decoder.cuda()
        criterion = criterion.cuda()
        decoder_input = text[:, 0].cuda()
        decoder_hidden.cuda()
        encoder_output = encoder(img).cuda()

    loss = 0.0
    for i in range(1, len(text[0])):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output)

        if opt.cuda:
            decoder_input = text[:, i].cuda()
        else:
            decoder_input = get_decoder_input(text, i)

        if not epoch:
            decoder_outputs = decoder_output * enlarge_decoder_output(times)
            # !!! don't use *= , will make torch confused !!!
            loss += criterion(decoder_outputs, decoder_input)
        else:
            loss += criterion(decoder_output, decoder_input)

    encoder.zero_grad()
    decoder.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss


def valid(opt, encoder, decoder, data_loader, max_iter=1, criterion=torch.nn.NLLLoss(), batch_size=1, get_loss=False):
    num_correct = 0
    num_total = 0
    test_iter = iter(data_loader)
    if not get_loss:
        max_iter = min(max_iter, len(data_loader))
        for i in range(np.random.randint(0, len(data_loader)-1)):
            data = test_iter.__next__()
    else:
        max_iter = len(data_loader)

    if opt.cuda:
        encoder.cuda()
        decoder.cuda()
        criterion = criterion.cuda()

    for e, d in zip(encoder.parameters(), decoder.parameters()):
        e.requires_grad = False
        d.requires_grad = False

    encoder.eval()
    decoder.eval()

    loss_val_avg = Averager()

    for i in range(1, max_iter+1):
        data = test_iter.__next__()
        img, text = data

        if not opt.cuda:
            decoder_input = get_decoder_input(text, 0)
            decoder_hidden = decoder.initHidden(batch_size)
            encoder_output = encoder(img)
            decoder_attentions = torch.zeros(len(text[0]), 8)
            # TODO, 8 is the width of the featuremap out from cnn, it may be the 8 * batch_size
        else:
            img.cuda()
            decoder_input.cuda()
            decoder_hidden.cuda()
            encoder_output = encoder(img).cuda()
            decoder_attentions.cuda()


        loss_val = 0
        decoded_label = []

        for l in range(1, len(text[0])):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output)

            loss_val += criterion(decoder_output, decoder_input)
            loss_val_avg.add(loss_val)

            decoder_attentions[l-1] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi.squeeze(1)
            decoded_label.append(decoder_input)
            # print(decoder_input)

            if decoder_input == 1:
                break

        pred_text = torch.from_numpy(np.array(decoded_label))
        for pred, target in zip(pred_text, text[0]):
            num_total += 1
            if target == torch.tensor(1):
                break
            if pred == target:
                num_correct += 1

    # print(num_correct, num_total)
    if not get_loss:
        return num_correct / float(num_total)
    else:
        return num_correct / float(num_total), loss_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers, you had better put it 4 times of your gpu')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=280, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
    parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=2e-3, help='select the max learning rate, default=2e-3')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
    parser.add_argument('--preload', action='store_true', default=False, help='enables preload')
    # parser.add_argument('--encoder', type=str, default='', help="path to encoder (to continue training)")
    # parser.add_argument('--decoder', type=str, default='', help='path to decoder (to continue training)')
    parser.add_argument('--experiment',
                        default='/home/hhhfccz/im2latex/attention_ocr.pytorch/expr/attention_ocr/',
                        help='Where to store samples and models')
    parser.add_argument('--displayInterval', type=int, default=10, help='Interval to be displayed')
    parser.add_argument('--saveInterval', type=int, default=2, help='Interval to be displayed')
    parser.add_argument('--adam', default=True, action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
    parser.add_argument('--random_sample', default=True, action='store_true',
                        help='whether to sample the dataset with random sampler')
    opt = parser.parse_args()

    print("------init------")
    opt.manualSeed = 118
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    nclass = len(get_chars("train")) + 3
    nc = 1

    if opt.cuda:
        cudnn.benchmark = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if not CUDA_AVAILABLE and opt.cuda:
        assert torch.cuda.is_available() == True, "ERROR: You don't have a CUDA device"
    if not opt.cuda and opt.preload:
        assert (opt.preload == True and not CUDA_AVAILABLE), "ERROR: You don't have a CUDA device"
    if opt.cuda and not opt.preload:
        print("WARNING: You choosed the CUDA, so you could run with --preload")

    # train dataset init
    train_dataset = Im2latex_Dataset(split="train", transform=None)
    print("the correspondence between tex chars and numbers: \n", json.dumps(train_dataset.chars2num, indent=4, sort_keys=True))

    assert train_dataset
    if not opt.random_sample:
        sampler = RandomSequentialSampler(train_dataset, opt.batchSize)
    else:
        sampler = None

    train_loader = dataloader.DataLoader(
        train_dataset, batch_size=opt.batchSize,
        shuffle=True, sampler=sampler,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio),
        pin_memory=True
    )
    length = len(train_loader)

    # test dataset init
    test_dataset = Im2latex_Dataset(split="test", transform=ResizeNormalize(opt.imgH, opt.imgW))

    test_loader = dataloader.DataLoader(
        test_dataset, batch_size=1,
        shuffle=True, num_workers=int(opt.workers),
        pin_memory=True
        )

    # network init
    encoder = encoderV1(opt.imgH, nc, opt.nh)
    decoder = decoderV2(opt.nh, nclass, dropout_p=0.2, batch_size=opt.batchSize)
    # For prediction of an indefinite long sequence
    # encoder.apply(weights_init)
    # decoder.apply(weights_init)
    # continue training or use the pretrained model to initial the parameters of the encoder and decoder
    if opt.encoder:
        print('loading pretrained encoder model from %s' % opt.encoder)
        encoder.load_state_dict(torch.load(opt.encoder))
    if opt.decoder:
        print('loading pretrained encoder model from %s' % opt.decoder)
        decoder.load_state_dict(torch.load(opt.decoder))
    print(encoder)
    print(decoder)

    # choose loss
    criterion = torch.nn.CrossEntropyLoss()

    # loss averager
    loss_avg = Averager()

    # setup optimizer
    if opt.adam:
        encoder_optimizer = optim.Adadelta(encoder.parameters(), lr=opt.lr)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), amsgrad=True)
    else:
        encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=opt.lr)
        decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=opt.lr)

    # train
    print("-----train-----")
    t0 = time.time()
    for epoch in range(opt.niter):
        i = 0

        if opt.preload:
            prefetcher = data_prefetcher(train_loader)
            img, text = prefetcher.__next__()
        else:
            train_iter = iter(train_loader)
            data = train_iter.__next__()
            img, text = data

        while i < length - 1:
            for e, d in zip(encoder.parameters(), decoder.parameters()):
                e.requires_grad = True
                d.requires_grad = True
            encoder.train()
            decoder.train()

            cost = trainBatch(opt, img, text, encoder, decoder, encoder_optimizer, decoder_optimizer, i, epoch, criterion=criterion)
            loss_avg.add(cost)
            i += 1

            if opt.preload:
                img, text = prefetcher.__next__()
            else:
                data = train_iter.__next__()
                img, text = data

            if i % opt.displayInterval == 0:
                # do val
                acc = valid(opt, encoder, decoder, data_loader=test_loader, criterion=criterion)

                print('[%d/%d][%d/%d] Loss: %f Acc: %f' % (epoch, opt.niter, i, length, loss_avg.val(), acc), end=' ')

                loss_avg.reset()

                t1 = time.time()
                print('Time: %s' % str(t1 - t0))
                t0 = time.time()

        # do saving
        acc_val, loss_val = valid(opt, encoder, decoder, data_loader=test_loader, criterion=criterion, get_loss=True)
        print('Time: ', time.strftime('%Y-%m-%d %H:%M:%S'), 'Acc: %f, Loss: %f' % (acc_val, loss_val), 'it\'s time to save one model')
        if epoch % opt.saveInterval == 0:
            torch.save(
                encoder.state_dict(), '{0}/encoder_epoch_{1}.pth.tar'.format(opt.experiment, epoch)
            )
            torch.save(
                decoder.state_dict(), '{0}/decoder_epoch_{1}.pth.tar'.format(opt.experiment, epoch)
            )
        print('Model saved')

    if opt.cuda:
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
