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


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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


def trainBatch(opt, img, text, encoder, decoder, encoder_optimizer, decoder_optimizer, times, epoch,
               criterion=torch.nn.NLLLoss()):
    if not opt.cuda:
        decoder_input = get_decoder_input(text, 0)
        decoder_hidden = decoder.initHidden(opt.batchSize)
        encoder_output = encoder(img)
    else:
        encoder.cuda()
        decoder.cuda()
        criterion = criterion.cuda()
        decoder_input = text[:, 0]
        decoder_hidden = decoder.initHidden(opt.batchSize).cuda()
        encoder_output = encoder(img).cuda()

    loss = 0.0
    for i in range(1, len(text[0])):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output)

        if opt.cuda:
            decoder_input = text[:, i]
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


def valid(opt, encoder, decoder, data_loader, max_iter=10, loss_val_avg=Averager(), criterion=torch.nn.NLLLoss(),
          batch_size=16, get_loss=True, test_all=False):
    num_correct = 0
    num_total = 0
    test_iter = iter(data_loader)
    if not test_all:
        max_iter = min(max_iter, len(data_loader))
    else:
        max_iter = len(data_loader)

    if opt.cuda and not test_all:
        encoder.cuda()
        decoder.cuda()
        criterion = criterion.cuda()

    for e, d in zip(encoder.parameters(), decoder.parameters()):
        e.requires_grad = False
        d.requires_grad = False

    encoder.eval()
    decoder.eval()

    loss_val = 0.0
    for i in range(1, max_iter + 1):
        data = test_iter.__next__()
        img, text = data
        length = len(text[0])
        decoded_labels = torch.zeros((length, batch_size))

        if not opt.cuda:
            decoder_input = get_decoder_input(text, 0)
            decoder_hidden = decoder.initHidden(batch_size)
            encoder_output = encoder(img)
        else:
            img = img.cuda()
            decoder_input = get_decoder_input(text, 0).cuda()
            decoder_hidden = decoder.initHidden(batch_size).cuda()
            encoder_output = encoder(img).cuda()
            decoded_labels = decoded_labels.cuda()

        for l in range(1, length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output)

            loss_val += criterion(decoder_output, decoder_input)
            loss_val_avg.add(loss_val)

            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi.squeeze(1)
            decoded_labels[l] = decoder_input

        # TODO: so slow
        for k in range(length):
            for pred, target in zip(decoded_labels[k].cpu(), text[:, k]):
                if int(target) == 2 and int(pred) == 2:
                    continue
                elif int(target) != 1:
                    num_total += 1
                    if int(pred) == int(target):
                        num_correct += 1
                else:
                    break

    if not get_loss:
        return num_correct / float(num_total)
    else:
        return num_correct / float(num_total), loss_val_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers, you had better put it '
                                                               '4 times of your gpu')
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size, default=32')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network, default=32')
    parser.add_argument('--imgW', type=int, default=280, help='the width of the input image to network, default=280')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state, default=256')
    parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for, default=10')
    parser.add_argument('--lr', type=float, default=5e-3, help='select the max learning rate, default=5e-3')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--experiment',
                        default='/home/hhhfccz/im2latex/attention_ocr.pytorch/expr/attention_ocr/',
                        help='where to store samples and models')
    parser.add_argument('--displayInterval', type=int, default=10, help='Interval to be displayed')
    parser.add_argument('--saveInterval', type=int, default=2, help='Interval to be displayed')
    parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
    parser.add_argument('--random_sample', default=True, action='store_true',
                        help='whether to sample the dataset with random sampler')
    parser.add_argument('--adam', default=True, action='store_true', help='whether to use adam')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
    parser.add_argument('--preload', action='store_true', default=False, help='enables preload')
    parser.add_argument('--continue_train', action='store_true', default=False, help="whether to continue training")
    opt = parser.parse_args()

    print("------init------")
    manual_seed = 118
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    nclass = len(get_chars("train")) + 3
    nc = 1

    if opt.cuda:
        cudnn.enabled = True
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
    print("the correspondence between tex chars and numbers: \n",
          json.dumps(train_dataset.chars2num, indent=4, sort_keys=True))

    assert train_dataset
    if opt.random_sample:
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
    # TODO: you can modified it 
    test_batch_size = 16
    test_loader = dataloader.DataLoader(
        test_dataset, batch_size=test_batch_size,
        shuffle=True, num_workers=int(opt.workers),
        pin_memory=True
    )

    # test dataset init
    valid_dataset = Im2latex_Dataset(split="validate", transform=ResizeNormalize(opt.imgH, opt.imgW))
    # TODO: you can modified it 
    valid_batch_size = 16
    valid_loader = dataloader.DataLoader(
        valid_dataset, batch_size=valid_batch_size,
        shuffle=False, num_workers=int(opt.workers),
        pin_memory=True
    )

    # network init
    encoder = encoderV1(opt.imgH, nc, opt.nh)
    decoder = decoderV2(opt.nh, nclass, dropout_p=0.3, batch_size=opt.batchSize)
    # setup optimizer
    if opt.adam:
        encoder_optimizer = optim.Adadelta(encoder.parameters(), lr=opt.lr, eps=1e-7, weight_decay=0.01)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.lr, eps=1e-7, betas=(opt.beta1, 0.999),
                                       weight_decay=0.01, amsgrad=True)
    else:
        encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=opt.lr)
        decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=opt.lr)
    if opt.continue_train:
        # continue training or use the pretrained model to initial the parameters of the encoder and decoder
        encoder_path = input('please enter your pretrained encoder_model path: ')
        print('loading pretrained encoder model from %s' % encoder_path)
        checkpoint_encoder = torch.load(encoder_path)
        encoder.load_state_dict(checkpoint_encoder['model'])
        encoder_optimizer.load_state_dict(checkpoint_encoder['optimizer'])

        decoder_path = input('please enter your pretrained decoder_model path: ')
        print('loading pretrained decoder model from %s' % decoder_path)
        checkpoint_decoder = torch.load(decoder_path)
        decoder.load_state_dict(checkpoint_decoder['model'])
        decoder_optimizer.load_state_dict(checkpoint_decoder['optimizer'])
    else:
        # For prediction of an indefinite long sequence
        encoder.apply(weights_init)
        decoder.apply(weights_init)

    print(encoder)
    print(decoder)

    # choose loss
    criterion = torch.nn.CrossEntropyLoss()

    # loss averager
    loss_avg = Averager()

    # train
    print("-----train-----")
    t0 = time.time()
    for epoch in range(opt.niter):
        i = 0

        if opt.preload:
            prefetcher = DataPrefetcher(train_loader)
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

            loss = trainBatch(opt, img, text, encoder, decoder, encoder_optimizer, decoder_optimizer, i, epoch,
                              criterion=criterion)
            loss_avg.add(loss)
            i += 1

            if opt.preload:
                img, text = prefetcher.__next__()
            else:
                data = train_iter.__next__()
                img, text = data

            if i % opt.displayInterval == 0:
                # do val
                acc, loss_test_avg = valid(opt, encoder, decoder, batch_size=test_batch_size, data_loader=test_loader,
                                           criterion=criterion)
                print('[%d/%d][%d/%d] TrainLoss: %f TestLoss: %f Acc: %f' % (
                    epoch, opt.niter, i, length, loss_avg.val(), loss_test_avg.val(), acc), end=' ')
                t1 = time.time()
                print('Time: %s' % str(t1 - t0))
                t0 = time.time()
                loss_avg.reset()
                loss_test_avg.reset()

        print('-----Model saved-----')
        if epoch % opt.saveInterval == 0:
            # do saving
            encoder_model = {'model': encoder.state_dict(), 'optimizer': encoder_optimizer.state_dict()}
            torch.save(
                encoder.state_dict(), '{0}/encoder_epoch_{1}.pth.tar'.format(opt.experiment, epoch)
            )
            decoder_model = {'model': decoder.state_dict(), 'optimizer': decoder_optimizer.state_dict()}
            torch.save(
                decoder.state_dict(), '{0}/decoder_epoch_{1}.pth.tar'.format(opt.experiment, epoch)
            )

        if opt.cuda:
            torch.cuda.empty_cache()

        # val all
        acc_val, loss_val_avg = valid(opt, encoder, decoder, batch_size=valid_batch_size, data_loader=valid_loader,
                                      criterion=criterion, test_all=True)
        print('Time: ', time.strftime('%Y-%m-%d %H:%M:%S'), 'Acc: %f, Loss: %f' % (acc_val, loss_val_avg.val()),
              'It\'s time to save one model')
        loss_val_avg.reset()


if __name__ == '__main__':
    main()
