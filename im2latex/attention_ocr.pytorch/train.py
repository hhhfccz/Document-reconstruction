import os
import sys
import time
import json
import warnings

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import dataloader

from config import get_options
from dataset import Im2latexDataset, AlignCollate, DataPrefetcher, ResizeNormalize
from model_ocr import EncoderCRNN, Decoder
from utils import *


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def trainBatch(opt, img, text, encoder, decoder, encoder_optimizer, decoder_optimizer, times, epoch,
               criterion=torch.nn.NLLLoss()):
    if not opt.cuda:
        decoder_input = get_decoder_input(text, 0)
        decoder_hidden = decoder.initHidden(opt.batchSize)
        encoder_output = encoder(img)
    else:
        encoder.cuda()
        decoder.cuda()
        criterion.cuda()
        decoder_input = text[:, 0].cuda()
        decoder_hidden = decoder.initHidden(opt.batchSize).cuda()
        encoder_output = encoder(img).cuda()

    loss = 0.0

    for i in range(1, len(text[0])):
        # print(decoder_input.shape, decoder_hidden.shape, encoder_output.shape)
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output)
        # print(decoder_output.shape, decoder_input.shape, i)
        if not opt.cuda:
            decoder_input = get_decoder_input(text, i)
        else:
            decoder_input = text[:, i].cuda()

        if not epoch:
            decoder_outputs = decoder_output * enlarge_decoder_output(times)
            # print(decoder_outputs.shape, decoder_input.shape)
            # !!! don't use *= , will make torch confused !!!
            loss += criterion(decoder_outputs, decoder_input) / opt.batchSize
        else:
            loss += criterion(decoder_output, decoder_input) / opt.batchSize

    if (times + 1) % 5 == 0:
        encoder.zero_grad()
        decoder.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss.item()


def valid(opt, encoder, decoder, data_loader, max_iter=10, criterion=torch.nn.NLLLoss(),
          batch_size=16, get_loss=True, test_all=False):
    test_iter = iter(data_loader)

    if not test_all:
        max_iter = min(max_iter, len(data_loader))
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

    loss_val = 0.0
    acc_val = 0.0
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
            # [batch_size, n_class]

            # don't work, but i don't know why
            # topv, topi = decoder_output.topk(1)
            # decoder_input = topi.squeeze(1)
            # decoded_labels[l, :] = decoder_input

            # TODO: Too slow, but it works
            for p in range(batch_size):
                decoder_output_ = decoder_output[p, :].view(1, -1)
                topv, topi = decoder_output_.topk(1)
                decoder_input_ = topi.squeeze(1)
                decoded_labels[l, p] = decoder_input_
                decoder_input[p] = decoder_input_

        decoded_labels_ = decoded_labels.cpu().numpy()
        text_ = text.cpu().numpy()
        # print(decoded_labels_, text_)
        acc_val += get_acc(length, decoded_labels_, text_)

        if opt.cuda:
            decoder_input = decoder_input.cuda()

    if not get_loss:
        return acc_val / max_iter
    else:
        return acc_val / max_iter, loss_val.item()


def main():
    opt = get_options()
    print("-----init-----")

    manual_seed = 118
    torch.manual_seed(manual_seed)

    nc = 1
    already_epoch = 0

    if opt.cuda:
        import torch.backends.cudnn as cudnn
        cudnn.enabled = True
        cudnn.benchmark = True

    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE and not opt.cuda:
        warnings.warn("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if not CUDA_AVAILABLE and opt.cuda:
        assert torch.cuda.is_available(), "ERROR: You don't have a CUDA device"
    if not opt.cuda and opt.preload:
        assert (opt.preload and not CUDA_AVAILABLE), "ERROR: You don't have a CUDA device"
    if opt.cuda and not opt.preload:
        warnings.warn("WARNING: You choosed the CUDA, so you could run with --preload")

    # train dataset init
    train_dataset = Im2latexDataset(split="train", transform=None)
    n_class = train_dataset.nclass + 3
    # print(n_class)

    assert train_dataset

    train_loader = dataloader.DataLoader(
        train_dataset, batch_size=opt.batchSize,
        shuffle=True, num_workers=int(opt.workers),
        collate_fn=AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio),
        pin_memory=True
    )
    length = len(train_loader)

    # test dataset init
    test_dataset = Im2latexDataset(split="test", transform=ResizeNormalize(opt.imgH, opt.imgW))

    assert test_dataset
    # MODIFIED: you can modified it 
    test_batch_size = 16
    test_loader = dataloader.DataLoader(
        test_dataset, batch_size=test_batch_size,
        shuffle=False, num_workers=int(opt.workers),
        pin_memory=True
    )

    # # valid dataset init
    # valid_dataset = Im2latex_Dataset(split="validate", transform=ResizeNormalize(opt.imgH, opt.imgW))

    # assert valid_dataset
    # # MODIFIED: you can modified it 
    # valid_batch_size = 16
    # valid_loader = dataloader.DataLoader(
    #     valid_dataset, batch_size=valid_batch_size,
    #     shuffle=False, num_workers=int(opt.workers),
    #     pin_memory=True
    # )

    # network init
    encoder = EncoderCRNN(opt.imgH, nc, opt.nh)
    decoder = Decoder(opt.nh, n_class, dropout_p=0.3, batch_size=opt.batchSize)
    # setup optimizer
    if opt.adam:
        encoder_optimizer = optim.Adadelta(encoder.parameters(), lr=opt.lr, eps=1e-7, weight_decay=0.01)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.lr, eps=1e-7, betas=(opt.beta1, 0.999),
                                       weight_decay=0.01, amsgrad=True)
    else:
        encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=opt.lr)
        decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=opt.lr)

    encoder_scheduler = torch.optim.lr_scheduler.OneCycleLR(encoder_optimizer, max_lr=opt.lr/3, epochs=opt.niter,
                                                            steps_per_epoch=length, cycle_momentum=False)
    decoder_scheduler = torch.optim.lr_scheduler.OneCycleLR(decoder_optimizer, max_lr=opt.lr/3, epochs=opt.niter,
                                                            steps_per_epoch=length, cycle_momentum=False)

    encoder.apply(weights_init)
    decoder.apply(weights_init)

    print(encoder)
    print(decoder)

    # choose loss
    criterion = torch.nn.NLLLoss(ignore_index=2)

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
                              encoder_scheduler)
            encoder_scheduler.step()
            decoder_scheduler.step()
            i += 1

            if opt.preload:
                img, text = prefetcher.__next__()
            else:
                data = train_iter.__next__()
                img, text = data

            if i % opt.displayInterval == 0:
                # do val
                acc, loss_test = valid(opt, encoder, decoder, batch_size=test_batch_size, data_loader=test_loader,
                                       criterion=criterion)
                print('[%d/%d][%d/%d] TrainLoss: %f TestLoss: %f Acc: %f' % (
                    epoch + already_epoch, opt.niter, i, length, loss, loss_test, acc), end=' ')
                t1 = time.time()
                print('Time: %s' % str(t1 - t0))
                t0 = time.time()

        print('-----Model saved-----')
        if epoch % opt.saveInterval == 0:
            # do saving
            encoder_model = {'model': encoder.state_dict(), 'optimizer': encoder_optimizer.state_dict(), 'epoch': epoch}
            torch.save(
                encoder_model, '{0}/encoder_epoch_{1}.pth.tar'.format(opt.experiment, epoch)
            )
            decoder_model = {'model': decoder.state_dict(), 'optimizer': decoder_optimizer.state_dict(), 'epoch': epoch}
            torch.save(
                decoder_model, '{0}/decoder_epoch_{1}.pth.tar'.format(opt.experiment, epoch)
            )

        if opt.cuda:
            torch.cuda.empty_cache()

        # # val all
        # acc_val, loss_val_avg = valid(opt, encoder, decoder, batch_size=valid_batch_size,
        # data_loader=valid_loader, test_all=True) print('Time: ', time.strftime('%Y-%m-%d %H:%M:%S'), 'Acc: %f,
        # Loss: %f' % (acc_val, loss_val_avg.val()), 'It\'s time to save one model') loss_val_avg.reset()


if __name__ == '__main__':
    main()
