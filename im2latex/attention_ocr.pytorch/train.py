# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2021/3/18
import sys
import time
import random
import warnings

import cv2
import torch.optim as optim
from torch.utils.data import dataloader
from torchsummary import summary
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from config import get_options
from dataset import Im2latexDataset, AlignCollate, DataPrefetcher, ResizeNormalize
from model_ocr import EncoderCRNN, Decoder
from utils import *


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def trainBatch(opt, img, text, encoder, decoder, encoder_optimizer, decoder_optimizer,
               criterion=torch.nn.NLLLoss()):
    if not opt.cuda:
        decoder_input = get_decoder_input(text, 0)
        decoder_hidden = decoder.initHidden(opt.batch_size)
        encoder_output = encoder(img)
    else:
        decoder_input = text[:, 0].cuda(non_blocking=True)
        decoder_hidden = decoder.initHidden(opt.batch_size).cuda(non_blocking=True)
        encoder_output = encoder(img).cuda(non_blocking=True)

    loss = 0.0
    length = len(text[0])

    for i in range(1, length):
        # print(decoder_input.shape, decoder_hidden.shape, encoder_output.shape)
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output)
        # print(decoder_output.shape, decoder_input.shape, i)
        if not opt.cuda:
            decoder_input = get_decoder_input(text, i)
        else:
            decoder_input = text[:, i].cuda(non_blocking=True)

        loss += criterion(decoder_output, decoder_input)

        # print(decoder_input)
        if (decoder_input == 2).all():
            break
    # print(loss)

    # if (times + 1) % opt.niter == 0:
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


@torch.no_grad()
def valid(opt, encoder, decoder, data_loader, max_iter=1, criterion=torch.nn.NLLLoss(),
          batch_size=1, test_all=False):
    test_iter = iter(data_loader)

    if not test_all:
        max_iter = min(max_iter, len(data_loader))
    else:
        max_iter = len(data_loader)

    encoder.eval()
    decoder.eval()

    bleu = 0.0
    chencherry = SmoothingFunction()
    for i in range(1, max_iter + 1):
        data = test_iter.__next__()
        img, text = data

        length = len(text[0])
        # print(length)
        decoded_labels = ["0"]
        decoder_input = torch.LongTensor(batch_size).fill_(0)
        decoder_hidden = decoder.initHidden(batch_size)
        encoder_output = encoder(img)

        if opt.cuda:
            decoder_input = decoder_input.cuda(non_blocking=True)
            decoder_hidden = decoder_hidden.cuda(non_blocking=True)
            encoder_output = encoder_output.cuda(non_blocking=True)

        for k in range(1, length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output)
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(1)
            decoded_labels.append(str(int(decoder_input)))

        t = str(text[0].tolist())[1:-1].split(", ")
        # print(t[1:-1], decoded_labels[1:-1])
        bleu += corpus_bleu(t[1:-1], decoded_labels[1:-1], smoothing_function=chencherry.method1)
    return bleu / max_iter


def main():
    opt = get_options()
    print("-----init-----")

    manual_seed = 118
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    nc = 1
    already_epoch = 0

    if opt.cuda:
        import torch.backends.cudnn as cudnn
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = True

    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE and not opt.cuda:
        warnings.warn("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if not CUDA_AVAILABLE and opt.cuda:
        assert torch.cuda.is_available(), "ERROR: You don't have a CUDA device"
    if not opt.cuda and opt.preload:
        assert (opt.preload and not CUDA_AVAILABLE), "ERROR: You don't have a CUDA device"
    if opt.cuda and not opt.preload:
        warnings.warn("WARNING: You choosed the CUDA, so you could run with --preload")

    # train dataloader init
    train_dataset = Im2latexDataset(split="train", transform=None)
    n_class = train_dataset.nclass + 3
    # print(n_class)

    assert train_dataset

    train_loader = dataloader.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        shuffle=True, num_workers=int(opt.workers),
        collate_fn=AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio),
        pin_memory=True, drop_last=True
    )
    length = len(train_loader)

    # test dataloader init
    test_dataset = Im2latexDataset(split="test", transform=ResizeNormalize(opt.imgH, opt.imgW))

    assert test_dataset
    # ATTENTION: you can't modified it
    test_batch_size = 1
    test_loader = dataloader.DataLoader(
        train_dataset, batch_size=test_batch_size,
        shuffle=True, num_workers=int(opt.workers),
        collate_fn=AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio),
        pin_memory=True, drop_last=True
    )

    # # valid dataloader init
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
    decoder = Decoder(opt.nh, n_class, dropout_p=0.5, batch_size=opt.batch_size)

    # setup optimizer
    if opt.adam:
        encoder_optimizer = optim.Adadelta(encoder.parameters(), lr=opt.lr, eps=1e-8, weight_decay=1e-3)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.lr, eps=1e-8, betas=(opt.beta1, 0.999),
                                       weight_decay=1e-3)
    else:
        encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=opt.lr)
        decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=opt.lr)

    encoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(encoder_optimizer, T_max=opt.niter)
    decoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, T_max=opt.niter)

    encoder.apply(weights_init)
    decoder.apply(weights_init)
    for e, d in zip(encoder.parameters(), decoder.parameters()):
        e.requires_grad = True
        d.requires_grad = True

    # choose loss
    criterion = torch.nn.NLLLoss()

    if opt.cuda:
        encoder.cuda()
        decoder.cuda()
        criterion.cuda()
    # summary(encoder, input_size=[(1, opt.imgH, opt.imgW)], batch_size=opt.batch_size, device="cpu")
    # summary(decoder, input_size=[(opt.batch_size)], batch_size=opt.batch_size, device="cpu")

    # train
    print("-----train-----")
    t0 = time.time()
    for epoch in range(opt.niter):
        i = 0

        if opt.preload:
            train_loader = DataPrefetcher(train_loader)

        for img, text in train_loader:
            encoder.train()
            decoder.train()

            loss = trainBatch(opt, img, text, encoder, decoder, encoder_optimizer, decoder_optimizer,
                              criterion=criterion)
            i += 1

            if i % opt.displayInterval == 0:
                # do val
                bleu = valid(opt, encoder, decoder, batch_size=test_batch_size, data_loader=test_loader,
                             criterion=criterion)
                print('[%d/%d][%d/%d] TrainLoss: %f Bleu Score: %f' % (
                    epoch + already_epoch, opt.niter, i, length, loss, bleu), end=' ')
                t1 = time.time()
                print('Time: %s' % str(t1 - t0))
                t0 = time.time()
            # for name, parameters in decoder.state_dict().items():
            #     if "weight" in name:
            #         print(name, ": ", parameters.detach().numpy())
        encoder_scheduler.step()
        decoder_scheduler.step()

        # print('-----Model saved-----')
        # if epoch % opt.saveInterval == 0:
        #     # do saving
        #     encoder_model = {'model': encoder.state_dict(), 'optimizer': encoder_optimizer.state_dict(), 'epoch': epoch}
        #     torch.save(
        #         encoder_model, '{0}/encoder_epoch_{1}.pth.tar'.format(opt.experiment, epoch)
        #     )
        #     decoder_model = {'model': decoder.state_dict(), 'optimizer': decoder_optimizer.state_dict(), 'epoch': epoch}
        #     torch.save(
        #         decoder_model, '{0}/decoder_epoch_{1}.pth.tar'.format(opt.experiment, epoch)
        #     )

        # # val all
        # acc_val, loss_val_avg = valid(opt, encoder, decoder, batch_size=valid_batch_size,
        # data_loader=valid_loader, test_all=True) print('Time: ', time.strftime('%Y-%m-%d %H:%M:%S'), 'Acc: %f,
        # Loss: %f' % (acc_val, loss_val_avg.val()), 'It\'s time to save one model') loss_val_avg.reset()


if __name__ == '__main__':
    main()
