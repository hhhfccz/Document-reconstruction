# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2021/3/18
import argparse


def get_options():

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers, you had better put it '
                                                               '4 times of your gpu')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size, default=32')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network, default=32')
    parser.add_argument('--imgW', type=int, default=280, help='the width of the input image to network, default=280')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state, default=256')
    parser.add_argument('--niter', type=int, default=20, help='number of epochs to train for, default=10')
    parser.add_argument('--lr', type=float, default=0.0015, help='select the max learning rate, default=2e-3')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--experiment',
                        default='/home/hhhfccz/im2latex/attention_ocr.pytorch/expr/attention_ocr/',
                        help='where to store samples and models')
    parser.add_argument('--displayInterval', type=int, default=7, help='Interval to be displayed')
    parser.add_argument('--saveInterval', type=int, default=2, help='Interval to be displayed')
    parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
    parser.add_argument('--adam', action='store_true', default=False, help='whether to use adam')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
    parser.add_argument('--preload', action='store_true', default=False, help='enables preload')
    parser.add_argument('--continue_train', action='store_true', default=False, help="whether to continue training")
    opt = parser.parse_args()

    return opt
