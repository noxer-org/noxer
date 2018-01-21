"""
Implementation of ACGAN, based on
https://github.com/znxlwm/pytorch-generative-model-collections.
"""

from sklearn.preprocessing import LabelBinarizer

import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, height, width, channels):
        super(generator, self).__init__()
        self.input_height = height
        self.input_width = width
        self.input_dim = 62 + 10
        self.output_dim = channels

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )
        initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, height, width, channels):
        super(discriminator, self).__init__()
        self.input_height = height
        self.input_width = width
        self.input_dim = channels
        self.output_dim = 1
        self.class_num = 10

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.dc = nn.Sequential(
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        self.cl = nn.Sequential(
            nn.Linear(1024, self.class_num),
        )
        initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        x = self.fc1(x)
        d = self.dc(x)
        c = self.cl(x)

        return d, c

class ACGAN(object):
    def __init__(self, height, width, channels,
                 use_gpu=True, epochs=32, batch_size=64,
                 lrG=0.0002, lrD=0.0002, beta1=0.5, beta2=0.999,
                 verbose=0):
        # parameters
        self.height = height
        self.width= width
        self.channels = channels
        self.epoch = epochs
        self.sample_num = 100
        self.batch_size = batch_size
        self.gpu_mode = use_gpu
        self.verbosity = verbose
        self.lrG = lrG
        self.lrD = lrD
        self.beta1 = beta1
        self.beta2 = beta2
        self.verbose = beta2

    def notify(self, message, verbosity=1):
        if self.verbosity >= verbosity:
            print(message)

    def train(self, X, Y, callback=None):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        # networks init
        self.G = generator(self.height, self.width, self.channels)
        self.D = discriminator(self.height, self.width, self.channels)
        G_optimizer = optim.Adam(self.G.parameters(), lr=self.lrG, betas=(self.beta1, self.beta2))
        D_optimizer = optim.Adam(self.D.parameters(), lr=self.lrD, betas=(self.beta1, self.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            BCE_loss = nn.BCELoss().cuda()
            CE_loss = nn.CrossEntropyLoss().cuda()
        else:
            BCE_loss = nn.BCELoss()
            CE_loss = nn.CrossEntropyLoss()

        # print('---------- Networks architecture -------------')
        # utils.print_network(self.G)
        # utils.print_network(self.D)
        # print('-----------------------------------------------')

        # load mnist
        self.z_dim = 62
        self.y_dim = 10

        # fixed noise & condition
        sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(10):
            sample_z_[i * self.y_dim] = torch.rand(1, self.z_dim)
            for j in range(1, self.y_dim):
                sample_z_[i * self.y_dim + j] = sample_z_[i * self.y_dim]

        temp = torch.zeros((10, 1))
        for i in range(self.y_dim):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(10):
            temp_y[i * self.y_dim: (i + 1) * self.y_dim] = temp

        sample_y_ = torch.zeros((self.sample_num, self.y_dim))
        sample_y_.scatter_(1, temp_y.type(torch.LongTensor), 1)

        # setup dataset
        data_Y, data_X = torch.FloatTensor(X), torch.FloatTensor(Y)

        if self.gpu_mode:
            y_real_, y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(torch.zeros(self.batch_size, 1).cuda())
        else:
            y_real_, y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))

        self.D.train()
        self.notify('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter in range(len(data_X) // self.batch_size):
                x_ = data_X[iter*self.batch_size:(iter+1)*self.batch_size]
                z_ = torch.rand((self.batch_size, self.z_dim))
                y_vec_ = data_Y[iter*self.batch_size:(iter+1)*self.batch_size]

                if self.gpu_mode:
                    x_, z_, y_vec_ = Variable(x_.cuda()), Variable(z_.cuda()), Variable(y_vec_.cuda())
                else:
                    x_, z_, y_vec_ = Variable(x_), Variable(z_), Variable(y_vec_)

                # update D network
                D_optimizer.zero_grad()

                D_real, C_real = self.D(x_)
                D_real_loss = BCE_loss(D_real, y_real_)
                mxv = torch.max(y_vec_, 1)[1]
                C_real_loss = CE_loss(C_real, mxv)

                G_ = self.G(z_, y_vec_)
                D_fake, C_fake = self.D(G_)
                D_fake_loss = BCE_loss(D_fake, y_fake_)
                mxv = torch.max(y_vec_, 1)[1]
                C_fake_loss = CE_loss(C_fake, mxv)

                D_loss = D_real_loss + C_real_loss + D_fake_loss + C_fake_loss
                self.train_hist['D_loss'].append(D_loss.data[0])

                D_loss.backward()
                D_optimizer.step()

                # update G network
                G_optimizer.zero_grad()

                G_ = self.G(z_, y_vec_)
                D_fake, C_fake = self.D(G_)

                G_loss = BCE_loss(D_fake, y_real_)
                C_fake_loss = CE_loss(C_fake, torch.max(y_vec_, 1)[1])

                G_loss += C_fake_loss
                self.train_hist['G_loss'].append(G_loss.data[0])

                G_loss.backward()
                G_optimizer.step()


                if ((iter + 1) % 10) == 0:
                    self.notify("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), len(data_X) // self.batch_size, D_loss.data[0], G_loss.data[0]))

            if callback is not None:
                callback()

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            self.notify(epoch)

        self.train_hist['total_time'].append(time.time() - start_time)
        self.notify("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))

from .base import GeneratorBase


def gpu_setting(kwargs):
    if 'gpu' in kwargs:
        use_gpu = kwargs['gpu']
    else:
        use_gpu = True

    return use_gpu


def get_callback(kwargs):
    if 'callback' in kwargs:
        return kwargs['callback']
    return None


class ACGANCategoryToImageGenerator(GeneratorBase):
    def __init__(self, use_gpu=True, epochs=32, batch_size=64,
                 lrG=0.0002, lrD=0.0002, beta1=0.5, beta2=0.999,
                 verbose=0):
        self.use_gpu = use_gpu
        self.epochs = epochs
        self.batch_size = batch_size
        self.lrG = lrG
        self.lrD = lrD
        self.beta1 = beta1
        self.beta2 = beta2
        self.verbose = verbose

        self.net = None
        self.bin = None

    def fit(self, X, Y, **kwargs):
        # shuffle dimensions to fit to pytorch conventions
        Y = np.transpose(Y, (0, 3, 1, 2))

        height = Y.shape[2]
        width = Y.shape[3]
        channels = Y.shape[1]

        # binarize the labels
        self.bin = LabelBinarizer()
        X = self.bin.fit_transform(X)

        self.net = ACGAN(
            height, width, channels,
            use_gpu=gpu_setting(kwargs),
            epochs = self.epochs,
            batch_size = self.batch_size,
            lrG = self.lrG,
            lrD = self.lrD,
            beta1 = self.beta1,
            beta2 = self.beta2,
            verbose = self.verbose,
        )

        self.net.train(X, Y, callback=get_callback(kwargs))
        self.net.D.eval()
        self.net.G.eval()


    def predict_noise(self, X, Z, **kwargs):
        if self.net is None:
            raise RuntimeError("Please run the fitting procedure first!")

        self.net.G.eval()

        iX = torch.FloatTensor(self.bin.transform(X))

        if gpu_setting(kwargs):
            iZ, iX = Variable(Z.cuda(), volatile=True), Variable(iX.cuda(), volatile=True)
        else:
            iZ, iX = Variable(Z, volatile=True), Variable(iX, volatile=True)

        Y = self.net.G(iZ, iX)
        Y = Y.cpu().data.numpy()
        Y = np.transpose(Y, (0, 2, 3, 1))
        return Y

    def predict(self, X, **kwargs):
        # generate noise
        Z = torch.rand((len(X), self.net.z_dim))

        # make generation
        return self.predict_noise(X, Z, **kwargs)
