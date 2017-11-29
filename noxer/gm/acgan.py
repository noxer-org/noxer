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
        self.output_dim = channels
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
        self.epoch = epochs
        self.sample_num = 100
        self.batch_size = batch_size
        self.gpu_mode = use_gpu
        self.verbosity = verbose

        # networks init
        self.G = generator(height, width, channels)
        self.D = discriminator(height, width, channels)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=lrG, betas=(beta1, beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=lrD, betas=(beta1, beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
            self.CE_loss = nn.CrossEntropyLoss()

        #print('---------- Networks architecture -------------')
        #utils.print_network(self.G)
        #utils.print_network(self.D)
        #print('-----------------------------------------------')

        # load mnist
        self.z_dim = 62
        self.y_dim = 10

        # fixed noise & condition
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(10):
            self.sample_z_[i*self.y_dim] = torch.rand(1, self.z_dim)
            for j in range(1, self.y_dim):
                self.sample_z_[i*self.y_dim + j] = self.sample_z_[i*self.y_dim]

        temp = torch.zeros((10, 1))
        for i in range(self.y_dim):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(10):
            temp_y[i*self.y_dim: (i+1)*self.y_dim] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.y_dim))
        self.sample_y_.scatter_(1, temp_y.type(torch.LongTensor), 1)
        if self.gpu_mode:
            self.sample_z_, self.sample_y_ = Variable(self.sample_z_.cuda(), volatile=True), Variable(self.sample_y_.cuda(), volatile=True)
        else:
            self.sample_z_, self.sample_y_ = Variable(self.sample_z_, volatile=True), Variable(self.sample_y_, volatile=True)

    def notify(self, message, verbosity=1):
        if self.verbosity > verbosity:
            print(message)

    def train(self, X, Y):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        # setup dataset
        data_Y, data_X = torch.LongTensor(X), torch.FloatTensor(Y)

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(torch.zeros(self.batch_size, 1).cuda())
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))

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
                self.D_optimizer.zero_grad()

                D_real, C_real = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)
                C_real_loss = self.CE_loss(C_real, torch.max(y_vec_, 1)[1])

                G_ = self.G(z_, y_vec_)
                D_fake, C_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)
                C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])

                D_loss = D_real_loss + C_real_loss + D_fake_loss + C_fake_loss
                self.train_hist['D_loss'].append(D_loss.data[0])

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_, y_vec_)
                D_fake, C_fake = self.D(G_)

                G_loss = self.BCE_loss(D_fake, self.y_real_)
                C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])

                G_loss += C_fake_loss
                self.train_hist['G_loss'].append(G_loss.data[0])

                G_loss.backward()
                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    self.notify("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), len(data_X) // self.batch_size, D_loss.data[0], G_loss.data[0]))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            self.visualize_results((epoch+1))

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

    def fit(self, X, Y, **kwargs):

        height = Y.shape[1]
        width = Y.shape[2]
        channels = Y.shape[3]

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

        self.net.train(X, Y)

    def predict_noise(self, X, Z, **kwargs):
        if self.net is None:
            raise RuntimeError("Please run the fitting procedure first!")

        self.net.G.eval()

        temp = torch.LongTensor(X)
        iX = torch.FloatTensor(len(X), 10)
        iX.zero_()
        iX.scatter_(1, temp, 1)

        if gpu_setting(kwargs):
            iZ, iX = Variable(Z.cuda(), volatile=True), Variable(iX.cuda(), volatile=True)
        else:
            iZ, iX = Variable(Z, volatile=True), Variable(iX, volatile=True)

        Y = self.net.G(torch.FloatTensor(iZ), torch.LongTensor(iX))
        return Y

    def predict(self, X, **kwargs):
        Z = torch.rand((self.batch_size, self.z_dim))
        return self.predict_noise(X, Z, **kwargs)
