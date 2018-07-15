import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import mxnet as mx
import argparse
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd
import numpy as np
import logging
from datetime import datetime
import os
import time

def fill_buf(buf, i, img, shape):
    n = buf.shape[0]//shape[1]
    m = buf.shape[1]//shape[0]

    sx = (i%m)*shape[0]
    sy = (i//m)*shape[1]
    buf[sy:sy+shape[1], sx:sx+shape[0], 0] = img
    buf[sy:sy+shape[1], sx:sx+shape[0], 1] = img
    buf[sy:sy+shape[1], sx:sx+shape[0], 2] = img
    return None

def visual(title, X, name):
    print(X.shape)
    X = X.reshape((-1, 28, 28))
    X = np.clip((X - np.min(X))*(255.0/(np.max(X) - np.min(X))), 0, 255).astype(np.uint8)
    n = np.ceil(np.sqrt(X.shape[0]))
    buff = np.zeros((int(n*X.shape[1]), int(n*X.shape[2]), 3), dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[1:3])
        buff = buff[:,:,::-1]
        plt.imshow(buff)
        plt.title(title)
        plt.savefig(name)

class netG(nn.Block):
    def __init__(self, **kwargs):
        super(netG, self).__init__(**kwargs)
        with self.name_scope():
            self.fcz = nn.HybridSequential()
            with self.fcz.name_scope():
                self.fcz.add(nn.Dense(256),
                             nn.BatchNorm(),
                             nn.LeakyReLU(0.0))

            self.fcy = nn.HybridSequential()
            with self.fcy.name_scope():
                self.fcy.add(nn.Dense(256),
                             nn.BatchNorm(),
                             nn.LeakyReLU(0.0))
            self.rest = nn.Sequential()
            with self.rest.name_scope():
                self.rest.add(nn.Dense(512),
                              nn.BatchNorm(),
                              nn.LeakyReLU(0.0),
                              nn.Dense(1024),
                              nn.BatchNorm(),
                              nn.LeakyReLU(0.0),
                              nn.Dense(784),
                              nn.Lambda('tanh'))

    def forward(self, z, y):
        fcz_outputs = self.fcz(z)
        fcy_outputs = self.fcy(y)
        rest_inputs = mx.nd.concat(fcz_outputs, fcy_outputs, dim=1)
        output = self.rest(rest_inputs)
        return output

class netD(nn.Block):
    def __init__(self, **kwargs):
        super(netD, self).__init__(**kwargs)
        self.fcX = nn.HybridSequential()
        with self.fcX.name_scope():
            self.fcX.add(nn.Dense(1024),
                         nn.LeakyReLU(0.2))
        self.fcy = nn.HybridSequential()
        with self.fcy.name_scope():
            self.fcy.add(nn.Dense(1024),
                         nn.LeakyReLU(0.2))
        self.final = nn.HybridSequential()
        with self.final.name_scope():
            self.final.add(nn.Dense(512),
                           nn.BatchNorm(),
                           nn.LeakyReLU(0.2),
                           nn.Dense(256),
                           nn.BatchNorm(),
                           nn.LeakyReLU(0.2),
                           nn.Dense(1, activation='sigmoid'))

    def forward(self, X, y):
        fcX_outputs = self.fcX(X)
        fcy_outputs = self.fcy(y)
        final_inputs = mx.nd.concat(fcX_outputs, fcy_outputs, dim=1)
        final_outputs = self.final(final_inputs)
        return final_outputs

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64, help='input batch size')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--nepoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda',action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./results', help='folder to output images and model checkpoints')
parser.add_argument('--check-point', default=True, help="save results at each epoch or not")

opt = parser.parse_args()
print(opt)
logging.basicConfig(level=logging.DEBUG)
nz = opt.nz
if opt.cuda:
    ctx = mx.gpu(0)
else:
    ctx = mx.cpu()

def transformer(data, label):
    data = data.reshape((1, 784))
    one_hot = mx.nd.zeros((10, ))
    one_hot[label] = 1
    # normalize to [-1, 1]
    data = data.astype(np.float32)/128 - 1
    return data, one_hot

train_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST('./data', train=True, transform=transformer),
    batch_size=opt.batch_size, shuffle=True, last_batch='discard')

val_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST('./data', train=False, transform=transformer),
    batch_size=opt.batch_size, shuffle=False)

real_label = mx.nd.ones((opt.batch_size, ), ctx=ctx)
fake_label = mx.nd.zeros((opt.batch_size, ), ctx=ctx)

metric = mx.metric.Accuracy()
GNet = netG()
DNet = netD()

outf = opt.outf

# loss
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)

# initialize the generator and the discriminator
GNet.initialize(mx.init.Normal(0.02), ctx=ctx)
DNet.initialize(mx.init.Normal(0.02), ctx=ctx)

# trainer for the generator and the discriminator
trainerG = gluon.Trainer(GNet.collect_params(), 'adam', {'learning_rate': opt.lr, 'beta1': opt.beta1})
trainerD = gluon.Trainer(DNet.collect_params(), 'adam', {'learning_rate': opt.lr, 'beta1': opt.beta1})

print("Training...")

stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')

iter = 0
for epoch in range(opt.nepoch):
    tic = time.time()
    btic = time.time()
    for data, label in train_data:
        if iter == 1:
            mx.profiler.set_config(profile_all=True, filename='cgan_profile_output.json')
            mx.profiler.set_state('run')
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        noise = mx.nd.random.normal(0, 1, shape=(opt.batch_size, 1, nz), ctx=ctx)
        noise_label = label.copy()

        with autograd.record():
            output = DNet(data, label)
            errD_real = loss(output, real_label)
            metric.update([real_label, ], [output, ])
            fake = GNet(noise, noise_label)
            output = DNet(fake.detach(), label)
            errD_fake = loss(output, fake_label)
            errD = errD_real + errD_fake
            metric.update([fake_label, ], [output, ])
            errD.backward()

        trainerD.step(opt.batch_size)

        with autograd.record():
            output = DNet(fake, noise_label)
            errG = loss(output, real_label)
            errG.backward()

        trainerG.step(opt.batch_size)

        name, acc = metric.get()
        logging.info('discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d' %(mx.nd.mean(errD).asscalar(), mx.nd.mean(errG).asscalar(), acc, iter, epoch))
        if iter % 1 == 0:
            visual('gout', fake.asnumpy(), name=os.path.join(outf,'fake_img_iter_%d.png' %iter))
            visual('data', data.asnumpy(), name=os.path.join(outf,'real_img_iter_%d.png' %iter))

        if iter == 3:
            mx.profiler.set_state('stop')
            break
        iter = iter + 1
        btic = time.time()

        name, acc = metric.get()
        metric.reset()
        logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
        logging.info('time: %f' % (time.time() - tic))

    break

