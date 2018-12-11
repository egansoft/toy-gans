import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch import distributions as dist
import random
import argparse
import numpy as np
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(opt.nlatent, opt.nhidden),
            nn.LeakyReLU(),
            nn.Linear(opt.nhidden, opt.nhidden),
            nn.LeakyReLU(),
            nn.Linear(opt.nhidden, 1),
        )

    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.loss = opt.loss
        self.main = nn.Sequential(
            nn.Linear(1, opt.nhidden),
            nn.LeakyReLU(),
            nn.Linear(opt.nhidden, opt.nhidden),
            nn.LeakyReLU(),
            nn.Linear(opt.nhidden, 1),
        )

    def forward(self, x):
        out = self.main(x)
        if self.loss == 'hinge':
            return torch.tanh(out)
        return torch.sigmoid(out)

def train(opt, D, G, underlying):
    optimizer_d = torch.optim.Adam(D.parameters(), lr=opt.lr)
    optimizer_g = torch.optim.Adam(G.parameters(), lr=opt.lr)

    for i in xrange(opt.niter):
        # Update D
        D.zero_grad()
        x_real = sample_underlying(opt, underlying)
        decide_real = D(x_real)

        z = torch.randn(opt.nbatch, opt.nlatent)
        x_fake = G(z)
        decide_fake = D(x_fake.detach())
        
        d_error = d_loss(opt, decide_real, decide_fake)
        d_error.backward()
        optimizer_d.step()

        # Update G
        G.zero_grad()
        decide_fake = D(x_fake)

        g_error = g_loss(opt, decide_fake)
        g_error.backward()
        optimizer_g.step()

        if (i+1) % opt.printfreq == 0:
            kl, buckets = eval_g(opt, underlying, G)
            print (i+1), d_error.item(), g_error.item(), kl

    demo_g(opt, underlying, G)

def decide_underlying(opt):
    weights = np.random.rand(opt.ntrue)
    weights /= np.sum(weights)
    means = np.random.rand(opt.ntrue) * 0.8 + 0.1
    stdevs = np.random.rand(opt.ntrue) * opt.stdevcap

    print 'weights', weights
    print 'means', means
    print 'stdevs', stdevs
    return (weights, means, stdevs)

def sample_underlying(opt, (weights, means, stdevs)):
    sample = torch.zeros(opt.nbatch, 1)
    for i in xrange(len(sample)):
        idx = np.random.choice(range(len(weights)), p=weights)
        sample[i, 0] = np.random.normal(means[idx], stdevs[idx])
    return sample

def d_loss(opt, decide_real, decide_fake):
    if opt.loss == 'logistic':
        score_real = torch.mean(torch.log(decide_real))
        inverted = torch.ones(opt.nbatch, 1) - decide_fake
        score_fake = torch.mean(torch.log(inverted))
        value = score_real + score_fake
        return -value
    if opt.loss == 'square':
        score_real = torch.mean(torch.pow(torch.add(decide_real, -1), 2))
        score_fake = torch.mean(torch.pow(decide_fake, 2))
        return 0.5 * score_real + 0.5 * score_fake
    if opt.loss == 'hinge':
        inv_real = torch.ones(opt.nbatch, 1) - decide_real
        score_real = torch.mean(torch.clamp(inv_real, min=0))
        inv_fake = torch.add(decide_fake, 1)
        score_fake = torch.mean(torch.clamp(inv_fake, min=0))
        return score_real + score_fake

def g_loss(opt, decide_fake):
    if opt.loss == 'logistic':
        value = torch.mean(torch.log(decide_fake))
        return -value
    if opt.loss == 'square':
        loss = torch.mean(torch.pow(torch.add(decide_fake, -1), 2))
        return loss
    if opt.loss == 'hinge':
        loss = -torch.mean(decide_fake)
        return loss

def likelihood(x_start, x_end, opt, (weights, means, stdevs)):
    low = norm.cdf((x_start * np.ones(opt.ntrue) - means) / stdevs)
    high = norm.cdf((x_end * np.ones(opt.ntrue) - means) / stdevs)
    ps = high - low
    p = np.dot(weights, ps)
    return p

def eval_g(opt, underlying, G):
    # Evaluate G through discretized KL divergence
    buckets = np.zeros(opt.nbuckets)
    for i in xrange(opt.ntests):
        z = torch.randn(opt.nbatch, opt.nlatent)
        x_fake = G(z)
        for x in x_fake:
            bucket = int(x.item() * opt.nbuckets)
            if bucket < 0 or bucket >= opt.nbuckets:
                continue
            buckets[bucket] += 1

    buckets = buckets / opt.ntests / opt.nbatch
    kl = 0
    for i, p_x in enumerate(buckets):
        x = float(i) / opt.nbuckets
        if p_x == 0:
            continue
        q_x = likelihood(x, x+1./opt.nbuckets, opt, underlying)
        if q_x < 1e-5:
            q_x = 1e-5
        div = p_x * np.log(p_x / q_x)
        kl += div

    return kl, buckets

def demo_g(opt, underlying, G):
    kl, buckets_g = eval_g(opt, underlying, G)
    buckets_r = np.zeros(opt.nbuckets)
    for i in xrange(opt.nbuckets):
        x = float(i) / opt.nbuckets
        q_x = likelihood(x, x+1./opt.nbuckets, opt, underlying)
        buckets_r[i] = q_x

    x_labels = np.linspace(0, 1, opt.nbuckets, endpoint=False)
    plt.bar(x_labels, buckets_g, alpha=0.5, label='g', color='b', width=1./opt.nbuckets)
    plt.bar(x_labels, buckets_r, alpha=0.5, label='r', color='g', width=1./opt.nbuckets)
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='logistic')
    parser.add_argument('--nbatch', type=int, default=100)
    parser.add_argument('--nhidden', type=int, default=30)
    parser.add_argument('--nlatent', type=int, default=10)
    parser.add_argument('--niter', type=int, default=2000)
    parser.add_argument('--ntrue', type=int, default=1)
    parser.add_argument('--stdevcap', type=float, default=0.01)
    parser.add_argument('--nbuckets', type=int, default=100)
    parser.add_argument('--ntests', type=int, default=100)
    parser.add_argument('--printfreq', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    opt = parser.parse_args()
    print opt

    underlying = decide_underlying(opt)
    D = Discriminator(opt)
    G = Generator(opt)
    train(opt, D, G, underlying)
