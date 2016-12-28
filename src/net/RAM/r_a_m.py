import sys
sys.path.append('./src/net/RAM')
sys.path.append('./src/common/loss_functions')
import chainer
from chainer import cuda, Variable
import chainer.functions as F
import chainer.links as L

import numpy as np
import cv2
from crop import crop
from loss_func_utils import loglikelihood


class RAM(chainer.Chain):
    def __init__(self, n_class, in_ch, n_e=128, n_h=256, g_size=13, n_step=8, scale=3, var=7.5):
        super().__init__(
            emb_l = L.Linear(2, n_e),  # embed location
            emb_x = L.Linear(g_size*g_size*scale*in_ch, n_e),  # embed image
            fc_loc_to_glimpse = L.Linear(n_e, n_h),  # loc to glimpse
            fc_image_to_glimpse = L.Linear(n_e, n_h),  # image to glimpse
            core_lstm = L.LSTM(n_h, n_h),  # core LSTM. in the paper, named recurrent network.
            fc_ha = L.Linear(n_h, n_class),  # core to action(from core_lstm to action)
            fc_hl = L.Linear(n_h, 2),  # core to loc(from core_lstm to loc). in the paper, named emission network.
            fc_hb = L.Linear(n_h, 1),  # core to baseline(from core_lstm to baseline)
        )
        self.g_size = g_size
        self.n_step = n_step
        self.scale = scale
        self.var = var

        self.train = True
        self.n_class = n_class
        self.active_learn = False

    def clear(self):
        self.loss = None
        self.accuracy = None
        self.core_lstm.reset_state()

    def vec2mat(self, x):
        shape = x.shape
        if len(shape)==2:
            n_image, n_element = shape
            h = w = int(n_element**0.5)
            return F.reshape(x, (n_image, 1, h, w))
        else:
            return x

    def __call__(self, x, t):
        x = self.vec2mat(x)
        self.clear()
        x.volatile = not self.train

        batch_size = x.data.shape[0] # batch size
        accum_negative_loglikelihood = 0

        # init mean location
        # in the papser l_init from context network with I_coarse(low resolution raw image).
        location = Variable( \
                np.random.uniform(-1, 1, size=(batch_size,2)).astype(np.float32), \
            volatile=not self.train)

        # forward n_steps times
        for i in range(self.n_step - 1):
            location, negative_loglikelihood, y, baseline = \
                                            self.forward(x, location)
            accum_negative_loglikelihood += -negative_loglikelihood

        location, negative_loglikelihood, y, baseline = \
                                            self.forward(x, location)

        # loss with softmax cross entropy
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)

        if self.train:
            # reward
            reward = self.xp.where( \
                        self.xp.argmax(y.data,axis=1)==t.data, 1, 0)
            # MSE between cost and baseline
            self.loss += F.sum((reward-baseline)*(reward-baseline))/batch_size
            # # truncate baseline
            baseline = Variable(baseline.data, volatile=not self.train)
            # loss with reinforce rule
            self.loss += 1e-1*F.sum(accum_negative_loglikelihood*(reward-baseline))/batch_size
        chainer.report({'loss': self.loss, 'accuracy': self.accuracy}, self)
        return self.loss

    def loc_net(self, h_truncated, _sampling):
        """
        Location Net(Emission network): truncate h
        calculate next location.
        """
        next_location_mean = F.clip(self.fc_hl(h_truncated), -1., 1.)

        if _sampling:
            # generate sample from N(mean,var)
            # add noise to next location
            # solve simple Kalman Filter
            eps = self.xp.random.normal(0,1,size=next_location_mean.data.shape).astype(np.float32)
            # add noise to next location
            next_location = next_location_mean + self.xp.sqrt(self.var)*eps
            next_location = F.clip(next_location, -1., 1.)
        else:
            next_location = next_location_mean

        # need truncation, because location is input value.
        next_location = Variable(next_location.data, volatile=not self.train)
        return next_location, next_location_mean

    def glimpse_net(self, glimpse, location):
        # glimpse network includes next three nodes.
        h_glimpse = F.elu(self.emb_x(glimpse))

        # Location Encoding
        h_location = F.elu(self.emb_l(location))

        # g is the  final glimpse feature vector
        # equal to f_g(theta_g)
        g = F.elu(self.fc_loc_to_glimpse(h_location) + \
                F.reshape(self.fc_image_to_glimpse(h_glimpse), (glimpse.data.shape[0],-1))
            )
        return g

    def compute_glimpse(self, x, location):
        # Retina Encoding
        glimpse = crop(x, center=location.data, size=self.g_size)

        # multi-scale glimpse
        for k in range(1, self.scale):
            s = self.xp.power(2,k)
            patch = crop(x, center=location.data, size=self.g_size*s)
            patch = F.average_pooling_2d(patch, ksize=s)
            glimpse = F.concat((glimpse, patch), axis=1)

        return glimpse

    def forward(self, x, location):
        """
        forward one step
        """
        glimpse = self.compute_glimpse(x, location)

        # Glimpse Net
        g = self.glimpse_net(glimpse, location)

        # Core Net
        # named recurrent network, in the paper .
        # equal to f_h(thita_h)
        # this node calculate belief state.
        h = self.core_lstm(g)

        h_truncated = Variable(h.data, volatile=not self.train)
        # Location Net(Emission network): truncate h
        # calculate next location.
        # use h, when calculate class and baseline.
        next_location, next_location_mean = self.loc_net(h_truncated, self.train)

        # calculate location policy
        negative_loglikelihood = \
            loglikelihood(next_location_mean, next_location, self.var)

        # Action Net
        # classification network
        # y is infered class.
        # use h, not use h_truncated.
        y = self.fc_ha(h)

        # Baseline
        # use h, not use h_truncated.
        baseline = F.clip(self.fc_hb(h), 0., 1.)
        baseline = F.reshape(baseline, (-1,))
        return location, negative_loglikelihood, y, baseline
