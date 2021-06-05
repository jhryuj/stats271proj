import os, time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle

# LQG function
# takes an input x.
# Takes observer/stimulus parameters
# Calculate Kalman and control filters

class LQG(object):
    def __init__(self, dynamics,
                 perfControl = False,
                 clip_gradients = False,
                 debug = False):
        ''''''
        self.nframes        = dynamics.config.nframes
        self.dtype          = dynamics.dtype
        self.debug          = debug
        self.clip_gradients = clip_gradients
        self.perfControl     = perfControl # use identity matrix rather than control

        # debug mode
        # if debug:
        #     self.clip_gradients = True

    def simulate(self, dyn, obs, x_fix = None,
                 L = None, K = None, batch_size=1):
        '''

        :param dyn: dynamics parameters
        :param params_observer: list of observer parameters
        :param B: batch size, per unique observer and dynamics parameter
        :param L list of control filters
        :param K list of kalman filters
        :return:
            - list of dictionary with fields
            - x_t, xhat_t, u_t, t_t
            -
        '''

        x_list     = []
        y_list   = []
        z_list   = []
        u_list     = []
        time_list     = []

        sparamslist = obs.sparamslist
        cparams     = obs.cparams

        for p in range(len(sparamslist)):
            currparams  = obs.generateParamMat(dyn,n=p)

            if L is None or K is None:
                currK, currL = self.calculateFilters(currparams, dyn)
            else:
                currL = L[p]
                currK = K[p]

            if x_fix is not None:
                x_tp1           = x_fix[min(p,len(x_fix)-1)][0,:,:,:] #(T,B,SS,1)
                batch_size  = x_fix[min(p,len(x_fix)-1)].shape[1]
            else:
                x_tp1       = tf.tile(tf.expand_dims(dyn.X0, axis=0),
                                  tf.constant([batch_size,1,1]))


            zt      = tf.zeros((batch_size,dyn.state_size,1),
                               dtype=self.dtype)
            ut_mean = tf.zeros((batch_size,dyn.control_size,1),
                               dtype=self.dtype)

            # make into a list
            x_n     = [x_tp1] # x0 already
            z_n  = []
            u_n     = []
            t_n     = []
            y_n     = []

            # simulate dynamics
            # technically should start
            for t in range(0, dyn.nframes):
                # x_{t+1}, yt, ut
                x_tp1, yt, zt, ut, ut_mean = \
                    self.simulate_onestep(x_tp1, zt, ut_mean,
                                          currK[t, :, :], currL[t,:,:],
                                          dyn, currparams,
                                          batch_size=batch_size)

                if x_fix is not None:
                    idx_remove = np.zeros(x_tp1.shape)
                    idx_remove[:,dyn.target_idx + dyn.target_vel_idx,:] = 1
                    idx_keep = np.ones(x_tp1.shape)
                    idx_keep[:,dyn.target_idx + dyn.target_vel_idx,:] = 0
                    x_tp1 = idx_keep * x_tp1 + \
                            idx_remove * x_fix[min(p,len(x_fix)-1)][t+1, :, : ,:]

                x_n     += [x_tp1]
                y_n     += [yt]
                z_n     += [zt]
                u_n     += [ut]
                t_n     += [t]


            x_list     += [tf.stack(x_n)]     # (T, state_size) # check dimensions
            y_list += [tf.stack(y_n)]
            z_list   += [tf.stack(z_n)]
            u_list     += [tf.stack(u_n)]
            time_list     += [tf.stack(t_n)]

        return x_list, z_list, u_list, time_list

    def simulate_onestep(self, xt, ztm1, utm1_mean, currKt, currLt,
                         dyn, currparams, batch_size=1):
        '''
        propagate dynamics one timestep
        Dynamics: x_{t + 1} = Ax_t + B[(I + C * rand)(u_t) + C0c * rand] + C0f * rand
        Feedback: y_t = [I + D * rand] H [x_t] + D0 * rand

        :param dyn: current dynamics object
        :param param_observer: current observer parameters
        :param B: batch size
        :return:
            x_{t+1}, y_t and their means.
            xt: shape = (batch_size, state_size)
            ut: shape = (batch_size, control_size)
        '''

        (batch_size1, state_size1, _)    = xt.shape
        control_size    = int(dyn.control_size)
        meas_size       = int(dyn.meas_size)
        state_size      = int(dyn.state_size)

        assert batch_size == batch_size1
        assert state_size == state_size1

        # matrices
        CM = currparams['CM']
        CA = currparams['CA']
        SM = currparams['SM']
        SA = currparams['SA']
        EA = currparams['EA']
        R  = currparams['R']

        # sample noise from gaussian distribution
        ca  = tf.random.normal((batch_size, dyn.control_size, 1), dtype=self.dtype)
        c0f = tf.random.normal((batch_size, state_size, 1), dtype=self.dtype)
        sa  = tf.random.normal((batch_size, meas_size,1), dtype=self.dtype) # sensory noise
        ea  = tf.random.normal((batch_size, state_size,1), dtype=self.dtype) # estimation noise

        # measurement
        yt_mean     = dyn.H @ xt
        yt          = yt_mean + SA @ sa

        # perception
        zt_mean     = dyn.A @ xt + dyn.B @ utm1_mean + \
                      currKt[None, :, :] @ (yt - dyn.H @ ztm1)
        zt          = zt_mean + EA @ ea # todo: check if the equations are okay with estimation noise

        # control
        ut_mean     = -1 * currLt[None, :, :] @ zt
        ut          = ut_mean + CA @ ca

        xtp1_mean = dyn.A @ xt + dyn.B @ ut
        xtp1      = xtp1_mean + dyn.C0f @ c0f

        return xtp1, yt, zt, ut, ut_mean


    def calculateFilters(self, param, dyn):
        A   = dyn.A
        B   = dyn.B
        H   = dyn.H
        Q   = dyn.Q
        C0f = dyn.C0f

        CM = param['CM']
        CA = param['CA']
        SM = param['SM']
        SA = param['SA']
        EA = param['EA']
        R  = param['R']

        # Noise (multiplicative ones are not used in LQG)
        # change from std to variance (i.e. original CA is the standard deviation; we convert them to variances)
        ## Dynamics: x_{t + 1} = Ax_t + B[(I + C * rand)(ut_mean) + CA * rand] + C0f * rand
        ## Feedback: y_t = [I + D * rand] H [x_t] + SA * rand

        C0_var  = C0f @ tf.transpose(C0f) + (B @ CA) @ tf.transpose(B @ CA) # additive noise; standard deviation => variance
        SA_var = SA @ tf.transpose(SA)

        # initialize weights
        K = [] # tf.zeros([dyn.nframes, dyn.state_size,dyn.meas_size], dtype=self.dtype)
        L = [] # tf.zeros([dyn.nframes, dyn.control_size,dyn.state_size], dtype=self.dtype)
        S1 = tf.zeros([dyn.state_size,dyn.state_size], dtype=self.dtype)
        X1 = tf.zeros([dyn.state_size,1], dtype=self.dtype)

        # estimator
        SiE = S1 # error covariance, intial value, zeros

        maxdKt =  0
        for t in range(0,self.nframes):
            # following Todorov 2005... but seems different from Simon 2006??
            # take inverse of cholesky rather than straight up inverse
            invsqrtvary = tf.linalg.inv(tf.linalg.cholesky(H @ SiE @ tf.transpose(H) + SA_var))
            newKt  = A @ SiE @ tf.transpose(H) @ (tf.transpose(invsqrtvary) @ invsqrtvary)
            newSiE = C0_var + (A - newKt @ H) @ SiE @ tf.transpose(A)

            if t > 0:
                dKt = tf.norm(newKt - K[t-1])
                maxdKt = max(maxdKt,dKt)

            if self.clip_gradients:
                K   += [self.grad_clipper(newKt, name='K' + str(t),debug=self.debug)]
                SiE      = tf.stop_gradient(newSiE) #todo: the main problem is here. is this a good approximation?
            else:
                K   += [newKt]
                SiE         = newSiE

        # controller (backward pass)
        SX = Q

        if self.perfControl:
            for t in range(self.nframes-1,-1,-1):
                L += tf.eye(Q.shape)

        else:
            maxdLt = 0
            for t in range(self.nframes-1,-1,-1):
                invsqrtvarSx    = tf.linalg.inv(tf.linalg.cholesky(R+tf.transpose(B)@SX@B))
                newLt           = tf.transpose(invsqrtvarSx) @ invsqrtvarSx @ tf.transpose(B) @ SX @ A
                newSX           = Q + tf.transpose(A) @ SX @ (A-B @ newLt)

                if t <  self.nframes-1:
                    dLt = tf.norm(newLt - L[-1])
                    maxdLt = max(maxdLt, dLt)

                if self.debug:
                    # weird that we get weights from states that are not estimable?
                    check01 = tf.transpose(A) @ SX @ (A-B @ newLt)
                    if check01[1,1] != 0:
                        check02 = check01
                    if t == 50:
                        check00 = tf.reduce_sum(newLt,axis=0)

                if self.clip_gradients:
                    L += [self.grad_clipper(newLt, name = 'L' + str(t),debug=self.debug)]
                    SX = tf.stop_gradient(newSX)
                else:
                    L += [newLt]
                    SX = newSX

        # reverse the calculated control filters in time.
        return tf.stack(K), tf.reverse(tf.stack(L),[0])

    # custom gradient clip for both clipping and debugging.
    @tf.custom_gradient
    def grad_clipper(y, name=None, debug=False):
        # dramatically clip gradients (scale them by norms)
        # todo: get rid of name and debug; this messes up eager execution.
        def backward(dy):
            norm = tf.norm(dy)
            if norm < 1e3:
                return dy
            elif norm < 1e10:  # scale gradients
                if debug and name is not None:  # track where the gradients became too big.
                    print(name + ' upstream gradient norm (' + str(
                        norm.numpy()) + ') too big. Normalizing gradient...')
                return tf.clip_by_norm(dy, 1e3)
            elif tf.math.is_nan(norm):
                # let this gradient flow...?
                print(
                    name + ' UPSTREAM GRADIENT IS NAN. IGNORING THIS GRADIENT!!!')
                return tf.ones(dy.shape)
            else:  # just cut off gradients (i.e. nan or inf?)
                if debug and name is not None:  # track where the gradients became too big.
                    print(name + ' upstream gradient norm (' + str(
                        norm.numpy()) + ') way too big. Clipping gradient...')
                return tf.clip_by_value(dy, -1e3, 1e3)

        return y, backward


