# Defines dynamical system for optimal control with hand and eye)
# dynamics = x_{t+1} = Ax_t + Bu_t + noise
# feedback = Hx_t + noise

## Dynamics: x_{t + 1} = Ax_t + B[(I + C * rand)(u_t) + C0c * rand] + C0f * rand
# rand is random noise
# additive noise;
        # object dynamic noise (C0f);
        # observer control additive noise (C0c); part of observer noise

## Feedback: y_t = [I + D * rand] H [x_t] + D0 * rand
# additive noise
# multiplicative noise

## State
# X1: state [Tx, Ty, Cx, dCx, Cy, dCy, (7) fCx, fCy]
# position (x,y), velocity (dx,dy), and force (f) of
# cursor (C), target (T) and eye (E)
# Y1: [Tx, Ty, Cx, Cy]

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from ..utils import alt_tile, batch_transpose

# todo: make
class hand(tf.Module):
    def __init__(self, prm,
                 scope = 'dynamics_hand'):
        """

        """
        raise NotImplementedError
        super(hand, self).__init__(name=scope)

        self.config = prm
        self.dtype = prm.dtype

        # vector sizes
        self.target_size = 2 # added for estimation
        self.state_size = 8
        self.meas_size = 4
        self.control_size = 2
        self.control_cursor_idx = [4,5]

        # state dynamics
        self.dt = prm.dt # the unit of control is harder to interpret now.
        self.dtt = self.dt / (prm.tau / 1000) # time constant(muscle) adjusted delay todo: check this
        # self.dtt_eye = self.dt/(prm.tau_eye/1000) # time constant(muscle) adjusted delay todo: check this

        # self.SX0 = tf.zeros((self.state_size,self.state_size)) # initial state uncertainty, no uncertainty in the beginning
        self.X0 = tf.constant(tf.zeros((self.state_size), dtype=self.dtype),
                              dtype=self.dtype,
                              name='X0')  # initialize state vector

        A_target = np.zeros((2,2))
        A_target[0, 0] = 1 # target(x)
        A_target[1, 1] = 1 # target(y)

        A_cursor = np.zeros((6, 6))
        A_cursor[0, 0] = 1
        A_cursor[0, 1] = self.dt
        A_cursor[1, 1] = 1 - self.dt * prm.b / prm.m # cursor(dx + damping)
        A_cursor[1, 4] = self.dt/ prm.m # a = Fdt / mass
        A_cursor[2, 2] = 1
        A_cursor[2, 3] = self.dt
        A_cursor[3, 3] = 1 - self.dt * prm.b / prm.m # cursor(dx + damping)
        A_cursor[3, 5] = self.dt / prm.m  # a = Fdt / mass
        A_cursor[4, 4] = 1 - self.dtt # cursor force(y) + decay by muscle time constant
        A_cursor[5, 5] = 1 - self.dtt # cursor force(y) + decay by muscle time constant

        A = np.block([A_target,A_cursor])

        # control gain
        B = np.zeros((self.state_size, self.control_size)); # can only control force
        B[-2, 0] = self.dtt # time constant adjusted delay
        B[-1, 1] = self.dtt

        # partial sensory evidence
        H = np.zeros((self.meas_size, self.state_size))
        H[0, 0] = 1 # Tx
        H[1, 1] = 1
        H[2, 2] = 1 # Cx
        H[3, 4] = 1

        # dynamics -- additive noise
        # todo: left off here.
        C0f = np.zeros((self.state_size,self.state_size)) # fixed state dynamics noise(target)
        C0f[0,] = self.dt * prm.stimstd
        C0f[0,] = self.dt * prm.stimstd

        self.A = tf.constant(A, dtype=self.dtype, name='A')
        self.B = tf.constant(B, dtype=self.dtype, name='B')
        self.H = tf.constant(H, dtype=self.dtype, name='H')
        self.C0f = tf.constant(C0f, dtype=self.dtype, name='C0f')  # state_size x state_size

        # other noise -- (to be inferred)
        C0c = [prm.c0c_hand, prm.c0c_eye]       # control additive noise.
        C   = [prm.c_hand, prm.c_eye] # control multiplicative noise scaling (2 parameters: cursor and eye)
        D0  = [prm.d0_target, prm.d0_cursor] # additive noise scaling
        D   = [prm.d]  # state multiplicative noise scaling

        if scope is not None:                 # we want to infer noise parameters
            with self.name_scope:
                self.C0c = tf.Variable(C0c, dtype = self.dtype, name='C0c')
                self.C   = tf.Variable(C, dtype = self.dtype, name='C')
                self.D0  = tf.Variable(D0, dtype = self.dtype, name = 'D0')
                self.D   = tf.Variable(D, dtype = self.dtype, name = 'D')
        else:
            self.C0c    = tf.constant(C0c, dtype = self.dtype, name='C0c')
            self.C      = tf.constant(C, dtype = self.dtype, name='C')
            self.D0     = tf.constant(D0, dtype = self.dtype, name='D0')
            self.D      = tf.constant(D, dtype = self.dtype, name='D')

        # penalties
        R = [prm.r, prm.r_eye]
        d = np.zeros((2,self.state_size))
        d[0,0] = 1  # target x position
        d[0,4] = -1 # cursor x position
        d[1, 2] = 1  # target y position
        d[1, 6] = -1  # cursor y position
        Q = np.matmul(d.transpose(),d) # state-size by state-size matrix

        if scope is not None:                 # we want to infer noise parameters
            with self.name_scope:
                self.R = tf.Variable(R, dtype = self.dtype, name = 'R')
        else:
            self.R = tf.constant(R, dtype = self.dtype, name='R')

        self.Q = tf.constant(Q, dtype = self.dtype, name='Q')

        self.nvars = sum([tf.reduce_prod(self.trainable_variables[idx].shape).numpy() for idx in range(len(
                self.trainable_variables))])

        # estimation noise
        self.E0 = tf.constant(prm.e0,dtype=self.dtype,name='E0')

    def __call__(self, xt, ut, target=None):
        ''' propagate dynamics one timestep
        Dynamics: x_{t + 1} = Ax_t + B[(I + C * rand)(u_t) + C0c * rand] + C0f * rand
        Feedback: y_t = [I + D * rand] H [x_t] + D0 * rand

        xt: shape = (batch_size, state_size)
        ut: shape = (batch_size, control_size)
        target: if the target trajectory is fixed, provide the next xt_next;
            shape = [state_size]

        returns: x_{t+1}, y_t and their means.
        '''
        (B, N) = xt.shape

        # sample noise from gaussian distribution
        c0c = tf.random.normal((B, self.control_size), dtype=self.dtype)
        c0f = tf.random.normal((B, self.state_size), dtype=self.dtype)
        if target is not None: # replace target velocities
            c0f[B, 1] = target[1]/self.C0f[1,1]
            c0f[B, 3] = target[3]/self.C0f[1, 1]

        c   = alt_tile(tf.random.normal((B, int(self.control_size /2)), dtype=self.dtype)) # /2, bc same noise source for x/y; shape = B x 2
        d0  = tf.random.normal((B, self.meas_size), dtype=self.dtype)
        d   = alt_tile(tf.random.normal((B, int(self.meas_size/2)), dtype=self.dtype))

        xt_next_mean = self.A @ tf.expand_dims(xt,-1) + \
                       self.B @ tf.eye(self.control_size, batch_shape=[B], dtype=self.dtype) @ tf.expand_dims(ut,-1)
        xt_next = self.A @ tf.expand_dims(xt,-1) + \
                  self.B @ ((tf.eye(self.control_size, batch_shape=[B], dtype=self.dtype) +
                             tf.linalg.diag(alt_tile(self.C) * c)) @ tf.expand_dims(ut,-1) +
                            tf.expand_dims(alt_tile(self.C0c) * c0c,-1)) + \
                  self.C0f @ tf.expand_dims(c0f,-1)
        yt_mean = self.H @ tf.expand_dims(xt,-1)
        yt      = (tf.eye(self.meas_size, batch_shape=[B], dtype=self.dtype) + tf.linalg.diag(alt_tile(self.D,4) * d)) @ \
                  self.H @ tf.expand_dims(xt,-1) + \
                  tf.expand_dims(alt_tile(self.D0,2)*d0,-1)
        # todo: how does D0 work here?

        # squeeze last dimension
        return tf.squeeze(xt_next, axis=-1), tf.squeeze(yt, axis=-1), tf.squeeze(xt_next_mean, axis=-1), tf.squeeze(yt_mean, axis=-1)

    def prob(self,xt,ut):
        '''
        Calculates negative log likelihood p(x_{t+1} | x_t, u_t, theta) from the dynamics equations
        Dynamics: x_{t + 1} = Ax_t + B[(I + C * rand)(u_t) + C0c * rand] + C0f * rand

        Assume xt is T x B X state_size
        '''

        raise NotImplementedError
        (T, B, N) = ut.shape
        xnext = xt[1:,:,:]
        x = xt[0:-1,:,:]
        u = ut[0:-1,:,:]

        mult_noise = self.B @ tf.linalg.diag(alt_tile(self.C)) @ tf.expand_dims(u,-1)

        vars  = self.C0f@tf.transpose(self.C0f) + \
                self.B@tf.linalg.diag(alt_tile(self.C0c))@ tf.transpose(self.B @ tf.linalg.diag(alt_tile(self.C0c))) + \
                mult_noise @ batch_transpose(mult_noise) \
                + tf.eye(self.state_size, batch_shape=[T-1,B], dtype=self.dtype)
        # todo: regularize to make somewhat invertible
        # N x N + T x B X N X N
        means = self.A @ tf.expand_dims(x,-1) + \
                self.B @ tf.eye(self.control_size, batch_shape=[T-1, B], dtype=self.dtype) @ tf.expand_dims(u,-1)

        nloglike = self.state_size/2 * tf.math.log(2 * np.pi)
        nloglike += 0.5 * tf.math.log(tf.linalg.det(vars) + tf.keras.backend.epsilon())
        nloglike += 0.5 * tf.squeeze(batch_transpose(tf.expand_dims(xnext,-1)-means) @ tf.linalg.pinv(vars) @
                                       (tf.expand_dims(xnext,-1)-means))
        try:
            tf.linalg.inv(vars)
        except:
            print('matrix not invertible')

        return tf.reduce_mean(nloglike) # mean over batch and time

    def transpose_mat(self,x):
        return tf.transpose(tf.expand_dims(x, -1), perm=[-1, -2])

    def qaudratic_form(self,x,A):
        return tf.transpose(tf.expand_dims(x, -1), perm=[-1, -2]) @ A @ x

    def plot(self):
        ''' plot dynamics'''
        raise NotImplementedError

    def plot_train_inf(self, cost, params, filename=None):
        # todo: move this to dynamics module???
        axes    = {}
        fig     = plt.figure(figsize=(10, 8))
        gs      = fig.add_gridspec(3, 3)

        # plot cost
        axes[0] = fig.add_subplot(gs[0, :])
        axes[0].plot(np.arange(len(cost)), cost, label='cost')
        axes[0].set_title("Inference cost")
        axes[0].set_xlabel("Iterations")
        axes[0].set_ylabel("Cost (nloglike)")

        # plot model parameters
        axes[1] = fig.add_subplot(gs[1, 0])
        axes[1].plot(np.arange(len(params['R'])), [params['R'][iter][0] for iter in range(len(params['R']))], 'r', label= 'cursor')
        axes[1].plot(np.arange(len(params['R'])), [params['R'][iter][1] for iter in range(len(params['R']))], 'b', label= 'eye')
        axes[1].set_title("R (control cost)")
        axes[1].set_xlabel("Iterations")
        axes[1].set_ylabel("value")

        axes[2] = fig.add_subplot(gs[1, 1])
        axes[2].plot(np.arange(len(params['C0c'])), [params['C0c'][iter][0] for iter in range(len(params['C0c']))], 'r', label= 'cursor')
        axes[2].plot(np.arange(len(params['C0c'])), [params['C0c'][iter][1] for iter in range(len(params['C0c']))], 'b', label= 'eye')
        axes[2].set_title("C0c (additive dynamics noise)")
        axes[2].set_xlabel("Iterations")
        axes[2].set_ylabel("value")

        axes[3] = fig.add_subplot(gs[1, 2])
        axes[3].plot(np.arange(len(params['C'])), [params['C'][iter][0] for iter in range(len(params['C']))], 'r',label= 'cursor')
        axes[3].plot(np.arange(len(params['C'])), [params['C'][iter][1] for iter in range(len(params['C']))], 'b', label= 'eye')
        axes[3].set_title("C (multiplicative dynamics noise)")
        axes[3].set_xlabel("Iterations")
        axes[3].set_ylabel("value")

        axes[4] = fig.add_subplot(gs[2, 0])
        axes[4].plot(np.arange(len(params['D0'])), [params['D0'][iter][0] for iter in range(len(params['D0']))], 'r',label= 'cursor')
        axes[4].plot(np.arange(len(params['D0'])), [params['D0'][iter][1] for iter in range(len(params['D0']))], 'b', label= 'eye')
        axes[4].set_title("D0 (additive feedback noise)")
        axes[4].set_xlabel("Iterations")
        axes[4].set_ylabel("value")

        axes[5] = fig.add_subplot(gs[2, 1])
        axes[5].plot(np.arange(len(params['D'])), [params['D'][iter][0] for iter in range(len(params['D']))], 'b')
        axes[5].set_title("D (multiplicative feedback noise)")
        axes[5].set_xlabel("Iterations")
        axes[5].set_ylabel("value")

        # show and save and close
        plt.show()
        if filename is not None:
            plt.savefig(filename)
        plt.close()

    def plot_sim_obs(self, x, xhat, batchN=1,ploteye = False, filename=None):
        # todo: move this to dynamics module?
        # X1: state [Tx, dTx, Ty, dTy,
        #        (5) Cx, dCx, Cy, dCy, (9) fCx, fCy,
        #       (11) Ex, dEx, Ey, dEy, (15) fEx, fEy
        #       (17) Tx-Ex, Ty - Ey, Cx-Ex, Cy-Ey]
        # xt: simulation
        # x_t: data

        if type(x) is list:
            x = tf.stack(x,axis=0)

        if type(xhat) is list:
            xhat = tf.stack(xhat,axis=0)

        [T,B,S] = x.shape

        axes    = {}
        fig     = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(1, 1)

        # plot behavior
        axes[0] = fig.add_subplot(gs[0, :])
        # model: target, cursor, eye
        # todo: plot more sample trajectories
        for b in range(batchN):
            axes[0].plot(np.arange(T) / self.config.shz, x[:T, b, 0], 'k-',
                         label='target')
            axes[0].plot(np.arange(T) / self.config.shz, x[:T, b, 4],
                         'r', label='output_cursor')

            #axes[0].plot(np.arange(T) / self.config.shz, x[:T, b, 16],
            #             'g:', label='output_target-eye')
            #axes[0].plot(np.arange(T) / self.config.shz, x[:T, b, 18],
            #             'g', label='output_cursor-eye')
            if ploteye:
                axes[0].plot(np.arange(T) / self.config.shz, x[:T, b, 10],
                            'b--', label='output_eye')
            if xhat is not None:
                axes[0].plot(np.arange(T) / self.config.shz, xhat[:T, b, 0],
                             'k:', label='estim_target')
                axes[0].plot(np.arange(T) / self.config.shz, xhat[:T, b, 4],
                             'r:', label='estim_cursor')
            if xhat is not None and ploteye:
                axes[0].plot(np.arange(T) / self.config.shz, xhat[:T, b, 10],
                             'b:', label='estim_eye')
            axes[0].set_title("Model Observer behavior")
            axes[0].set_xlabel("Time (s)")
            axes[0].set_ylabel("Hor. Pos")
            axes[0].legend()

        # show and save and close
        plt.show()
        if filename is not None:
            plt.savefig(filename)
        plt.close()

