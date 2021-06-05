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
# X1: state [Tx, dTx, Ty, dTy,
#        (5) Cx, dCx, Cy, dCy, (9) fCx, fCy,
#       (11) Ex, dEx, Ey, dEy, (15) fEx, fEy
#       (17) Tx - Ex, Ty - Ey, Cx - Ex, Cy - Ey]
# position (x,y), velocity (dx,dy), and force (f) of
# cursor (C), target (T) and eye (E)
# Y1: [Tx - Ex, Ty - Ey, Cx - Ex, Cy - Ey]

# Ensures that parameters to be optimized are positive through tf.Variable constraint (elu)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from ..utils import alt_tile, batch_transpose

class handeye(tf.Module):
    def __init__(self, prm, scope = None):
        """

        """
        super(handeye, self).__init__(name=scope)

        self.config = prm
        self.dtype = prm.dtype

        self.multnoise = prm.multnoise
        self.moveeye = prm.moveeye

        self.target_size     = 4
        self.state_size     = 20
        self.X0             = tf.constant(tf.zeros((self.state_size),dtype=self.dtype), dtype=self.dtype, name='X0') # initialize state vector
        self.meas_size      = 4
        self.control_size   = 4

        self.target_idx         = [0, 2]  # target position index
        self.cursor_idx         = [4, 6]  # cursor position index
        self.target_vel_idx     = [1, 3]  # target velocity index
        self.cursor_vel_idx     = [5, 7]  # cursor velocity index

        self.control_cursor_idx = [8,9] # for estimation
        self.target_noise_idx = [1, 3]  # for loglikelihood estimation
        # self.SX0 = tf.zeros((self.state_size,self.state_size)) # initial state uncertainty, no uncertainty in the beginning

        # state dynamics
        self.dt = prm.dt # the unit of control is harder to interpret now.
        self.dtt = self.dt / (prm.tau / 1000) # time constant(muscle) #
        # todo: divide by 1000 bc of ms? or multiply??
        # adjusted delay todo: check this
        self.dtt_eye = self.dt/(prm.tau_eye/1000) # time constant(muscle)
        # adjusted delay todo: check this

        A = np.zeros((self.state_size,self.state_size))
        A[0, 0] = 1 # target(x)
        A[0, 1]= self.dt # note that the velocity doesn't propagate.
        A[2, 2] = 1 # target(y)
        A[2, 3] = self.dt

        A[4, 4] = 1 # cursor(x)
        A[4, 5] = self.dt
        A[5, 5] = 1 - self.dt * prm.b / prm.m # cursor(dx + damping)
        A[5, 8] = self.dt/ prm.m # a = Fdt / mass
        A[6, 6] = 1 # cursor(y)
        A[6, 7] = self.dt
        A[7, 7] = 1 - self.dt * prm.b / prm.m # cusor(dy + damping)
        A[7, 9] = self.dt / prm.m # a = Fdt / mass
        A[8, 8] = 0 #1 - self.dtt # cursor force(x) + decay by muscle time constant todo: should this be 0?
        A[9, 9] = 0 #1 - self.dtt # cursor force(y) + decay by muscle time constant todo: should this be 0?

        A[10, 10] = 1 # eye(x)
        A[10, 11] = self.dt
        A[11, 11] = 1 - self.dt * prm.b / prm.m_eye # eye(dx + damping)
        A[11, 14] = self.dt / prm.m_eye # a = Fdt / mass
        A[12, 12] = 1 # eye(y)
        A[12, 13] = self.dt
        A[13, 13] = 1 - self.dt * prm.b / prm.m_eye # eye(dy + damping)
        A[13, 15] = self.dt / prm.m_eye # a = Fdt / mass
        A[14, 14] = 1 - self.dtt_eye # eye force(x) + decay by muscle time constant
        A[15, 15] = 1 - self.dtt_eye # eye force(y) + decay by muscle time constant

        A[16, 16] = 1 # Tx - Ex
        A[16, 1] = self.dt # dTx
        A[16, 11] = -self.dt # -dEx
        A[17, 17] = 1 # Ty - Ey
        A[17, 3] = self.dt
        A[17, 13] = -self.dt
        A[18, 18] = 1 # Cx - Ex
        A[18, 5] = self.dt # dCx
        A[18, 11] = -self.dt # -dEx
        A[19, 19] = 1 # Cy - Ey
        A[19, 7] = self.dt # dCy
        A[19, 13] = -self.dt # -dEy

        # control gain
        B = np.zeros((self.state_size, self.control_size)); # can only control force
        B[8, 0] = self.dtt # time constant adjusted delay
        B[9, 1] = self.dtt
        B[14, 2] = self.dtt_eye # time constant adjusted delay
        B[15, 3] = self.dtt_eye

        # partial sensory evidence
        H = np.zeros((self.meas_size, self.state_size))
        H[0, 16] = 1 # Tx - Ex
        H[1, 17] = 1
        H[2, 18] = 1 # Cx - Ex
        H[3, 19] = 1

        # dynamics -- additive noise
        C0f = np.zeros((self.state_size,self.state_size)) # fixed state dynamics noise(target)
        C0f[1,1] = prm.stimstd
        C0f[3,3] = prm.stimstd

        self.A = tf.constant(A, dtype=self.dtype, name='A')
        self.B = tf.constant(B, dtype=self.dtype, name='B')
        self.H = tf.constant(H, dtype=self.dtype, name='H')
        self.C0f = tf.constant(C0f, dtype=self.dtype, name='C0f')  # state_size x state_size

        # other noise to be inferred
        self.d0_target = tf.Variable(prm.d0_target, dtype=self.dtype, name='d0_target', constraint=tf.nn.relu)
        self.d0_cursor = tf.Variable(prm.d0_cursor, dtype=self.dtype, name='d0_cursor', constraint=tf.nn.relu)

        self.c0c_hand = tf.Variable(prm.c0c_hand, dtype=self.dtype, name='c0c_hand', constraint=tf.nn.relu)

        if self.moveeye:
            self.c0c_eye = tf.Variable(prm.c0c_eye, dtype=self.dtype, name='c0c_eye', constraint=tf.nn.relu)
        else:
            self.c0c_eye = tf.constant(0, dtype=self.dtype, name='c0c_eye')

        if self.multnoise:
            self.c_hand     = tf.Variable(prm.c_hand, dtype=self.dtype, name='c_hand', constraint=tf.nn.relu)
            self.d          = tf.Variable(prm.d, dtype=self.dtype, name='d', constraint=tf.nn.relu)
            if self.moveeye:
                self.c_eye = tf.Variable(prm.c_eye, dtype=self.dtype, name='c_eye', constraint=tf.nn.relu)
            else:
                self.c_eye = tf.constant(0, dtype=self.dtype, name='c_eye')
        else:
            self.c_hand     = tf.constant(0, dtype=self.dtype, name='c_hand')
            self.d          = tf.constant(0, dtype=self.dtype, name='d')
            self.c_eye      = tf.constant(0, dtype=self.dtype, name='c_eye')

        # penalties
        self.r = tf.Variable(prm.r, dtype=self.dtype, name='r', constraint=tf.nn.relu)
        if self.moveeye:
            self.r_eye = tf.Variable(prm.r_eye, dtype=self.dtype, name='r_eye', constraint=tf.nn.relu)
        else:
            self.r_eye = tf.constant(0, dtype=self.dtype, name='r_eye')

        d = np.zeros((2,self.state_size))
        d[0,0] = 1  # target x position
        d[0,4] = -1 # cursor x position
        d[1, 2] = 1  # target y position
        d[1, 6] = -1  # cursor y position
        Q = np.matmul(d.transpose(),d) # state-size by state-size matrix

        self.Q = tf.constant(Q, dtype = self.dtype, name='Q')

        self.nvars = sum([tf.reduce_prod(self.trainable_variables[idx].shape).numpy() for idx in range(len(
                self.trainable_variables))])

        # estimation noise
        self.E0 = tf.constant(prm.e0,dtype=self.dtype,name='E0')

        # aggregate into other parameters
        self.create_trainable_mat()

    def create_trainable_mat(self):
        # all of the behavior are generated through this mat.
        self.C0c = tf.stack([self.c0c_hand, self.c0c_eye])       # control additive noise.
        self.C   = tf.stack([self.c_hand, self.c_eye]) # control multiplicative noise scaling (2 parameters: cursor and eye)
        self.D0  = tf.stack([self.d0_target, self.d0_cursor]) # additive noise scaling
        self.D   = tf.stack([self.d])  # state multiplicative noise scaling
        self.R = tf.stack([self.r, self.r_eye])

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

    def __str__(self):
        ''' Print the parameters of the model.'''
        strrep = 'c0c_hand = ' + str(self.c0c_hand.numpy())
        strrep += '\nc0c_eye = ' + str(self.c0c_eye.numpy())
        strrep += '\nd0_target = ' + str(self.d0_target.numpy())
        strrep += '\nd0_cursor = ' + str(self.d0_cursor.numpy())
        strrep += '\nr_hand = ' + str(self.r.numpy())
        strrep += '\nr_eye = ' + str(self.r_eye.numpy())
        return strrep

    def tparams2dic(self):
        dict = {}
        dict['c0c_hand']    = self.c0c_hand.numpy()
        dict['d0_target']   = self.d0_target.numpy()
        dict['d0_cursor']   = self.d0_cursor.numpy()
        dict['r']           = self.r.numpy()

        if self.moveeye:
            dict['c0c_eye'] = self.c0c_eye.numpy()
            dict['r_eye']   = self.r_eye.numpy()

        if self.multnoise:
            dict['d'] = self.d.numpy()
            dict['c_hand'] = self.c_hand.numpy()
            if self.moveeye:
                dict['c_eye'] = self.c_eye.numpy()

        return dict

    def update_params(self, **kwargs):
        paramdict = self.tparams2dic()
        for k,v in kwargs.items():
            if k in paramdict:
                (self.__dict__[k]).assign(v)

        self.create_trainable_mat()

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
