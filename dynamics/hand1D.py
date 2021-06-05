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
# X1: state [(0) Tx, dTx
#           (2) Cx, dCx, (4) fCx]
# position (x,y), velocity (dx,dy), and force (f) of
# cursor (C), target (T) and eye (E)
# Y1: [Cx - Ex]

# Ensures that parameters to be optimized are positive through tf.Variable constraint (elu)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class hand1D(tf.Module):
    # init
    # parameter dictionary
    # dictionary to matrix
    # simulate; calls the call step
    # simulate_onetime

    def __init__(self, prm, scope = None, dtype= tf.dtypes.float32):
        super(hand1D, self).__init__(name=scope)
        self.config         = prm
        self.dtype          = dtype

        self.xy = False

        self.target_size    = 2
        self.state_size     = 5
        self.meas_size      = 2 # target and cursor
        self.control_size   = 1
        self.target_idx = [0]  # target position index
        self.cursor_idx = [2]  # cursor position index
        self.target_vel_idx = [1]  # target velocity index
        self.cursor_vel_idx = [3]  # cursor velocity index
        self.control_cursor_idx = [4]
        self.target_noise_idx   = [1]

        self.target_all_idx = [0, 1]
        self.cursor_all_idx = [2,3,4]

        # it is impossible given the dynamics to estimate the current stimulus velocity.
        self.estim_size = self.state_size - len(self.target_vel_idx)  # estimation size

        # time
        self.nframes        = prm.nframes
        self.dt = prm.dt        # the unit of control is harder to interpret now.
        self.ddt = self.dt / (prm.tau / 1000) # time constant(muscle) #
        self.ddt_eye = self.dt/(prm.tau_eye/1000) # time constant(muscle)

        self.build_dyanmics()

    def build_dyanmics(self):
        # X1: state [(0) Tx, dTx
        #           (2) Cx, dCx, (4) fCx
        prm = self.config

        A = np.zeros((self.state_size,self.state_size))
        A[0, 0] = 1         # target(x)
        A[0, 1]= self.dt    # target(x += dx)

        A[2, 2] = 1         # cursor(x)
        A[2, 3] = self.dt   # cursor(x += dx)
        A[3, 3] = 1 - self.dt * prm.b / prm.m # cursor(dx += damping)
        A[3, 4] = self.dt/ prm.m # cursor(dx += ddx); a = Fdt / mass
        A[4, 4] = 1 - self.ddt # cursor (ddx += decay)

        # control gain
        B = np.zeros((self.state_size, self.control_size)) # can only control force
        B[4, 0] = self.ddt # time constant adjusted delay

        # partial sensory evidence
        H = np.zeros((self.meas_size, self.state_size))
        H[0, 0] = 1 # Tx
        H[1, 2] = 1 # Cx

        # dynamics -- additive noise
        C0f         = 1e-7 *np.eye(self.state_size) # fixed state dynamics noise(target)
        C0f[1,1]    = prm.stimstd * np.sqrt(prm.shz) # add to target velocity, note: this is multiplied by dt later

        self.A      = tf.constant(A, dtype=self.dtype, name='A')
        self.B      = tf.constant(B, dtype=self.dtype, name='B')
        self.H      = tf.constant(H, dtype=self.dtype, name='H')
        self.C0f    = tf.constant(C0f, dtype=self.dtype, name='C0f')  # state_size x state_size

        d = np.zeros((1,self.state_size))
        d[0,0] = 1      # target x position
        d[0,2] = -1     # cursor x position
        Q = np.matmul(d.transpose(),d) # state-size by state-size matrix

        self.Q = tf.constant(Q, dtype = self.dtype, name='Q')

        self.X0 = tf.constant(tf.zeros((self.state_size,1), dtype=self.dtype),
                              dtype=self.dtype,
                              name='X0')  # initialize state vector

        #todo: change this part. take better care of deterministic dynamics!!
        # the state is not estimable if it is completely random (stimulus velocity)
        # This is if the given state is not updated by previous state, control,
        # and we have no measurements of it
        contributions = tf.math.reduce_sum(tf.abs(self.A),axis= 1) + \
                        tf.math.reduce_sum(tf.abs(self.B), axis=1) + \
                        tf.math.reduce_sum(tf.abs(self.H), axis=0)
        self.estimable_idx = tf.math.is_finite(contributions) # (contributions != 0)  #tf.math.is_finite(contributions) #(contributions != 0) # tf.math.is_finite(contributions)

        # assuming that the control is stochastic.
        # this index gives sources of noise given z_t (xhat)
        contributions = np.sum(np.abs(C0f),axis=1) + np.sum(np.abs(B),axis=1)
        self.state_stochastic_idx = (contributions != 0)

        return True

    def dynamicsDict(self):
        dict        = {}
        dict['A']   = self.A
        dict['B']   = self.B
        dict['H']   = self.H
        dict['C0f'] = self.C0f
        dict['Q']   = self.Q

        return dict