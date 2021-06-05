import tensorflow as tf

#todo: string representation
class handeye_params():
    def __init__(self):
        '''
        Initialize default parameters
        '''
        self.dtype = tf.dtypes.float32

        ######################## experimenter parameters ########################
        self.shz     = 20
        self.tsec    = 20            # seconds (experiment or simulation)
        self.nframes = self.tsec*self.shz            # duration in number of time steps
        self.Nsim    = 50               # number of simulations
        self.dt      = 1/self.shz         # time steps (sec)
        self.stimstd = 1             # stimulus

        ######################## observer parameters ########################
        ## fixed parameters
        self.m       = 0.7          # mass (hand) (kg)
        self.m_eye   = 0.7          # mass (eye) (kg)
        self.b       = 0            # damping (N/sec)
        self.tau     = 40           # time constant (hand) (msec) todo: fix this
        self.tau_eye = 16.6         # time constant eye (msec) todo: fix this
        self.delay   = 10           # perception delay in frames (todo)

        ## adjustable parameters
        self.multnoise  = False      # add multiplicative noise as a learnable parameter
        self.moveeye    = False  # add multiplicative noise as a learnable parameter

        self.c0c_hand    = 0.5        # control additive noise;  todo: change this back
        self.c0c_eye     = 0          # control additive noise;  todo: change this back
        self.c_hand      = 0          # control-dependent noise
        self.c_eye       = 0          # control-dependent noise (eye)
        self.d0_target   = 1          # state additive noise; sensory uncertainty (target)
        self.d0_cursor   = 0.01       # state additive noise; sensory uncertainty (cursor)
        self.d           = 0          # state-dependent noise (retinotopic position dependent uncertainty)
        self.e0          = 0          # estimation noise

        self.r           = 0.1        # control signal penalty (hand)
        self.r_eye       = 0         # control penalty (eye) #todo: change back to 0.001 ish?
        #v = 0.2          # velocity penalty

    def __call__(self):
        # return dictionary with learnable parameters
        dict = {}
        dict['c0c_hand'] = self.c0c_hand
        dict['c0c_eye'] = self.c0c_eye
        dict['d0_target'] = self.d0_target
        dict['d0_cursor'] = self.d0_cursor
        dict['r'] = self.r
        dict['r_eye'] = self.r_eye

        return dict
        # raise NotImplementedError

    def __str__(self):
        strrep =  'nframes = ' + str(self.nframes)
        strrep += '\nc0c_hand = ' + str(self.c0c_hand)
        strrep += '\nd0_target = ' + str(self.d0_target)
        strrep += '\nd0_cursor = ' + str(self.d0_cursor)
        strrep += '\nr = ' + str(self.r)
        strrep += '\nc0c_eye = ' + str(self.c0c_eye)
        strrep += '\nr_eye = ' + str(self.r_eye)
        return strrep
        raise NotImplementedError

    def update_params(self,**kwargs):
        allowed_keys = self.__dict__
        self.__dict__.update(
            (k, v) for k, v in kwargs.items() if k in allowed_keys)

        self.nframes = self.tsec*self.shz            # duration in number of time steps
        self.dt      = 1/self.shz         # time steps (sec)

        #    for key, value in kwargs.items():
        #      setattr(self, key, value)

