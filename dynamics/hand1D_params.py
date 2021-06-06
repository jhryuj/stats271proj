import tensorflow as tf

class hand1D_params():
    def __init__(self):
        '''
        Initialize default parameters
        '''
        ######################## experimenter parameters ########################
        self.shz     = 30                       # sampling rate
        self.tsec    = 15                        # seconds (experiment or simulation)
        self.nframes = self.tsec*self.shz       # duration in number of time steps
        self.Nsim    = 50                       # number of simulations
        self.dt      = 1/self.shz               # time steps (sec)
        self.stimstd = 1                        # stimulus std

        ######################## observer parameters ########################
        ## fixed parameters
        self.m       = 0.7          # mass (hand) (kg)
        self.m_eye   = 0.7          # mass (eye) (kg)
        self.b       = 0            # damping (N/sec)
        self.tau     = 40           # time constant (hand) (msec) todo: fix this
        self.tau_eye = 16.6         # time constant eye (msec) todo: fix this
        self.delay_sec = 0.12  # perception delay in frames seconds
        self.delay   = tf.floor(self.delay_sec * self.shz)

    def update_params(self,**kwargs):
        allowed_keys = self.__dict__
        self.__dict__.update(
            (k, v) for k, v in kwargs.items() if k in allowed_keys)

        self.nframes    = self.tsec*self.shz            # duration in number of time steps
        self.dt         = 1/self.shz         # time steps (sec)
        self.delay      = tf.floor(self.delay_sec * self.shz)