import tensorflow as tf
import numpy as np

class observer(tf.Module):
    def __init__(self,
                 N = 1,
                 ca = None, r = None, sa = None,
                 scope=None, dtype = tf.dtypes.float32):
        super(observer, self).__init__(name=scope)

        self.moveeye    = False
        self.multnoise  = False
        self.dtype      = dtype

        self.cparams = self.generateControlParams(ca0 = ca, r = r)
        self.sparamslist = self.generateSensoryParams(salist = sa, N = N)

    def generateControlParams(self, ca0 = None, r= None):
        # "control params" but really all parameters fxied to observer
        cparams = {}

        if ca0 is None:
            ca = tf.random.uniform((1,),minval=0,maxval=0.2,dtype=self.dtype)
        else:
            ca = tf.constant(ca0, shape=(1,),dtype=self.dtype)

        if r is None:
            r = tf.random.uniform((1,),minval=0,maxval=0.005,dtype=self.dtype)
        else:
            r = tf.constant(r, shape=(1,),dtype=self.dtype)

        # additive control noise
        cparams['log_ca_hand'] = \
            tf.Variable(tf.math.log(ca),
                        dtype=self.dtype,
                        name=self.name + '_log_ca_hand')

        # control cost
        cparams['log_r'] =\
            tf.Variable(tf.math.log(r),
                        shape=(1,),
                        dtype=self.dtype,
                        name=self.name + '_log_r')

        # tf.Variable(tf.math.log(),
        #             dtype=self.dtype,
        #             name='log_r', import_scope=self.name_scope)

        # small additive estimation noise (regularization?)
        cparams['log_ea'] = tf.constant(tf.cast(tf.math.log(0.001),self.dtype),
                                        shape = (1,),
                                        dtype=self.dtype,
                                        name=self.name + '_log_ea')

        # fix small cursor std
        cparams['log_sa_cursor'] = \
            tf.constant(tf.cast(tf.math.log(0.001),self.dtype),
                        shape=(1,),
                        dtype=self.dtype,
                        name=self.name + '_log_sa_cursor')

        if self.moveeye:
            raise NotImplementedError
            sparams[n]['log_ca_eye'] = self.c0c_eye.numpy()

        if self.multnoise:
            raise NotImplementedError
            sparams[n]['log_cm_hand'] = self.c_hand.numpy()
            if self.moveeye:
                sparams[n]['log_cm_eye'] = self.c_eye.numpy()

        return cparams


    def generateSensoryParams(self,salist = None, N = 1):
        sparamslist = []
        for n in range(N):
            sparams = {}
            # target uncertainty std
            if salist is None:
                sa = tf.random.uniform((1,),minval=0,maxval=2,
                                       dtype=self.dtype)
            else:
                sa = tf.constant(salist[min(n,len(salist)-1)],
                                 shape=(1,),
                                 dtype=self.dtype)

            sparams['log_sa_target'] = \
                tf.Variable(tf.math.log(sa),
                            dtype=self.dtype,
                            name=self.name + '_log_sa_target' + str(n)+ '_')

            if self.moveeye:
                raise NotImplementedError
                sparamslist[n]['log_r_eye'] = self.r_eye.numpy()

            if self.multnoise:
                raise NotImplementedError
                sparamslist[n]['log_sm'] = self.d.numpy()

            sparamslist += [sparams]


        return sparamslist


    def generateParamMat(self, dyn, n = 0, cparams = None, sparams = None):
        '''
        Takes parameters and converts them to matrix consistent with the dynamics module

        :param cparams:
        :param sparams: sparam for a single stimulus (not a list)
        :param dyn:
        :return:
        '''

        if cparams is None:
            cparams = self.cparams
        if sparams is None:
            sparams = self.sparamslist[n]

        assert cparams is not None and sparams is not None

        paramMats = {}

        # control noise
        paramMats['CM']    = tf.zeros((dyn.control_size,dyn.control_size))
        paramMats['CA']    = tf.linalg.diag(tf.math.exp(cparams['log_ca_hand']))

        # sensory noise
        paramMats['SM']    = tf.zeros((dyn.meas_size,dyn.meas_size))
        paramMats['SA']    = tf.linalg.diag(tf.math.exp(
            tf.concat([sparams['log_sa_target'],
                       cparams['log_sa_cursor']],axis=0)))

        # same estimation noise for cursor and targetr
        paramMats['EA'] = tf.linalg.diag(tf.math.exp(
            tf.repeat(cparams['log_ea'], dyn.state_size)))

        paramMats['R'] = tf.linalg.diag(tf.math.exp(cparams['log_r']))

        if dyn.xy:
            # note that multiplicative XY have same source of noise.
            raise NotImplementedError
            paramMats['C0c']
            paramMats['D0']
            paramMats['R']

        if self.multnoise:
            raise NotImplementedError
            paramMats['C']
            paramMats['D']

        if self.moveeye:
            raise NotImplementedError

        paramMats['A']   = dyn.A
        paramMats['B']   = dyn.B
        paramMats['H']   = dyn.H
        paramMats['Q']   = dyn.Q
        paramMats['C0f'] = dyn.C0f

        return paramMats

    def printVariables(self):
        str = ''
        for var in self.trainable_variables:
            str += '{0:s} = {1:.3e} \n'.format(var.name, var.numpy().item())
        print(str)
