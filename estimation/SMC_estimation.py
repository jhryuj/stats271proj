import tensorflow as tf
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np

from src.sofc.estimation.lqg import LQG
from tensorflow.linalg import matrix_transpose as transpose

# sequential monte carlo
class SMC():
    def __init__(self, debug=False):
        '''
        Initialize EM

        '''

        self.debug = debug
        # self.dtype = dtype

    def calculate_apprx_elbo(self, trainset, dyn, model):
        # run one step
        obs = trainset['obs']
        x   = trainset['x']
        u   = trainset['u']
        NTB = tf.reduce_sum([xn.shape[0]*xn.shape[1] for xn in x])

        # calculate expectations
        expectations = self.e_step(trainset, dyn, model)

        # calculate elbo
        elbo, elbolike, elbotrans = self.m_step(trainset, dyn, model, expectations)
        elbo         = elbo  / tf.cast(NTB,elbo.dtype)

        if self.debug and False:
            plt.figure()
            plt.subplot(2,1,1)
            for n in range(len(elbolike)):
                plt.plot(elbolike[n])
            plt.title('Likelihood elbo')
            #plt.yscale('log')
            plt.xlabel('Timeframes')

            plt.subplot(2, 1, 2)
            for n in range(len(elbolike)):
                plt.plot(elbotrans[n])
            plt.title('Transition elbo')
            #plt.yscale('log')
            plt.xlabel('Timeframes')

            plt.show()

        elbolike_sum = 0
        elbotrans_sum = 0
        for n in range(len(elbolike)):
            elbolike_sum += tf.reduce_sum(elbolike[n])
            elbotrans_sum += tf.reduce_sum(elbotrans[n])

        results = {}
        results['elbo']         = tf.squeeze(elbo).numpy()
        results['elbolike']     = tf.squeeze((elbolike_sum / tf.cast(NTB,elbo.dtype))).numpy()
        results['elbotrans']    = tf.squeeze((elbotrans_sum / tf.cast(NTB,elbo.dtype))).numpy()
        results['NTB']          = tf.cast(NTB,elbo.dtype)
        results['expectations'] = expectations
        results['elbolike_all']  = elbolike
        results['elbotrans_all'] = elbotrans

        return results
