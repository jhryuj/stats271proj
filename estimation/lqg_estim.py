import copy, os
import tensorflow as tf
import numpy as np

import pickle
import matplotlib.pyplot as plt
from tqdm import trange

from src.sofc.dynamics.hand1D import hand1D
from src.sofc.dynamics.hand1D_params import hand1D_params
from src.sofc.estimation.observer import observer
from src.sofc.estimation.lqg import LQG
from src.sofc.estimation.EM_estimation import EM_estimation

DEBUG = True # run assertion checks and plot some stuff during run

# dynamics to be used
alldtypes   = tf.dtypes.float32
dyn         = hand1D(hand1D_params(),dtype=alldtypes)
lqg         = LQG(dyn, clip_gradients = False, debug = DEBUG)
conds       = 1 # number of conditions (sensory parameters to generate)
batchsize   = 10
Niter       = 2

runEM = EM_estimation(debug=DEBUG)

for true_sa in [0.25,0.5,0.75,1,1.5,2]:
    ############################ check if the thing works first ############################
    # generate training set
    trainset = {}
    trainset['obs'] = observer(N=conds,
                               ca=0.1, r=0.001, sa=[true_sa],
                               scope='observer', dtype=alldtypes)
    print('Data (true) parameters:')
    trainset['obs'].printVariables()
    x, xhat, u, tt = \
        lqg.simulate(dyn, trainset['obs'], batch_size=batchsize)
    x, xhat, u, tt = \
        lqg.simulate(dyn, trainset['obs'], x_fix=[x[0]], batch_size=batchsize)
    trainset['x'] = x
    trainset['xhat'] = xhat
    trainset['u'] = u

    # Check elbo of the data and true parameters
    max_res = runEM.calculate_apprx_elbo(trainset, dyn, trainset['obs'])
    max_exp = max_res['expectations']
    max_elbo = max_res['elbo']
    max_logpz = runEM.prob_latent(max_res['expectations'], xhat, dyn)

    elbos   = []
    params  = []
    logpz   = []

    # plot data
    if DEBUG and False:
        print('show the training set data')
        plt.figure(figsize=(5, 4));
        plt.subplot(2, 1, 1)
        plt.title('Estimation (Kalman filter)')
        plt.plot(x[0][:, 1, 0], 'k', label='target')
        for n in range(conds):
            plt.plot(xhat[n][:, 1, 0],
                     alpha=0.7, linewidth=2, label='estimation' + str(n))
        plt.legend()
        plt.xlim((0, 150))

        plt.subplot(2, 1, 2)
        plt.title('Control (Control filter)')
        plt.plot(x[0][:, 1, 0], 'k', label='target')
        for n in range(conds):
            plt.plot(x[n][:, 1, 2],
                     alpha=0.7, label='control' + str(n))
        plt.legend()
        plt.xlim((0, 150))
        plt.show()

        # plt.figure()
        # plt.plot(xhat[0][:,1,0,0],'k',label = 'estim_tp')
        # plt.plot(xhat[0][:,1,1,0],'r',label = 'estim_tv')
        # plt.plot(xhat[0][:,1,2,0],'y',label = 'estim_cp')
        # plt.plot(xhat[0][:,1,3,0],'b',label = 'estim_cv')
        # plt.plot(xhat[0][:,1,4,0],'g',label = 'estim_ca')
        # plt.legend()
        # plt.show()

    for iter in trange(50):
        # build a model observer
        model = observer(N = conds,
                         ca=0.1, r=0.001, sa=None,
                         scope = 'model', dtype = alldtypes)
        # print('model parameters:')
        # model.printVariables()

        # check that the filters are different
        obsparams = trainset['obs'].generateParamMat(dyn, n=0)
        modelparams = model.generateParamMat(dyn, n=0)
        obs_K, obs_L = lqg.calculateFilters(obsparams, dyn)
        mod_K, mod_L = lqg.calculateFilters(modelparams, dyn)

        # plot filters
        if DEBUG and False:
            check00 = tf.norm(obs_L[:, 0, 0] - mod_L[:,0,0])
            plt.figure(figsize=(15, 12))
            plt.subplot(2,1,1)
            plt.plot(obs_K[:, 0, 0], 'r', label='observer K')
            plt.plot(mod_K[:,0,0],'b', label='model K')
            plt.title('Kalman filter weights (position)')
            plt.legend()

            plt.subplot(2,1,2)
            plt.plot(obs_L[:, 0, 0], 'r', label='observer L')
            plt.plot(mod_L[:,0,0],'b', label='model L')
            plt.title('Control filter weights (position)')
            plt.legend()
            plt.show()

        # Check elbo of the data and true parameters
        rand_res    = runEM.calculate_apprx_elbo(trainset, dyn, model)
        rand_exp    = rand_res['expectations']
        rand_elbo   = rand_res['elbo']
        rand_logpz = runEM.prob_latent(rand_res['expectations'], xhat, dyn)

        for v in model.trainable_variables:
            if 'log_sa' in v.name:
                sa = tf.math.exp(v).numpy().item()

        elbos   += [rand_elbo]
        params  += [sa]
        logpz   += [rand_logpz]

        # print('Elbo with true parameters: {0:.3e}'.format(max_res['elbo']))
        # print('Elbo with random model: {0:.3e}'.format(rand_res['elbo']))
        # print('log(p(z|x)) with true parameters: {0:.3e}'.format(max_logpz))
        # print('log(p(z|x)) with true parameters: {0:.3e}'.format(rand_logpz))
        #
        # print('Is true latent state more likely under the true model:')
        # print(max_logpz >= rand_logpz)
        #
        # print('Is true elbo higher than random elbo: ')
        print(max_res['elbo'] >= rand_res['elbo'])
        # print(max_res['elbolike'] >= rand_res['elbolike']) # basically true all the time
        # print(max_res['elbotrans'] >= rand_res['elbotrans'])

        # plot elbos over time
        if DEBUG and False:
            plt.figure(figsize=(15, 8))
            plt.subplot(2,1,1)
            for n in range(conds):
                plt.plot(max_res['elbolike_all'][n]/batchsize,
                         linewidth = 1, alpha =0.5, color='r',label='true model')
                plt.plot(rand_res['elbolike_all'][n]/batchsize,
                         linewidth = 1, alpha=0.5, color='b', label='random model')
            plt.title('Likelihood elbo')
            #plt.yscale('log')
            plt.xlabel('Timeframes')

            plt.subplot(2, 1, 2)
            for n in range(conds):
                plt.plot(max_res['elbotrans_all'][n]/batchsize,
                         linewidth=1, alpha=0.5,color = 'r',label='true model')
                plt.plot(rand_res['elbotrans_all'][n]/batchsize,
                         linewidth=1, alpha=0.5,color = 'b', label='random model')
            plt.title('Transition elbo')
            #plt.yscale('log')
            plt.xlabel('Timeframes')
            plt.legend()

            plt.show()

        # plot expectations in state space and control space
        if DEBUG and False:
            check00 = tf.norm(max_exp['mu_tT'][0] - rand_exp['mu_tT'][0])
            check01 = tf.norm(max_exp['Sigma_tT'][0] - rand_exp['Sigma_tT'][0])

            # state space
            plt.figure(figsize=(15, 8))
            plt.subplot(2,2,1)
            d=0; dmu=0;
            plt.plot(x[0][:, 0, d, 0], 'g:', label='cursor')
            plt.plot(xhat[0][:, 0, d, 0], 'k:', label='cursor_estim')
            plt.plot(max_exp['mu_tT'][0][:, 0, dmu, 0], 'r:',
                     linewidth=1, label='true model mu_tT', alpha=0.4)
            var = max_exp['Sigma_tT'][0][:, dmu, dmu]  # variance of the mean muz for target
            plt.fill_between(tf.range(max_exp['mu_tT'][0].shape[0]),
                             max_exp['mu_tT'][0][:, 0, dmu, 0] - 3 * tf.math.sqrt(var),
                             max_exp['mu_tT'][0][:, 0, dmu, 0] + 3 * tf.math.sqrt(var),
                             alpha=0.2, color='r', label='true model std')
            plt.plot(rand_exp['mu_tT'][0][:, 0, dmu, 0], 'b:',
                     linewidth=1, label='random model mu_tT', alpha=0.4)
            var = rand_exp['Sigma_tT'][0][:, dmu, dmu]  # variance of the mean muz for target
            plt.fill_between(tf.range(rand_exp['mu_tT'][0].shape[0]),
                             rand_exp['mu_tT'][0][:, 0, dmu, 0] - 3 * tf.math.sqrt(var),
                             rand_exp['mu_tT'][0][:, 0, dmu, 0] + 3 * tf.math.sqrt(var),
                             alpha=0.2, color='b', label='random model std')
            plt.legend()
            plt.xlim([20, 100])

            plt.subplot(2, 2, 2)
            plt.title('Cursor position vs mu')
            d=2; dmu=1;
            plt.plot(x[0][:, 0, d, 0], 'g:', label='cursor')
            plt.plot(xhat[0][:, 0, d, 0], 'k:', label='cursor_estim')
            plt.plot(max_exp['mu_tT'][0][:, 0, dmu, 0], 'r:',
                     linewidth=1, label='true model mu_tT', alpha=0.4)
            var = max_exp['Sigma_tT'][0][:, dmu, dmu]  # variance of the mean muz for target
            plt.fill_between(tf.range(max_exp['mu_tT'][0].shape[0]),
                             max_exp['mu_tT'][0][:, 0, dmu, 0] - 3 * tf.math.sqrt(var),
                             max_exp['mu_tT'][0][:, 0, dmu, 0] + 3 * tf.math.sqrt(var),
                             alpha=0.2, color='r', label='true model std')
            plt.plot(rand_exp['mu_tT'][0][:, 0, dmu, 0], 'b:',
                     linewidth=1, label='random model mu_tT', alpha=0.4)
            var = rand_exp['Sigma_tT'][0][:, dmu, dmu]  # variance of the mean muz for target
            plt.fill_between(tf.range(rand_exp['mu_tT'][0].shape[0]),
                             rand_exp['mu_tT'][0][:, 0, dmu, 0] - 3 * tf.math.sqrt(var),
                             rand_exp['mu_tT'][0][:, 0, dmu, 0] + 3 * tf.math.sqrt(var),
                             alpha=0.2, color='b', label='random model std')
            plt.legend()
            plt.xlim([20, 100])

            plt.subplot(2, 2, 3)
            plt.title('Cursor velocity vs mu')
            d=3; dmu=2;
            plt.plot(x[0][:, 0, d, 0], 'g:', label='cursor')
            plt.plot(xhat[0][:, 0, d, 0], 'k:', label='cursor_estim')
            plt.plot(max_exp['mu_tT'][0][:, 0, dmu, 0], 'r:',
                     linewidth=1, label='true model mu_tT', alpha=0.4)
            var = max_exp['Sigma_tT'][0][:, dmu, dmu]  # variance of the mean muz for target
            plt.fill_between(tf.range(max_exp['mu_tT'][0].shape[0]),
                             max_exp['mu_tT'][0][:, 0, dmu, 0] - 3 * tf.math.sqrt(var),
                             max_exp['mu_tT'][0][:, 0, dmu, 0] + 3 * tf.math.sqrt(var),
                             alpha=0.2, color='r', label='true model std')
            plt.plot(rand_exp['mu_tT'][0][:, 0, dmu, 0], 'b:',
                     linewidth=1, label='random model mu_tT', alpha=0.4)
            var = rand_exp['Sigma_tT'][0][:, dmu, dmu]  # variance of the mean muz for target
            plt.fill_between(tf.range(rand_exp['mu_tT'][0].shape[0]),
                             rand_exp['mu_tT'][0][:, 0, dmu, 0] - 3 * tf.math.sqrt(var),
                             rand_exp['mu_tT'][0][:, 0, dmu, 0] + 3 * tf.math.sqrt(var),
                             alpha=0.2, color='b', label='random model std')
            plt.legend()
            plt.xlim([20, 100])

            #todo: units on the acceleration are completely wrong!!
            plt.subplot(2, 2, 4)
            plt.title('Cursor acceleration vs mu')
            d=4; dmu=3;
            thismu = max_exp['mu_tT'][0][:, 0, dmu, 0]
            #thismu = thismu/tf.reduce_max(thismu)
            thisx = x[0][:, 0, d, 0]
            #thisx = thisx/tf.reduce_max(thisx)
            thisz = xhat[0][:, 0, d, 0]
            #thisz = thisz / tf.reduce_max(thisz)
            plt.plot(thisx, 'g:', label='cursor')
            plt.plot(thisz, 'k:', label='cursor_estim')
            plt.plot(thismu, 'r:',
                     linewidth=1, label='true model mu_tT', alpha=0.4)
            var = max_exp['Sigma_tT'][0][:, dmu, dmu]  # variance of the mean muz for target
            plt.fill_between(tf.range(max_exp['mu_tT'][0].shape[0]),
                             thismu - 3 * tf.math.sqrt(var),
                             thismu + 3 * tf.math.sqrt(var),
                             alpha=0.2, color='r', label='true model std')

            thismu = rand_exp['mu_tT'][0][:, 0, dmu, 0]
            #thismu = thismu/tf.reduce_max(thismu)

            plt.plot(thismu, 'b:',
                     linewidth=1, label='random model mu_tT', alpha=0.4)
            var = rand_exp['Sigma_tT'][0][:, dmu, dmu]  # variance of the mean muz for target
            plt.fill_between(tf.range(rand_exp['mu_tT'][0].shape[0]),
                             thismu - 3 * tf.math.sqrt(var),
                             thismu + 3 * tf.math.sqrt(var),
                             alpha=0.2, color='b', label='random model std')
            plt.legend()
            plt.xlim([20, 100])
            plt.show()

            ## control space
            plt.figure(figsize=(15, 8))
            plt.title('Control force')
            # plt.subplot(2,2,1)
            plt.plot(u[0][:, 0, 0, 0], 'k', label='exerted force')
            mumean = tf.squeeze(-1 * obs_L @ xhat[0][:, 0, :, :])
            plt.plot(mumean, 'k:', label='Lt * xhat')
            plt.fill_between(tf.range(u[0].shape[0]),
                             mumean- 3 * obsparams['CA'][0,0],
                             mumean + 3 * obsparams['CA'][0,0],
                             alpha=0.2, color='k', label='Lt * xhat 3 std')

            ut_mutT = tf.squeeze(-1 * tf.gather(obs_L[:-1,:,:],[0,2,3,4],axis=2) @
                                 max_exp['mu_tT'][0][:, 0, :, :])
            plt.plot(ut_mutT, 'r:', label='Lt * mu_tT')
            covar = tf.gather(obs_L[:-1,:,:],[0,2,3,4],axis=2) @ max_exp['Sigma_tT'][0] @ \
                  tf.linalg.matrix_transpose(tf.gather(obs_L[:-1,:,:],[0,2,3,4],axis=2))  # variance of the mean muz for target
            var = covar[:,0,0] # + obsparams['CA'][0,0]
            plt.fill_between(tf.range(ut_mutT.shape[0]),
                             ut_mutT - 3 * tf.math.sqrt(var),
                             ut_mutT + 3 * tf.math.sqrt(var),
                             alpha=0.2, color='r', label='Lt * mu_tT 3 std')

            ut_mutT = tf.squeeze(-1 * tf.gather(mod_L[:-1,:,:],[0,2,3,4],axis=2) @
                                 rand_exp['mu_tT'][0][:, 0, :, :])
            plt.plot(ut_mutT, 'b:', label='random Lt * mu_tT')
            covar = tf.gather(mod_L[:-1,:,:],[0,2,3,4],axis=2) @ \
                    rand_exp['Sigma_tT'][0] @ \
                  tf.linalg.matrix_transpose(tf.gather(mod_L[:-1,:,:],[0,2,3,4],axis=2))  # variance of the mean muz for target
            var = covar[:,0,0] + obsparams['CA'][0,0]
            plt.fill_between(tf.range(ut_mutT.shape[0]),
                             ut_mutT - 3 * tf.math.sqrt(var),
                             ut_mutT + 3 * tf.math.sqrt(var),
                             alpha=0.2, color='b', label='random Lt * mu_tT 3 std')

            plt.legend()
            plt.xlim([20, 40])
            plt.show()

    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Elbos')
    plt.scatter(params, elbos, 20,'b', label='Random parameters')
    plt.axvline(x = true_sa, color='r', label='True parameter')
    plt.axhline(y = max_elbo, color='r')
    plt.xlabel('Sensory uncertainty')
    plt.ylabel('Elbo')
    range = np.quantile(elbos,0.90) - np.quantile(elbos,0.1)
    plt.ylim([np.quantile(elbos,0.1) - range/10,
              np.quantile(elbos,0.90)+range/10])

    plt.subplot(1,2,2)
    plt.title('log prob of true perception')
    plt.scatter(params, logpz, 20,'b', label='Random parameters')
    plt.axvline(x = true_sa, color='r', label='True parameter')
    plt.axhline(y = max_logpz, color='r')
    plt.xlabel('Sensory uncertainty')
    plt.ylabel('log prob of true perception')
    plt.ylim([np.quantile(logpz, 0.1) - range / 10,
              np.quantile(logpz, 0.90) + range / 10])
    plt.legend()

    plt.show()

############################ train some models ############################
# trainset = {}
# trainset['obs'] = observer(N=conds,
#                            ca=0.1, r=0.001, sa=[0.5, 1, 2],
#                            scope='observer', dtype=alldtypes)
# x, xhat, u, tt = \
#     lqg.simulate(dyn, trainset['obs'], batch_size=batchsize)
# x, xhat, u, tt = \
#     lqg.simulate(dyn, trainset['obs'], x_fix=[x[0]], batch_size=batchsize)
# trainset['x'] = x
# trainset['xhat'] = xhat
# trainset['u'] = u
#
# allmodels = []
# allelbos   = []
#
# print('True parameters: ')
# trainset['obs'].printVariables()
#
# max_res = runEM.calculate_apprx_elbo(trainset, dyn, trainset['obs'])
# max_exp = max_res['expectations']
# max_elbo = max_res['elbo']
# max_logpz = runEM.prob_latent(max_res['expectations'], xhat, dyn)
#
# Niters_mstep = 3
# for iter in range(Niter):
#     # build a model observer
#     model = observer(N=conds,ca=0.1, r=0.001, sa=None,
#                      scope='model', dtype=alldtypes)
#     print('Model initial parameters: ')
#     model.printVariables()
#
#     # run EM Algorithm
#     elbos = runEM(trainset, dyn, model, lr=0.1,
#                   Niters=5, Niters_mstep=Niters_mstep)
#
#     # plot elbos
#     plt.figure(figsize=(15, 8))
#     plt.title('Elbo')
#     plt.plot(np.arange(len(elbos)) / Niters_mstep, elbos,
#              'b', label= 'trained model')
#     plt.plot(np.arange(len(elbos)) / Niters_mstep, [max_elbo] * len(elbos),
#              'r', label='true model')
#     plt.xlabel('Iterations')
#     plt.ylabel('Elbo per timepoint')
#     plt.legend()
#     plt.show()
#
#     print('Recovered parameters: ')
#     model.printVariables()
#
#     allmodels += [model]
#     allelbos += [elbos]
#
#
# nparams         =  len(trainset['obs'].trainable_variables)
# keys_obs        = [v.name for v in trainset['obs'].trainable_variables]
# vals_obs        = [v.numpy().item() for v in trainset['obs'].trainable_variables]
# paramvals       = [[v.numpy().item() for v in m.trainable_variables] for m in allmodels]
# for p in range(nparams):
#     print(keys_obs[p][9:-2])
#     trueparam           = vals_obs[p]
#     recovered_params    = [x[p] for x in paramvals]
#     recovered_mean = np.mean(recovered_params)
#     recovered_std = np.std(recovered_params)
#     print('True: {0:.3e}; Recovered mean: {1:.3e}; Recovered std: {2:.3e}'.format(
#         trueparam,recovered_mean,recovered_std))
#
# plt.figure(figsize=(15, 8))
# plt.title('Sensory uncertainty')
# for p in range(3,nparams):
#     trueparam           = vals_obs[p]
#     recovered_params    = [x[p] for x in paramvals]
#     plt.scatter(trueparam, trueparam, 20, 'r',
#             label='true')
#     plt.scatter([trueparam] * len(recovered_params), recovered_params, 20, 'b',
#             label='recovered')
# plt.xlabel('True parameter value')
# plt.ylabel('Recovered parameter value')
# plt.legend()
# plt.show()
#
# print('done.\n')



