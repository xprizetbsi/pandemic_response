import sys
import os
from traceback import print_exc
from pprint import pprint
from common.utils import read_params, get_data
from common.stats import RSS, MSPE, RMSE
from common.linalg import as_array, as_matrix, init_weights
from common.config import data_type
from time import time
from numpy import cov, tile, std, average, mean, eye, ones, corrcoef, inf
from numpy.random import choice, multivariate_normal, normal, uniform
from numpy import *
from scipy.stats import norm


class ParticleFilter(object):

    def __init__(self, num_part, params={}):
        self.num_part = num_part
        self.weights = init_weights(num_part)
        
        self.x_prior = None
        
    def step(self, y, predict_P=False, index=1):
        X = self.x_prior
        states = X[:2, :]
        params = X[2:, :]
        print('A1',X[index].A1.shape)
        s_std = std(X[index].A1) 
        
        tmp_ws = as_array([norm.pdf(y, x[0, index], s_std) for x in states.T])
        n_weights = self.weights * tmp_ws
        sum_weights = n_weights.sum()
        if sum_weights != 0:
            n_weights /= sum_weights
            neff = 1.0 / (n_weights ** 2).sum() 
        
        if sum_weights == 0 or neff < self.num_part/2.0 : 
            print('index',index,'resamples')
            idx = choice(range(X.shape[1]), X.shape[1], p=self.weights)
            self.weights = tile(as_array(1.0 / self.num_part), self.num_part)
            self.x_post = X[:, idx]
            
        else:
            self.x_post = X
            self.weights = n_weights

        print('weight:',self.weights.shape, 'n_weight',n_weights.sum(), 'curr weight',self.weights.sum())
        p_mean = average(params, axis=1, weights=self.weights).A1
        p_cov = cov(params, aweights=self.weights)
        self.x_post[2:, :] = multivariate_normal(p_mean, p_cov, X.shape[1]).T
 
        for i, x in enumerate(self.x_post[2:, :].T): #iterate over all parameter post estimat.
            # resample any particle parameters (alpha,beta) 
            if x.any() < 0:
                while True:
                    new = multivariate_normal(p_mean, p_cov, 1).T
                    if new.all() > 0  and new[0, 1] > new[0, 2]:
                        self.x_post[2:, i] = new
                        break
        
    def fit(self, x):
        self.x_prior = x 
        



class BaseSEIR(object):
    '''The abstract base for the SIR model and its variant models'''
    def __init__(self, params):
        init_i = float(params.get('init_i', 0.0))
        # get init_i as float from the params, if it does not exist bring 0.0 as it's value
        
        self.set_init_i(init_i)
        #Setup
            #Initial i as init_i
            #Initial e as (1-i)*gamma 
            #Initial s as 1 - (i+e)
            
            #Set self.Is value as i
            #Set self.Ss value as s
            
        
        self.epoch = 0
        #Start the epoch as 0
        
        self.epochs = params.get('epochs', 52)
        #Get the epochs from the parameters, with there is no value epochs = 52
        
        print('initialize BaseSIR')
        self.fit(params, True)
        #Pass
        
        
        
    def check_bounds(self, x, low_bnd=0, up_bnd=1):
        """Check bounds is used to check if a value x is between 0 and 1, 
        if it is greater or smaller, defines it's value as the upper or
        lower limit"""
        if x < low_bnd: 
            x = 0.0
        elif x > up_bnd: 
            x = 1.0
        return x

    def fit(self, params, refit=False):
        pass

    def set_init_i(self, i, s=inf):
        self.i = float(i)
        iinv = 1 - self.i
        self.e = iinv*params['gamma']
        self.s = 1 - (i + self.e) if float(s) is inf else i

        self.Is = [self.i]
        self.Es = [self.e]
        self.Ss = [self.s]


class SIR(BaseSIR):
    '''The SIR model'''
    def __init__(self, params):
        super(SIR, self).__init__(params)
        self.CDC_obs = get_data(self.CDC)
        print('initialize SIR')
        
    def fit(self, params, refit=True):
        print('call SIR.fit')
        if not refit:
            self.filter_type = params.get('filter_type', '')
            assert 'beta' in params, 'The paramter beta is missing.'
            self.beta = float(params['beta'])

            assert 'alpha' in params, 'The paramter alpha is missing.'
            self.alpha = float(params['alpha'])

            assert 'CDC' in params, 'The paramter CDC is missing.'
            self.CDC = params['CDC']

            self.filter = None
        if refit:
            self.filter_type = params.get('filter_type', '')
            self.beta = float(params.get('beta', 0)) or self.beta
            self.alpha = float(params.get('alpha', 0)) or self.alpha
            print('alpha:',self.alpha)
            self.CDC = params.get('CDC') or self.CDC

            self.epochs = params.get('epochs', 52)
            self.epoch = params.get('epoch', 0)
            
        print('n epochs',self.epochs,'epoch',self.epoch,'refit',refit)
        self.init_i = float(params.get('init_i', self.i))
        self.set_init_i(self.init_i)
        self.score = 0.0

        self.filtering = params.get('filtering', False)
        if self.filtering: 
            self._init_filter()

        return self

    def predict(self):
        while self.epoch < self.epochs - 1:
            self.update_states()    
            self.epoch += 1
        self.get_score()

    def update_states(self):
        self.s += self._delta_s()
        self.i += self._delta_i()

        self.s = self.check_bounds(self.s)
        self.i = self.check_bounds(self.i)
        """We will need to add an Self.Es here... study the best way.."""
        
        self.Is.append(self.i)
        self.Ss.append(self.s)

    def predict_with_filter(self,params):
        if not self.filtering or not self.filter:
            raise Exception('The filtering flag must be set True, \
                             and the filter needs to be inialized')

        F = self.filter 
        while self.epoch < self.epochs - 1:
            x = as_matrix([self.s, self.i]).T
            F.fit(x,params,refit=True)
            y = as_matrix([self.CDC_obs[self.epoch]]).T
            F.step(y)

            self.s = self.check_bounds(F.x_post[0, 0])
            self.i = self.check_bounds(F.x_post[1, 0])
            self.update_states()
            self.epoch += 1
            
        self.get_score()
        self.score 
                    
    def _delta_s(self):
        return - self.alpha * self.s * self.i 

    """Insert def _delta_e(self): #here"""
    
    def _delta_i(self):
        return self.alpha * self.s * self.i - self.beta * self.i
    

    def get_score(self):
        self.outcome = [x for _, x in enumerate(self.Is)]
        self.scores = {}
        self.scores['SSE'] = RSS(self.outcome, self.CDC_obs, 1)
        self.scores['RMSE'] = RMSE(self.CDC_obs, self.outcome, 1)
        self.scores['MSPE'] = MSPE(self.CDC_obs, self.outcome, 1)
        self.scores['CORR'] = corrcoef(self.CDC_obs, self.outcome, 1)

    def _init_filter(self):
        num_states = 2
        num_obs = 1
        A = as_matrix([[1, -self.alpha], 
                       [0, 1 + self.alpha - self.beta]])

        B = as_matrix([0, 1])

        Cov = eye(num_states, dtype=data_type) * 0.0001

        V = Cov.copy()
        W = eye(num_obs, dtype=data_type) * 0.0001
        
        #self.filter = KalmanFilter(num_states, num_obs, A, B, V, W, Cov)
    
    def construct_B(self, with_param=False):
        B = as_matrix([0, 1, 0, 0]) if with_param else as_matrx([0, 1])
        return B

class ParticleSIR(SIR):
    
    def __init__(self, num_enbs, params,alphas=[],betas=[]):
        self.num_enbs = num_enbs
        super(ParticleSIR, self).__init__(params) 
        del self.alpha
        del self.beta
        
        self.current_Is = uniform(0, self.i * 2, num_enbs)
        self.current_Ss = ones(num_enbs) - self.current_Is
        self.alphas = uniform(0., 1, num_enbs)
        self.betas = uniform(0., 1, num_enbs)

        self.weights = [init_weights(num_enbs)] # matrix-like

        for i in range(num_enbs):
            if self.alphas[i] < self.betas[i]:
                self.alphas[i], self.betas[i] = self.betas[i], self.alphas[i]  

        self.Is = [self.current_Is.tolist()]
        self.Ss = [self.current_Ss.tolist()]

        self.alpha_list = []#alphas
        self.beta_list = []
        print("ParticleSIR initialized")
    
    def update_states(self):
        for j in range(self.num_enbs):
            s = self.current_Ss[j]
            i = self.current_Is[j]
            s += self._delta_s(self.current_Ss[j], self.current_Is[j], 
                               self.alphas[j])
            i += self._delta_i(self.current_Ss[j], self.current_Is[j], 
                               self.alphas[j], self.betas[j])

            s = self.check_bounds(s)
            i = self.check_bounds(i)

            self.current_Is[j] = i
            self.current_Ss[j] = s

        self.Is.append(self.current_Is.tolist())
        self.Ss.append(self.current_Ss.tolist())

    def _init_filter(self):
        num_states = 4
        num_obs = 1
        print('intialize particle filter')
        self.filter = ParticleFilter(self.num_enbs)

    def predict_with_filter(self,params):
        F = self.filter

        while self.epoch < self.epochs - 1:
            X = as_matrix([self.current_Ss, self.current_Is, 
                           self.alphas, self.betas])
        
            F.fit(X)
            y = self.CDC_obs[self.epoch]
            F.step(y, predict_P=False)
            self.weights.append(F.weights)

            x_post = F.x_post
            for j in range(self.num_enbs):
                self.current_Ss[j] = self.check_bounds(x_post[0, j])
                self.current_Is[j] = self.check_bounds(x_post[1, j])
                self.alphas[j] = self.check_bounds(x_post[2, j])#, inf)    # self.check_bounds(x_post[2, j]) #, inf)
                self.betas[j] = self.check_bounds(x_post[3, j])#, inf)
                #print(self.alphas)

            self.update_states()
            self.epoch += 1
            self.alpha_list.append(mean(self.alphas))
            self.beta_list.append(mean(self.betas))
        self.get_score()

    def _delta_s(self, s, i, alpha):
        return - alpha * s * i

    def _delta_i(self, s, i, alpha, beta):
        return alpha * s * i - beta * i

    def check_par_bounds(self, par):
        if par < 0: par = 0
        return par

    def get_score(self):
        I_mat = as_array(self.Is)
        for i, w in enumerate(self.weights):
            I_mat[i] *= w 

        self.IS = sum(I_mat, axis=1)

        time_gap = 1 # self.epochs / 52
        idx = [x for x in range(self.epochs) if not x % time_gap]

        self.score = RSS(self.CDC_obs, self.IS[idx])
        self.scores = {}
        self.scores['SSE'] = self.score
        self.scores['RMSE'] = RMSE(self.CDC_obs, self.IS[idx])
        self.scores['MSPE'] = MSPE(self.CDC_obs, self.IS[idx])
        self.scores['CORR'] = corrcoef(self.CDC_obs, self.IS[idx])[0, 1]
        return self.score

def write_file(path, year, sir, out_str):
    directory = 'outs%s/%s' % (year, path)
    if not os.path.exists(directory): os.makedirs(directory)

    with open('%s/%s_%s_en_out' % (directory, ens, year), 'ab') as f:
        f.write('{}\n'.format(out_str).encode())
    with open('%s/%s_%s_en_out_par' % (directory, ens, year), 'ab') as f:
        f.write('{},{}\n'.format(mean(sir.alphas), mean(sir.betas)).encode())
    return sir.score

def sim_psir_filtered(ens, year, date, params=None):
    if not params:
        params = read_params('./data/params/params%s.csv' % year)
    params['filtering'] = True
    params['time_varying'] = False
    sir = ParticleSIR(ens, params)

    sir.predict_with_filter()
 
    out_str = ','.join(map(str, sir.IS))
    pprint(sir.scores)
    path = '%s_%s_predictions_example' % (date[0],date[1])
    write_file(path, year, sir, out_str)
    
    