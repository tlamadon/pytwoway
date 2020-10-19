'''
Computes a bunch of estimates from an event study data set:
    - AKM variance decomposition
    - Andrews bias correction
    - KSS bias correction
Does this through class FEsolver
'''

import pyamg
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix,coo_matrix,diags,linalg
import time
import pyreadr
import os
import logging
from multiprocessing import Pool, TimeoutError, set_start_method
from timeit import default_timer as timer
import itertools
import pickle
import time
import argparse
import json
import glob, sys

# Try to use tqdm
try:
    from tqdm import tqdm,trange
except ImportError:
    trange = range

# def pipe_qcov(df, e1, e2): # FIXME I moved this from above, also this is used only in commented out code
#     v1 = df.eval(e1)
#     v2 = df.eval(e2)
#     return np.cov(v1, v2)[0][1]

class FEsolver:
    '''
    Uses multigrid and partialing out to solve two way Fixed Effect model

    @ FIXME I think delete everything below this, it's basically contained in the class/functions within the class

    takes as an input this adata
    and creates associated A = [J W] matrix which are AKM dummies

    provides methods to do A x Y but also (A'A)^-1 A'Y solve method

    '''

    def __init__(self, params):
        '''
        Purpose:
            Initialize FEsolver object

        Arguments:
            params (dictionary): dictionary of parameters
                data (Pandas DataFrame): labor data. Contains the following columns:
                    wid (worker id)
                    y1 (compensation 1)
                    y2 (compensation 2)
                    f1i (firm id 1)
                    f2i (firm id 2)
                    m (0 if stayer, 1 if mover)
                ncore (int): number of cores to use
                ndraw_pii (int): number of draws to compute leverage
                ndraw_tr (int): number of draws to compute heteroskedastic correction
                hetero (bool): if True, compute heteroskedastic correction
                statsonly (bool): if True, return only basic statistics
                out (string): if statsonly is True, this is the file where the statistics will be saved
                batch (): @ FIXME I don't know what this is

        Returns:
            Object of type FEsolver
        '''
        start_time = time.time()
        self.params = params
        self.res = {} # Results dictionary

        # Save some commonly used parameters as attributes
        self.ncore = self.params['ncore'] # number of cores to use
        self.ndraw_pii = self.params['ndraw_pii'] # number of draws to compute leverage
        self.ndraw_trace = self.params['ndraw_tr'] # number of draws to compute hetero correction
        self.compute_hetero = self.params['hetero']

        # Store some parameters in results dictionary
        self.res['cores'] = self.ncore
        self.res['ndp'] = self.ndraw_pii
        self.res['ndt'] = self.ndraw_trace

        # Begin cleaning and analysis
        self.prep_data() # Prepare data
        self.init_prepped_adata() # Use cleaned adata to generate some attributes
        self.compute_early_stats() # Use cleaned data to compute some statistics

        if self.params['statsonly']: # If only returning early statistics
            self.save_early_stats()

        else: # If running analysis
            self.create_fe_solver() # Solve FE model
            self.compute_trace_approximation_fe() # Compute trace approxmation

            # If computing heteroskedastic correction
            if self.compute_hetero:
                self.compute_leverages_Pii() # Solve he model
                self.compute_trace_approximation_he() # Compute trace approximation

            self.collect_res() # Collect all results

        end_time = time.time()

        self.res['total_time'] = end_time - start_time

        self.save_res() # Save results to json

        logger.info('------ DONE -------')

    def __getstate__(self):
        '''
        Purpose:
            Defines how the model is pickled

        Arguments:
            Nothing

        Returns:
            Nothing
        '''
        odict = {k: self.__dict__[k] for k in self.__dict__.keys() - {'ml'}}
        return odict

    def __setstate__(self, d):
        '''
        Purpose:
            Defines how the model is unpickled

        Arguments:
            d (dictionary): attribute dictionary

        Returns:
            Nothing
        '''
        # Need to recreate the simple model and the search representation
        self.__dict__ = d # Make d the attribute dictionary
        self.ml = pyamg.ruge_stuben_solver(self.M)

    @staticmethod
    def load(filename):
        '''
        Purpose:
            Load files for heteroskedastic correction

        Arguments:
            filename (string): file to load

        Returns:
            fes: loaded file
        '''
        fes = None
        with open(filename, 'rb') as infile:
            fes = pickle.load(infile)
        return fes

    def save(self, filename):
        '''
        Purpose:
            Save FEsolver class to filename as pickle

        Arguments:
            filename (string): filename to save to

        Returns:
            Nothing
        '''
        with open(filename, 'wb') as outfile:
            pickle.dump(self, outfile)

    def prep_data(self):
        '''
        Purpose:
            Do some initial data cleaning

        Arguments:
            Nothing

        Returns:
            Nothing
        '''
        logger.info('Preparing the data')

        data = self.params['data']
        sdata = data[data['m'] == 0].reset_index(drop=True)
        jdata = data[data['m'] == 1].reset_index(drop=True)

        logger.info('Data movers={} stayers={}'.format(len(jdata), len(sdata)))
        self.res['nm'] = len(jdata)
        self.res['ns'] = len(sdata)
        self.res['n_firms'] = len(np.unique(pd.concat([jdata['f1i'], jdata['f2i'], sdata['f1i']], ignore_index=True)))
        self.res['n_workers'] = len(np.unique(pd.concat([jdata['wid'], sdata['wid']], ignore_index=True)))
        self.res['n_movers'] = len(np.unique(pd.concat([jdata['wid']], ignore_index=True)))
        #res['year_max'] = int(sdata['year'].max())
        #res['year_min'] = int(sdata['year'].min())

        # Make wids unique per row
        jdata.set_index(np.arange(self.res['nm']) + 1)
        sdata.set_index(np.arange(self.res['ns']) + 1 + self.res['nm'])
        jdata['wid'] = np.arange(self.res['nm']) + 1
        sdata['wid'] = np.arange(self.res['ns']) + 1 + self.res['nm']

        # Combine the 2 data-sets
        self.adata = pd.concat([sdata[['wid', 'f1i', 'y1']].assign(cs=1, m=0), jdata[['wid', 'f1i', 'y1']].assign(cs=1, m=1), jdata[['wid', 'f2i', 'y2']].rename(columns={'f2i': 'f1i', 'y2': 'y1'}).assign(cs=0, m=1)])
        self.adata = self.adata.reset_index(drop=True) # FIXME changed from set_index(pd.Series(range(len(self.adata))))
        self.adata['wid'] = self.adata['wid'].astype('category').cat.codes + 1

    def init_prepped_adata(self):
        '''
        Purpose:
            Use prepped adata to initialize class attributes

        Arguments:
            adata (Pandas DataFrame): labor data. Contains the following columns:
                wid (worker id)
                y1 (compensation 1)
                y2 (compensation 2)
                f1i (firm id 1)
                f2i (firm id 2)
                m (0 if stayer, 1 if mover)

        Returns:
            Nothing
        '''
        nf = self.adata.f1i.max() # Number of firms
        nw = self.adata.wid.max() # Number of workers
        nn = len(self.adata) # Number of observations
        self.nf = nf
        self.nw = nw
        self.nn = nn
        logger.info('data nf:{} nw:{} nn:{}'.format(nf, nw, nn))

        # Matrices for the cross-section
        J = csc_matrix((np.ones(nn), (self.adata.index, self.adata.f1i - 1)), shape=(nn, nf)) # Firms
        J = J[:, range(nf - 1)]  # Normalize one firm to 0
        self.J = J
        W = csc_matrix((np.ones(nn), (self.adata.index, self.adata.wid - 1)), shape=(nn, nw)) # Workers
        self.W = W
        # Dw = diags((W.T * W).diagonal()) # FIXME changed from .transpose() to .T ALSO commented this out since it's not used
        Dwinv = diags(1.0 / ((W.T * W).diagonal())) # FIXME changed from .transpose() to .T
        self.Dwinv = Dwinv

        logger.info('Prepare linear solver')

        # Finally create M
        M = J.T * J - J.T * W * Dwinv * W.T * J # FIXME changed from .transpose() to .T
        self.M = M
        self.ml = pyamg.ruge_stuben_solver(M)

        # L = diags(fes.M.diagonal()) - fes.M
        # r = linalg.eigsh(L,k=2,which='LM')

        # Create cross-section matrices
        # cs == 1 ==> looking at y1 for movers (cs = cross section)
        mdata = self.adata[self.adata['cs'] == 1] # FIXME changed from adata.query('cs==1') (I ran %timeit and slicing is faster)
        mdata = mdata.reset_index(drop=True) # FIXME changed from set_index(pd.Series(range(len(mdata))))

        nnq = len(mdata) # Number of observations
        self.nnq = nnq
        Jq = csc_matrix((np.ones(nnq), (mdata.index, mdata.f1i - 1)), shape=(nnq, nf))
        self.Jq = Jq[:, range(nf - 1)]  # normalizing one firm to 0
        self.Wq = csc_matrix((np.ones(nnq), (mdata.index, mdata.wid - 1)), shape=(nnq, nw))
        self.Yq = mdata['y1']

        # Save time variable
        self.last_invert_time = 0

    def weighted_quantile(self, values, quantiles, sample_weight=None, values_sorted=False, old_style=False): # FIXME was formerly a function outside the class
        '''
        Purpose:
            Very close to numpy.percentile, but supports weights.
            NOTE: quantiles should be in [0, 1]!

        Arguments:
            :param values: numpy.array with data
            :param quantiles: array-like with many quantiles needed
            :param sample_weight: array-like of the same length as `array`
            :param values_sorted: bool, if True, then will avoid sorting of initial array
            :param old_style: if True, will correct output to be consistent with numpy.percentile.

        Returns:
            :return: numpy.array with computed quantiles.
        '''
        values = np.array(values)
        quantiles = np.array(quantiles)
        if sample_weight is None:
            sample_weight = np.ones(len(values))
        sample_weight = np.array(sample_weight)
        assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
            'quantiles should be in [0, 1]'

        if not values_sorted:
            sorter = np.argsort(values)
            values = values[sorter]
            sample_weight = sample_weight[sorter]

        weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
        if old_style:
            # To be convenient with numpy.percentile
            weighted_quantiles -= weighted_quantiles[0]
            weighted_quantiles /= weighted_quantiles[-1]
        else:
            weighted_quantiles /= np.sum(sample_weight)

        return np.interp(quantiles, weighted_quantiles, values)

    def weighted_var(self, v, w): # FIXME was formerly a function outside the class
        '''
        Purpose:
            Compute weighted variance @ FIXME I don't know what this function really does

        Arguments:
            v: @ FIXME I don't know what this is
            w: @ FIXME I don't know what this is

        Returns:
            v0: @ FIXME I don't know what this is
        '''
        m0 = np.sum(w * v) / np.sum(w)
        v0 = np.sum(w * (v - m0) ** 2) / np.sum(w)

        return v0

    def compute_early_stats(self):
        '''
        Purpose:
            Compute some early statistics

        Arguments:
            Nothing

        Returns:
            Nothing
        '''
        fdata = self.adata.groupby('f1i').agg({'m':'sum', 'y1':'mean', 'wid':'count' })
        self.res['mover_quantiles'] = self.weighted_quantile(fdata['m'], np.linspace(0, 1, 11), fdata['wid']).tolist()
        self.res['size_quantiles'] = self.weighted_quantile(fdata['wid'], np.linspace(0, 1, 11), fdata['wid']).tolist()
        self.res['between_firm_var'] = self.weighted_var(fdata['y1'], fdata['wid'])
        self.res['var_y'] = self.adata[self.adata['cs'] == 1]['y1'].var() # FIXME changed from adata.query('cs==1') (I ran %timeit and slicing is faster)

        # extract woodcock moments using sdata and jdata
        # get averages by firms for stayers
        #dsf  = adata.query('cs==1').groupby('f1i').agg(y1sj=('y1','mean'), nsj=('y1','count'))
        #ds   = pd.merge(adata.query('cs==1'), dsf, on="f1i")
        #ds.eval("y1s_lo    = (nsj * y1sj - y1) / (nsj - 1)",inplace=True)
        #res['woodcock_var_psi']   = ds.query('nsj  > 1').pipe(pipe_qcov, 'y1', 'y1s_lo')
        #res['woodcock_var_alpha'] = np.minimum( jdata.pipe(pipe_qcov, 'y1','y2'), adata.query('cs==1')['y1'].var() - res['woodcock_var_psi'] )
        #res['woodcock_var_eps'] = adata.query('cs==1')['y1'].var() - res['woodcock_var_alpha'] - res['woodcock_var_psi']
        #logger.info("[woodcock] var psi = {}", res['woodcock_var_psi'])
        #logger.info("[woodcock] var alpha = {}", res['woodcock_var_alpha'])
        #logger.info("[woodcock] var eps = {}", res['woodcock_var_eps'])

    def save_early_stats(self):
        '''
        Purpose:
            Save the early statistics computed in compute_early_stats()

        Arguments:
            Nothing

        Returns:
            Nothing
        '''
        with open(self.params['out'], 'w') as outfile:
            json.dump(self.res, outfile)
        logger.info('saved results to {}'.format(self.params['out']))
        logger.info('--statsonly was passed as argument, so we skip all estimation.')
        logger.info('------ DONE -------')
        # sys.exit() # FIXME I don't think this is necessary (does it even work?) since this is now a class object

    def create_fe_solver(self):
        '''
        Purpose:
            Solve FE model

        Arguments:
            Nothing

        Returns:
            Nothing
        '''
        self.Y = self.adata.y1

        # try to pickle the object to see its size
        # self.save('tmp.pkl') # FIXME should we delete these 2 lines?

        logger.info('Extract firm effects')

        psi_hat, alpha_hat = self.solve(self.Y)

        logger.info('Solver time {:2.4f} seconds'.format(self.last_invert_time))
        logger.info('Expected total time {:2.4f} minutes'.format( (self.ndraw_trace * (1 + self.compute_hetero) + self.ndraw_pii * self.compute_hetero) * self.last_invert_time / 60))

        self.E = self.Y - self.mult_A(psi_hat, alpha_hat)

        self.res['solver_time'] = self.last_invert_time

        fe_rsq = 1 - np.power(self.E, 2).mean() / np.power(self.Y, 2).mean()
        logger.info('Fixed effect R-square {:2.4f}'.format(fe_rsq))

        self.var_fe = np.var(self.Jq * psi_hat)
        self.cov_fe = np.cov(self.Jq * psi_hat, self.Wq * alpha_hat)[0][1]
        self.tot_var  = np.var(self.Y)
        logger.info('[fe] var_psi={:2.4f} cov={:2.4f} tot={:2.4f}'.format(self.var_fe, self.cov_fe, self.tot_var))

        self.var_e = self.nn / (self.nn - self.nw - self.nf + 1) * np.power(self.E, 2).mean()
        logger.info('[ho] variance of residuals {:2.4f}'.format(self.var_e))

    def compute_leverages_Pii(self):
        '''
        Purpose:
            Compute leverages for heteroskedastic correction

        Arguments:
            Nothing

        Returns:
            Nothing
        '''
        self.Pii = np.zeros(self.nn)
        self.Sii = np.zeros(self.nn)

        if len(self.params['levfile']) > 1:
            logger.info('[he] starting heteroskedastic correction, loading precomputed files')

            files = glob.glob('{}*'.format(self.params['levfile']))
            logger.info('[he] found {} files to get leverages from'.format(len(files)))
            self.res['lev_file_count'] = len(files)
            assert len(files) > 0, "Didn't find any leverage files!"

            for f in files:
                pp = np.load(f)
                self.Pii += pp / len(files)

        elif self.ncore > 1:
            logger.info('[he] starting heteroskedastic correction p2={}, using {} cores, batch size {}'.format(self.ndraw_pii, self.ncore, self.params['batch']))
            set_start_method('spawn')
            with Pool(processes=self.ncore) as pool:
                Pii_all = pool.starmap(self.leverage_approx, [self.params['batch'] for _ in range(self.ndraw_pii // self.params['batch'])])

            for pp in Pii_all:
                Pii += pp / len(Pii_all)

        else:
            Pii_all = list(itertools.starmap(self.leverage_approx, [self.params['batch'] for _ in range(self.ndraw_pii // self.params['batch'])]))

            for pp in Pii_all:
                self.Pii += pp / len(Pii_all)

        I = 1.0 * self.adata.eval('m == 1')
        max_leverage = (I * self.Pii).max()

        # Attach the computed Pii to the dataframe
        self.adata['Pii'] = self.Pii
        logger.info('[he] Leverage range {:2.4f} to {:2.4f}'.format(self.adata.query('m == 1').Pii.min(), self.adata.query('m == 1').Pii.max()))

        # Give stayers the variance estimate at the firm level
        self.adata['Sii'] = self.Y * self.E / (1 - self.Pii)
        S_j = self.adata.query('m == 1').rename(columns={'Sii': 'Sii_j'}).groupby('f1i')['Sii_j'].agg('mean')

        self.adata = pd.merge(self.adata, S_j, on='f1i')
        self.adata['Sii'] = np.where(self.adata['m'] == 1, self.adata['Sii'], self.adata['Sii_j'])
        self.Sii = self.adata['Sii']

        logger.info('[he] variance of residuals in heteroskedastic case: {:2.4f}'.format(self.Sii.mean()))

    def compute_trace_approximation_fe(self):
        '''
        Purpose:
            Compute FE trace approximation

        Arguments:
            Nothing

        Returns:
            Nothing
        '''
        logger.info('Starting FE trace correction ndraws={}, using {} cores'.format(self.ndraw_trace, self.ncore))
        self.tr_var_ho_all = np.zeros(self.ndraw_trace)
        self.tr_cov_ho_all = np.zeros(self.ndraw_trace)

        for r in trange(self.ndraw_trace):
            # Generate -1 or 1
            Zpsi = 2 * np.random.binomial(1, 0.5, self.nf - 1) - 1
            Zalpha = 2 * np.random.binomial(1, 0.5, self.nw) - 1

            R1 = self.Jq * Zpsi
            psi1, alpha1 = self.mult_AAinv(Zpsi, Zalpha)
            R2_psi = self.Jq * psi1
            R2_alpha = self.Wq * alpha1

            # Trace corrections
            self.tr_var_ho_all[r] = np.cov(R1, R2_psi)[0][1]
            self.tr_cov_ho_all[r] = np.cov(R1, R2_alpha)[0][1]

            logger.debug('FE [traces] step {}/{} done.'.format(r, self.ndraw_trace))

    def compute_trace_approximation_he(self):
        '''
        Purpose:
            Compute heteroskedastic trace approximation

        Arguments:
            Nothing

        Returns:
            Nothing
        '''
        logger.info('Starting he trace correction ndraws={}, using {} cores'.format(self.ndraw_trace, self.ncore))
        self.tr_var_he_all = np.zeros(self.ndraw_trace)
        self.tr_cov_he_all = np.zeros(self.ndraw_trace)

        for r in trange(self.ndraw_trace):
            Zpsi = 2 * np.random.binomial(1, 0.5, self.nf - 1) - 1
            Zalpha = 2 * np.random.binomial(1, 0.5, self.nw) - 1

            psi1, alpha1 = self.mult_AAinv(Zpsi, Zalpha)
            R2_psi = self.Jq * psi1
            R2_alpha = self.Wq * alpha1

            psi2, alpha2 = self.mult_AAinv(*self.mult_Atranspose(self.Sii * self.mult_A(Zpsi, Zalpha)))
            R3_psi = self.Jq * psi2

            # Trace corrections
            self.tr_var_he_all[r] = np.cov(R2_psi, R3_psi)[0][1]
            self.tr_cov_he_all[r] = np.cov(R2_alpha, R3_psi)[0][1]

            logger.debug('he [traces] step {}/{} done.'.format(r, self.ndraw_trace))

    def collect_res(self):
        '''
        Purpose:
            Collect all results

        Arguments:
            Nothing

        Returns:
            Nothing
        '''
        self.res['tot_var'] = self.tot_var
        self.res['eps_var_ho'] = self.var_e
        self.res['eps_var_fe'] = np.var(self.E)
        self.res['tr_var_ho'] = np.mean(self.tr_var_ho_all)
        self.res['tr_cov_ho'] = np.mean(self.tr_cov_ho_all)
        logger.info('[ho] VAR tr={:2.4f} (sd={:2.4e})'.format(self.res['tr_var_ho'], np.std(self.tr_var_ho_all)))
        logger.info('[ho] COV tr={:2.4f} (sd={:2.4e})'.format(self.res['tr_cov_ho'], np.std(self.tr_cov_ho_all)))

        if self.compute_hetero:
            self.res['eps_var_he'] = self.Sii.mean()
            self.res['min_lev'] = self.adata.query('m == 1').Pii.min()
            self.res['max_lev'] = self.adata.query('m == 1').Pii.max()
            self.res['tr_var_he'] = np.mean(self.tr_var_he_all)
            self.res['tr_cov_he'] = np.mean(self.tr_cov_he_all)
            self.res['tr_var_ho_sd'] = np.std(self.tr_var_ho_all)
            self.res['tr_cov_ho_sd'] = np.std(self.tr_cov_ho_all)
            self.res['tr_var_he_sd'] = np.std(self.tr_var_he_all)
            self.res['tr_cov_he_sd'] = np.std(self.tr_cov_he_all)
            logger.info('[he] VAR tr={:2.4f} (sd={:2.4e})'.format(self.res['tr_var_he'], np.std(self.tr_var_he_all)))
            logger.info('[he] COV tr={:2.4f} (sd={:2.4e})'.format(self.res['tr_cov_he'], np.std(self.tr_cov_he_all)))

        # ----- FINAL ------
        logger.info('[ho] VAR fe={:2.4f} bc={:2.4f}'.format(self.var_fe, self.var_fe - self.var_e * self.res['tr_var_ho']))
        logger.info('[ho] COV fe={:2.4f} bc={:2.4f}'.format(self.cov_fe, self.cov_fe - self.var_e * self.res['tr_cov_ho']))

        if self.compute_hetero:
            logger.info('[he] VAR fe={:2.4f} bc={:2.4f}'.format(self.var_fe, self.var_fe - self.res['tr_var_he']))
            logger.info('[he] COV fe={:2.4f} bc={:2.4f}'.format(self.cov_fe, self.cov_fe - self.res['tr_cov_he']))

        self.res['var_y'] = np.var(self.Yq)
        self.res['var_fe'] = self.var_fe
        self.res['cov_fe'] = self.cov_fe
        self.res['var_ho'] = self.var_fe - self.var_e * self.res['tr_var_ho']
        self.res['cov_ho'] = self.cov_fe - self.var_e * self.res['tr_cov_ho']

        if self.compute_hetero:
            self.res['var_he'] = self.var_fe - self.res['tr_var_he']
            self.res['cov_he'] = self.cov_fe - self.res['tr_cov_he']

    def save_res(self):
        '''
        Purpose:
            Save results as json

        Arguments:
            Nothing

        Returns:
            Nothing
        '''
        with open(self.params['out'], 'w') as outfile:
            json.dump(self.res, outfile)

        logger.info('Saved results to {}'.format(self.params['out']))

    def solve(self, Y):
        '''
        Purpose:
            Compute (A'A)^-1 A'Y, the least squares estimate of A [psi_hat, alpha_hat] = Y

        Arguments:
            Y (Pandas DataFrame): labor data

        Returns:
            psi_hat: estimated firm fixed effects @ FIXME correct datatype
            alpha_hat: estimated worker fixed effects @ FIXME correct datatype
        '''
        J_transpose_Y, W_transpose_Y = self.mult_Atranspose(Y) # This gives A'Y
        psi_hat, alpha_hat = self.mult_AAinv(J_transpose_Y, W_transpose_Y)

        return psi_hat, alpha_hat

    def mult_A(self, psi, alpha):
        '''
        Purpose:
            Multiplies A = [J W] stored in the object by psi and alpha (used, for example, to compute estimated outcomes and sample errors)

        Arguments:
            psi: firm fixed effects @ FIXME correct datatype
            alpha: worker fixed effects @ FIXME correct datatype

        Returns:
            J_psi + W_alpha (CSC Matrix): firms * firm fixed effects + workers * worker fixed effects
        '''
        # J_psi = self.J * psi
        # W_alpha = self.W * alpha

        return self.J * psi + self.W * alpha # J_psi + W_alpha

    def mult_Atranspose(self, v):
        '''
        Purpose:
            Multiplies the transpose of A = [J W] stored in the object by v

        Arguments:
            v: what to multiply by @ FIXME correct datatype

        Returns:
            J_transpose_V (CSC Matrix): firms * v
            W_transpose_V (CSC Matrix): workers * v
        '''
        # J_transpose_V = self.J.T * v # FIXME changed from .transpose() to .T
        # W_transpose_V = self.W.T * v # FIXME changed from .transpose() to .T

        return self.J.T * v, self.W.T * v # J_transpose_V, W_transpose_V

    def mult_AAinv(self, psi, alpha):
        '''
        Purpose:
            Multiplies gamma = [psi;alpha] by (A'A)^(-1) where A = [J W] stored in the object

        Arguments:
            psi: firm fixed effects @ FIXME correct datatype
            alpha: worker fixed effects @ FIXME correct datatype

        Returns:
            psi_out: estimated firm fixed effects @ FIXME correct datatype
            alpha_out : estimated worker fixed effects @ FIXME correct datatype
        '''
        # inter1 = self.ml.solve( psi , tol=1e-10 )
        # inter2 = self.ml.solve(  , tol=1e-10 )
        # psi_out = inter1 - inter2

        start = timer()
        psi_out = self.ml.solve(psi - self.J.T * (self.W * (self.Dwinv * alpha)), tol=1e-10) # FIXME changed from .transpose() to .T
        self.last_invert_time = timer() - start

        alpha_out = - self.Dwinv * (self.W.T * (self.J * psi_out)) + self.Dwinv * alpha # FIXME changed from .transpose() to .T

        return psi_out, alpha_out

    def proj(self, y): # FIXME should this y be Y?
        '''
        Purpose:
            Solve y, then project onto X space of data stored in the object

        Arguments:
            y (Pandas DataFrame): labor data

        Returns:
            Projection of psi, alpha solved from y onto X space
        '''
        return self.mult_A(*self.solve(y))

    def leverage_approx(self, ndraw_pii):
        '''
        Purpose:
            Compute an approximate leverage using ndraw_pii

        Arguments:
            ndraw_pii (int): number of draws

        Returns:
            Pii (NumPy Array): @ FIXME I don't know what this function does, so I don't know what Pii is
        '''
        Pii = np.zeros(self.nn)

        # Compute the different draws
        for r in trange(ndraw_pii):
            R2  = 2 * np.random.binomial(1, 0.5, self.nn) - 1
            Pii += 1 / ndraw_pii * np.power(self.proj(R2), 2.0)

        logger.info('Done with batch')

        return Pii

# Begin logging
logger = logging.getLogger('akm')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('akm_spam.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
