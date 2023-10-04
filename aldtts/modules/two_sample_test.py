from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
import numpy as np

from scipy.stats import mannwhitneyu #type: ignore
from scipy.stats import kruskal #type: ignore
from scipy.stats import epps_singleton_2samp, cramervonmises_2samp


from aldtts.core.two_sample_test import TwoSampleTest

if TYPE_CHECKING:
    ...

@dataclass
class MWUTwoSampleTest(TwoSampleTest):
    
    def test(self, samples1, samples2):

        #def step(i):
        #    return mannwhitneyu(samples1[i], samples2[i])
        
        #idx = np.arange(len(samples1))
        #results = np.array(list(map(step, idx)))

        U1, p = mannwhitneyu(samples1, samples2, axis=1)
        U2 = samples1.shape[1] * samples2.shape[1] - U1
        U = np.min((U1,U2), axis=0)
        r = 1 - (2*U)/(samples1.shape[1] * samples2.shape[1])

        #U = results[:,0,:]
        #p = results[:,1,:]
        return r, p

@dataclass
class EppsSingTwoSampleTest(TwoSampleTest):
    
    def test(self, samples1, samples2):

        #def step(i):
        #    return mannwhitneyu(samples1[i], samples2[i])
        
        #idx = np.arange(len(samples1))
        #results = np.array(list(map(step, idx)))
        t, p = np.vectorize(epps_singleton_2samp, signature='(n),(n)->(),()')(samples1[:,:,0], samples2[:,:,0])


        return t[:,None], p[:,None]


@dataclass
class CramMisTwoSampleTest(TwoSampleTest):
    
    def test(self, samples1, samples2):

        #def step(i):
        #    return mannwhitneyu(samples1[i], samples2[i])
        
        #idx = np.arange(len(samples1))
        #results = np.array(list(map(step, idx)))
        def test(x,y):
            res = cramervonmises_2samp(x,y, method='asymptotic')
            return res.statistic, res.pvalue

        t, p = np.vectorize(test, signature='(n),(n)->(),()')(samples1[:,:,0], samples2[:,:,0])

        return t[:,None], p[:,None]

@dataclass
class KWHTwoSampleTest(TwoSampleTest):
    
    def test(self, samples1, samples2):

        #def step(i):
        #    return mannwhitneyu(samples1[i], samples2[i])
        
        #idx = np.arange(len(samples1))
        #results = np.array(list(map(step, idx)))

        t, p = kruskal(samples1, samples2, axis=1)

        #U = results[:,0,:]
        #p = results[:,1,:]
        return t, p
    
