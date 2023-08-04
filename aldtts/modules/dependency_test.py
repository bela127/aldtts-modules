from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

import numpy as np

import subprocess
from typing import TYPE_CHECKING

from xicor.xicor import Xi
from fcit import fcit

from dataclasses import dataclass
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import chi2_contingency
from scipy.stats import ttest_1samp
import numpy as np
import pandas as pd
from sklearn.utils import resample, shuffle

from alts.core.configuration import init, post_init, pre_init
from aldtts.core.dependency_test import DependencyTest

from aldtts.modules.XtendedCorrel import hoeffding

from hyppo.independence.dcorr import Dcorr
from hyppo.independence.hsic import Hsic
from hyppo.independence.hhg import HHG
from hyppo.independence.mgc import MGC
from hyppo.independence.kmerf import KMERF
from hyppo.independence.base import IndependenceTest
from hyppo.tools.common import perm_test

if TYPE_CHECKING:
    from typing import Dict
    from typing_extensions import Self #type: ignore
    from alts.core.data_sampler import DataSampler
    from aldtts.modules.multi_sample_test import MultiSampleTest
    from alts.core.query.query_sampler import QuerySampler
    from aldtts.core.dependency_measure import DependencyMeasure


@dataclass
class SampleTest(DependencyTest):
    query_sampler: QuerySampler = init()
    data_sampler: DataSampler = init()
    multi_sample_test : MultiSampleTest = init()

    def post_init(self):
        super().post_init()
        self.data_sampler = self.data_sampler(exp_modules = self.exp_modules)
        self.multi_sample_test = self.multi_sample_test()
        self.query_sampler = self.query_sampler(exp_modules=self.exp_modules)

    def test(self):

        queries = self.query_sampler.sample()

        sample_queries, samples = self.data_sampler.query(queries)

        t, p = self.multi_sample_test.test(samples)

        return t, p

@dataclass
class DependencyMeasureTest(DependencyTest):
    dependency_measure: DependencyMeasure = init()
    distribution: Dict = pre_init(default_factory=dict)
    scores: list = pre_init(default_factory=list)
    iterations: int = init(default=100)
    
    def post_init(self):
        super().post_init()
        self.dependency_measure = self.dependency_measure(self.exp_modules)

    def calc_p_value(self,t):
        greater = sum([v for k,v in self.distribution.items() if k >= t])
        return greater / len(self.scores)

    def calc_var(self):
        variance = np.var(self.scores)
        return variance

    def test(self):
        self.scores = []

        queries = self.data_pools.result.queries

        samples = self.data_pools.result.results

        self.scores.append(self.dependency_measure.apply(samples))
        
        for i in range(1,self.iterations):
            samples = tuple(shuffle(x) for x in samples)
            self.scores.append(self.dependency_measure.apply(samples))

        self.distribution = {item:self.scores.count(item) for item in self.scores}

        t, p, v = self.executeTest(samples)

        return t, p

    def executeTest(self, samples):
        t = self.dependency_measure.apply(samples)
        p = self.calc_p_value(t)
        v = self.calc_var()
        return t,p,v 

    @property
    @classmethod
    def __name__(cls):
        return f"{super().__name__}{np.random.uniform(0,1):04d}"
    

    
@dataclass
class Pearson(DependencyTest):

    def test(self):
        results = self.data_pools.result.results
        quries = self.data_pools.result.queries
        
        x = quries.flatten()
        y = results.flatten()
        t, p = pearsonr(x, y)
        return np.asarray([t]), np.asarray([p])

@dataclass
class Spearmanr(DependencyTest):

    def test(self):
        results = self.data_pools.result.results
        quries = self.data_pools.result.queries
        
        x = quries.flatten()
        y = results.flatten()
        t, p = spearmanr(x, y)
        return np.asarray([t]), np.asarray([p])

class Kendalltau(DependencyTest):

    def test(self):
        results = self.data_pools.result.results
        quries = self.data_pools.result.queries
        
        x = quries.flatten()
        y = results.flatten()
        t, p = kendalltau(x, y)
        return np.asarray([t]), np.asarray([p])
@dataclass
class FIT(DependencyTest):

    def test(self):
        quries = self.data_pools.result.queries
        results = self.data_pools.result.results

        x = quries
        y = results
        p, d0_stats, d1_stats = fcit.test(x, y, plot_return=True)
        t, _ = ttest_1samp(d0_stats / d1_stats, 1)
        return np.asarray([t]), np.asarray([p])
@dataclass
class XiCor(DependencyTest):

    def test(self):
        quries = self.data_pools.result.queries
        results = self.data_pools.result.results

        x = quries.flatten()
        y = results.flatten()

        t, p = self.xi(x, y)
        
        return np.asarray([t]), np.asarray([p])
    
    def xi(self,a,b):
        xi_obj = Xi(a,b)
        t = xi_obj.correlation
        p = xi_obj.pval_asymptotic(ties=False, nperm=1000)
        return t,p
@dataclass
class Hoeffdings(DependencyTest):

    def test(self):
        queries = self.data_pools.result.queries
        results = self.data_pools.result.results
        x = queries.flatten()
        y = results.flatten()
        
        p = hoeffding(x, y)
        t = 0  
        return np.asarray([t]), np.asarray([p])


@dataclass
class hypoDcorr(DependencyTest):

    def test(self):
        queries = self.data_pools.result.queries
        results = self.data_pools.result.results
        x = queries.flatten()
        y = results.flatten()
        stat,pvalue = Dcorr().test(x,y,workers=-1,reps=1000)
        return np.asarray([stat]), np.asarray([pvalue])

@dataclass
class hypoHsic(DependencyTest):

    def test(self):
        queries = self.data_pools.result.queries
        results = self.data_pools.result.results
        x = queries.flatten()
        y = results.flatten()
        stat,pvalue = Hsic().test(x,y,workers=-1,reps=1000)
        return np.asarray([stat]), np.asarray([pvalue])

@dataclass
class hypoHHG(DependencyTest):

    def test(self):
        queries = self.data_pools.result.queries
        results = self.data_pools.result.results
        x = queries.flatten()
        y = results.flatten()
        stat,pvalue = HHG().test(x,y,workers=-1,reps=1000, auto=True)
        return np.asarray([stat]), np.asarray([pvalue])

@dataclass
class hypoMGC(DependencyTest):

    def test(self):
        queries = self.data_pools.result.queries
        results = self.data_pools.result.results
        x = queries.flatten()
        y = results.flatten()
        stat,pvalue,_ = MGC().test(x,y,workers=-1,reps=1000)
        return np.asarray([stat]), np.asarray([pvalue])

@dataclass
class hypoKMERF(DependencyTest):

    def test(self):
        queries = self.data_pools.result.queries
        results = self.data_pools.result.results
        x = queries.flatten()
        y = results.flatten()
        stat,pvalue,_ = KMERF().test(y,x,workers=-1,reps=1000)
        return np.asarray([stat]), np.asarray([pvalue])

