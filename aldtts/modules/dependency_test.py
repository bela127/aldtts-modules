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
import numpy as np
import pandas as pd
from sklearn.utils import resample, shuffle

from alts.core.configuration import init, post_init, pre_init
from aldtts.core.dependency_test import DependencyTest

from aldtts.modules.XtendedCorrel import hoeffding

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

    def __post_init__(self):
        super().__post_init__()
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
    iterations: int = init(default=0)
    
    def __init__(self, dependency_measure: DependencyMeasure, iterations:int = 100):
        self.dependency_measure = dependency_measure
        self.iterations = iterations

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
            samples = tuple(shuffle(x) for x in samples) #DevSkim: ignore DS148264 
            self.scores.append(self.dependency_measure.apply(samples))

        self.distribution = {item:self.scores.count(item) for item in self.scores}

        t, p, v = self.executeTest(samples)

        return t, p, v

    def executeTest(self, samples):
        t = self.dependency_measure.apply(samples)
        p = self.calc_p_value(t)
        v = self.calc_var()
        return t,p,v 

@dataclass
class Pearson(DependencyTest):

    def test(self):
        results = self.data_pools.result.results
        x = [item for sublist in results for item in sublist]
        t, p = pearsonr(x, x)
        return t,p,0
@dataclass
class Spearmanr(DependencyTest):

    def test(self):
        results = self.data_pools.result.results
        x = [item for sublist in results for item in sublist]
        t, p = spearmanr(x, x)
        return t,p,0
@dataclass
class Kendalltau(DependencyTest):

    def test(self):
        results = self.data_pools.result.results
        x = [item for sublist in results for item in sublist]
        t, p = kendalltau(x, x)
        return t,p,0
@dataclass
class FIT(DependencyTest):

    def test(self):
        queries = self.data_pools.result.queries

        samples = self.data_pools.result.results
        samples = np.array(samples)
        p = fcit.test(samples, samples)
        return 0,p,0
@dataclass
class XiCor(DependencyTest):

    def test(self):
        queries = self.data_pools.result.queries

        samples = self.data_pools.result.results
        x = [item for sublist in samples for item in sublist]
        xi_obj = Xi(x,x)
        #t = 0 if and only if X and Y are independent
        t = xi_obj.correlation
        p = xi_obj.pval_asymptotic(ties=False, nperm=1000)      
        return t, p, 0
@dataclass
class Hoeffdings(DependencyTest):

    def test(self):
        queries = self.data_pools.result.queries

        samples = self.data_pools.result.results
        x = [item for sublist in samples for item in sublist]
        y = [item for item in x]
        samples = np.array(y)
        p = hoeffding(samples,samples)    
        return 0, p,0

@dataclass
class chi_square(DependencyTest):

    def test(self):
        queries = self.data_pools.result.queries

        samples = self.data_pools.result.results
        r, p, dof, expected = chi2_contingency(samples, samples)
        return r,p,0

@dataclass
class A_dep_test(DependencyTest):

    def test(self):
        queries = self.data_pools.result.queries

        samples = self.data_pools.result.results
        return 0, 0, 0

@dataclass
class IndepTest(DependencyTest):

    def test(self):
        queries = self.data_pools.result.queries

        samples = self.data_pools.result.results
        x = np.asarray([item.tolist() for sublist in samples for item in sublist])
        dataFile = 'run_data_store/indepTestData.csv'
        df = pd.DataFrame(x)
        df.to_csv(dataFile, sep=",", header=['true'], index=False)
 
        command = 'Rscript'
        path = 'C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/r_scripts/IndepTest.r'
        cmd = [command, path, '--vanilla'] 
        if(len(x)>50):
            output = subprocess.check_output(cmd)
            line = next(x for x in output.splitlines() if x.startswith(b'[1]'))
            p = float(line.split()[1])
        else:
            p = 0
        return 0, p,0

@dataclass
class CondIndTest(DependencyTest):

    def test(self):
        queries = self.data_pools.result.queries

        samples = self.data_pools.result.results
        x = np.asarray([item.tolist() for sublist in samples for item in sublist])
        dataFile = 'run_data_store/condIndTestData.csv'
        df = pd.DataFrame(x)
        df.to_csv(dataFile, sep=",", header=['true'], index=False)

        output = subprocess.check_output(["Rscript",  "--vanilla", "C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/r_scripts/CondIndTest.r"])
        line = next(x for x in output.splitlines() if x.startswith(b'[1]'))
        p = float(line.split()[1])
        return 0, p, 0
@dataclass
class LISTest(DependencyTest):

    def test(self):
        queries = self.data_pools.result.queries

        samples = self.data_pools.result.results
        x = np.asarray([item.tolist() for sublist in samples for item in sublist])
        dataFile = 'run_data_store/LISTestData.csv'
        df = pd.DataFrame(x)
        df.to_csv(dataFile, sep=",", header=['true'], index=False)

        if(len(x)>20 and len(x)<200):
            output = subprocess.check_output(["Rscript",  "--vanilla", "C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/r_scripts/LISTest.r"])
            p = float(output.splitlines()[5].split()[1])
            t = float(output.splitlines()[8].split()[1])
        else:
            p = 0
            t = 0
        return t, p,0
