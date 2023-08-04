from __future__ import annotations
from abc import abstractproperty
from typing import TYPE_CHECKING

from dataclasses import dataclass
from alts.core.data.constrains import QueryConstrain, ResultConstrain

import numpy as np

from aldtts.core.test_interpolation import TestInterpolator
from alts.core.configuration import init

if TYPE_CHECKING:
    from typing_extensions import Self #type: ignore
    from alts.core.data.data_sampler import DataSampler


@dataclass
class KNNTestInterpolator(TestInterpolator):
    data_sampler: DataSampler = init()

    def post_init(self):
        super().post_init()
        self.data_sampler = self.data_sampler(exp_modules= self.exp_modules)

    def query(self, queries):

        queries1 = queries[:,0,:]
        queries2 = queries[:,1,:]

        sample_queries1, samples1 = self.data_sampler.query(queries1)
        sample_queries2, samples2 = self.data_sampler.query(queries2)

        t,p = self.test.test(samples1, samples2)

        u = self.uncertainty((queries1, queries2), sample_queries1, sample_queries2)
        #u = 0

        return t, p, u
    
    def uncertainty(self, queries, queries1, queries2):
        query1, query2 = queries
        dists1 = np.linalg.norm(queries1-query1[:,None,:], axis=2)
        dists2 = np.linalg.norm(queries2-query2[:,None,:], axis=2)
        mean_dist = np.mean(np.concatenate((dists1,dists2), axis=1),axis=1)
        return mean_dist

    def query_constrain(self) -> QueryConstrain:
        return self.data_sampler.query_constrain()
    
    def result_constrain(self) -> ResultConstrain:
        return ResultConstrain(shape=(3,))


