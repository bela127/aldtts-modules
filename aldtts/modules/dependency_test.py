from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from alts.core.experiment_module import ExperimentModule

if TYPE_CHECKING:
    from typing_extensions import Self #type: ignore
    from alts.core.data_sampler import DataSampler
    from aldtts.modules.multi_sample_test import MultiSampleTest
    from alts.core.query.query_sampler import QuerySampler


@dataclass
class DependencyTest(ExperimentModule):
    query_sampler: QuerySampler
    data_sampler: DataSampler
    multi_sample_test : MultiSampleTest

    def test(self):

        queries = self.query_sampler.sample()

        sample_queries, samples = self.data_sampler.query(queries)

        t, p = self.multi_sample_test.test(samples)

        return t, p

    def __call__(self, exp_modules = None, **kwargs) -> Self:
        obj = super().__call__(exp_modules, **kwargs)
        obj.data_sampler = obj.data_sampler(exp_modules)
        obj.multi_sample_test = obj.multi_sample_test()
        obj.query_sampler = obj.query_sampler(obj.data_sampler)
        return obj
