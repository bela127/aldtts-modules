from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from alts.core.experiment_module import ExperimentModule
from alts.core.configuration import init

if TYPE_CHECKING:
    from typing_extensions import Self #type: ignore
    from alts.core.data_sampler import DataSampler
    from aldtts.modules.multi_sample_test import MultiSampleTest
    from alts.core.query.query_sampler import QuerySampler


@dataclass
class DependencyTest(ExperimentModule):
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

