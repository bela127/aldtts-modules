from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from alts.core.experiment_modules import InitQueryExperimentModules
from alts.core.configuration import init
from alts.modules.query.query_sampler import LatinHypercubeQuerySampler


if TYPE_CHECKING:
    from aldtts.modules.test_interpolation import TestInterpolator
    from aldtts.modules.dependency_test import DependencyTest
    from alts.core.data.queried_data_pool import QueriedDataPool
    from alts.core.data.data_pool import DataPool
    from alts.core.query.query_sampler import QuerySampler
    from typing_extensions import Self


@dataclass
class DependencyExperiment(InitQueryExperimentModules):
    dependency_test: DependencyTest = init()

    initial_query_sampler: QuerySampler = LatinHypercubeQuerySampler(num_queries=10)

    def __post_init__(self):
        super().__post_init__()
        self.dependency_test = self.dependency_test(exp_modules=self)

    def run(self):
        t,p = self.dependency_test.test()


@dataclass
class InterventionDependencyExperiment(DependencyExperiment):
    test_interpolator: TestInterpolator = init()

    def __post_init__(self):
        super().__post_init__()
        self.test_interpolator = self.test_interpolator(self)

    def run(self):
        t,p = self.dependency_test.test()

