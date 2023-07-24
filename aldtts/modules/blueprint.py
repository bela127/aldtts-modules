from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
from alts.core.blueprint import Blueprint
from alts.modules.data_process.process import DataSourceProcess
from alts.modules.oracle.data_source import LineDataSource
from alts.core.data.data_pools import ResultDataPools
from alts.modules.queried_data_pool import FlatQueriedDataPool
from alts.core.oracle.oracles import POracles
from alts.modules.oracle.query_queue import FCFSQueryQueue
from alts.modules.data_process.time_source import IterationTimeSource
from alts.modules.stopping_criteria import TimeStoppingCriteria
from aldtts.modules.evaluator import LogPValueEvaluator
from alts.modules.oracle.augmentation import NoiseAugmentation
from aldtts.modules.experiment_modules import DependencyExperiment
from alts.modules.data_sampler import KDTreeKNNDataSampler, KDTreeRegionDataSampler
from alts.modules.query.query_sampler import LastQuerySampler, RandomChoiceQuerySampler, UniformQuerySampler, LatinHypercubeQuerySampler
from aldtts.modules.multi_sample_test import KWHMultiSampleTest
from aldtts.modules.dependency_test import SampleTest
from alts.core.query.query_selector import ResultQuerySelector
from alts.modules.query.query_decider import ThresholdQueryDecider
from alts.modules.query.query_optimizer import NoQueryOptimizer
from aldtts.modules.selection_criteria import QueryTestNoSelectionCritera

if TYPE_CHECKING:
    from typing import Iterable
    from alts.core.data_process.time_source import TimeSource
    from alts.core.data_process.process import Process
    from alts.core.stopping_criteria import StoppingCriteria
    from alts.core.experiment_modules import ExperimentModules
    from alts.core.evaluator import Evaluator
    from alts.core.oracle.oracles import Oracles
    from alts.core.data.data_pools import DataPools

@dataclass
class DTBlueprint(Blueprint):
    repeat: int =100
    time_source: TimeSource = IterationTimeSource()
    process: Process = DataSourceProcess(data_source=NoiseAugmentation(data_source=LineDataSource((2,),(1,)), noise_ratio=2.0))
    data_pools: DataPools = ResultDataPools(result= FlatQueriedDataPool())
    oracles: Oracles = POracles(process=FCFSQueryQueue())

    stopping_criteria: StoppingCriteria = TimeStoppingCriteria(stop_time=300)

    experiment_modules: ExperimentModules = DependencyExperiment(
        query_selector=ResultQuerySelector(
            query_optimizer=NoQueryOptimizer(
                selection_criteria= QueryTestNoSelectionCritera(),
                query_sampler=UniformQuerySampler(num_queries = 4),
            ),
            query_decider=ThresholdQueryDecider(threshold=0.0),
            ),
        dependency_test=SampleTest(
            query_sampler = LastQuerySampler(),
            data_sampler = KDTreeRegionDataSampler(0.05),
            multi_sample_test=KWHMultiSampleTest()
            ),
    )


    evaluators: Iterable[Evaluator] =(LogPValueEvaluator(),)