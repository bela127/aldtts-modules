from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

import numpy as np

from alts.core.query.query_decider import QueryDecider

if TYPE_CHECKING:
    from typing import Tuple
    from nptyping import NDArray, Number, Shape

@dataclass    
class UnpackAllQueryDecider(QueryDecider):
    def decide(self, query_candidates: NDArray[Shape["query_nr, ... query_dims"], Number], scores: NDArray[Shape["query_nr, [query_score]"], Number]) -> Tuple[bool, NDArray[Shape["query_nr, ... query_dims"], Number]]:
        queries = query_candidates.reshape(query_candidates.shape[0]*query_candidates.shape[1],-1)
        return True, queries
