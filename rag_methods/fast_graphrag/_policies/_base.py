from dataclasses import dataclass, field
from typing import Any, Generic, Iterable, Tuple, Type

from scipy.sparse import csr_matrix

from .._llm._llm_openai import BaseLLMService
from .._storage._base import BaseGraphStorage
from .._types import GTEdge, GTId, GTNode, TIndex


@dataclass
class BasePolicy:
    config: Any = field()


####################################################################################################
# GRAPH UPSERT POLICIES
####################################################################################################


@dataclass
class BaseNodeUpsertPolicy(BasePolicy, Generic[GTNode, GTId]):
    async def __call__(
        self, llm: BaseLLMService, target: BaseGraphStorage[GTNode, GTEdge, GTId], source_nodes: Iterable[GTNode]
    ) -> Tuple[BaseGraphStorage[GTNode, GTEdge, GTId], Iterable[Tuple[TIndex, GTNode]]]:
        raise NotImplementedError


@dataclass
class BaseEdgeUpsertPolicy(BasePolicy, Generic[GTEdge, GTId]):
    async def __call__(
        self, llm: BaseLLMService, target: BaseGraphStorage[GTNode, GTEdge, GTId], source_edges: Iterable[GTEdge]
    ) -> Tuple[BaseGraphStorage[GTNode, GTEdge, GTId], Iterable[Tuple[TIndex, GTEdge]]]:
        raise NotImplementedError


@dataclass
class BaseGraphUpsertPolicy(BasePolicy, Generic[GTNode, GTEdge, GTId]):
    nodes_upsert_cls: Type[BaseNodeUpsertPolicy[GTNode, GTId]] = field()
    edges_upsert_cls: Type[BaseEdgeUpsertPolicy[GTEdge, GTId]] = field()
    _nodes_upsert: BaseNodeUpsertPolicy[GTNode, GTId] = field(init=False)
    _edges_upsert: BaseEdgeUpsertPolicy[GTEdge, GTId] = field(init=False)

    def __post_init__(self):
        self._nodes_upsert = self.nodes_upsert_cls(self.config)
        self._edges_upsert = self.edges_upsert_cls(self.config)

    async def __call__(
        self,
        llm: BaseLLMService,
        target: BaseGraphStorage[GTNode, GTEdge, GTId],
        source_nodes: Iterable[GTNode],
        source_edges: Iterable[GTEdge],
    ) -> Tuple[
        BaseGraphStorage[GTNode, GTEdge, GTId],
        Iterable[Tuple[TIndex, GTNode]],
        Iterable[Tuple[TIndex, GTEdge]],
    ]:
        raise NotImplementedError


####################################################################################################
# RANKING POLICIES
####################################################################################################


class BaseRankingPolicy(BasePolicy):
    def __call__(self, scores: csr_matrix) -> csr_matrix:
        assert scores.shape[0] == 1, "Ranking policies only supports batch size of 1"
        return scores
