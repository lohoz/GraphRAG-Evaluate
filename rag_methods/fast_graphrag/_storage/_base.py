from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
    final,
)

from scipy.sparse import csr_matrix  # type: ignore

from .._types import GTBlob, GTEdge, GTEmbedding, GTId, GTKey, GTNode, GTValue, TIndex, TScore
from .._utils import logger

from ._namespace import Namespace


@dataclass
class BaseStorage:
    config: Optional[Any] = field()
    namespace: Optional[Namespace] = field(default=None)
    _mode: Optional[Literal["insert", "query"]] = field(init=False, default=None)
    _in_progress: Optional[bool] = field(init=False, default=None)

    def set_in_progress(self, in_progress: bool) -> None:
        self._in_progress = in_progress

    @final
    async def insert_start(self):
        if self._mode == "query":
            logger.info("Switching from query to insert mode.")
            if self._in_progress is not False:
                t = (
                    f"[{self.__class__.__name__}] Cannot being insert before committing query operations."
                    "Committing query operations now."
                )
                logger.error(t)
                await self._query_done()
                self._in_progress = False
        self._mode = "insert"

        if self._in_progress is not True:
            await self._insert_start()

    @final
    async def query_start(self):
        if self._mode == "insert":
            logger.info("Switching from insert to query mode.")
            if self._in_progress is not False:
                t = (
                    f"[{self.__class__.__name__}] Cannot being query before commiting insert operations."
                    "Committing insert operations now."
                )
                logger.error(t)
                await self._insert_done()
                self._in_progress = False
        self._mode = "query"

        if self._in_progress is not True:
            await self._query_start()

    @final
    async def insert_done(self) -> None:
        if self._mode == "query":
            t = f"[{self.__class__.__name__}] Trying to commit insert operations in query mode."
            logger.error(t)
        else:
            if self._in_progress is not False:
                await self._insert_done()
            else:
                logger.warning(f"[{self.__class__.__name__}] No insert operations to commit.")

    @final
    async def query_done(self) -> None:
        if self._mode == "insert":
            t = f"[{self.__class__.__name__}] Trying to commit query operations in insert mode."
            logger.error(t)
        else:
            if self._in_progress is not False:
                await self._query_done()
            else:
                logger.warning(f"[{self.__class__.__name__}] No query operations to commit.")

    async def _insert_start(self):
        """Prepare the storage for inserting."""
        pass

    async def _insert_done(self):
        """Commit the storage operations after inserting."""
        if self._mode == "query":
            logger.error("Trying to commit insert operations in query mode.")

    async def _query_start(self):
        """Prepare the storage for querying."""
        pass

    async def _query_done(self):
        """Release the storage after querying."""
        if self._mode == "insert":
            logger.error("Trying to commit query operations in insert mode.")


####################################################################################################
# Blob Storage
####################################################################################################


@dataclass
class BaseBlobStorage(BaseStorage, Generic[GTBlob]):
    async def get(self) -> Optional[GTBlob]:
        raise NotImplementedError

    async def set(self, blob: GTBlob) -> None:
        raise NotImplementedError


####################################################################################################
# Key-Value Storage
####################################################################################################


@dataclass
class BaseIndexedKeyValueStorage(BaseStorage, Generic[GTKey, GTValue]):
    async def size(self) -> int:
        raise NotImplementedError

    async def get(self, keys: Iterable[GTKey]) -> Iterable[Optional[GTValue]]:
        raise NotImplementedError

    async def get_by_index(self, indices: Iterable[TIndex]) -> Iterable[Optional[GTValue]]:
        raise NotImplementedError

    async def get_index(self, keys: Iterable[GTKey]) -> Iterable[Optional[TIndex]]:
        raise NotImplementedError

    async def upsert(self, keys: Iterable[GTKey], values: Iterable[GTValue]) -> None:
        raise NotImplementedError

    async def upsert_by_index(self, indices: Iterable[TIndex], values: Iterable[GTValue]) -> None:
        raise NotImplementedError

    async def delete(self, keys: Iterable[GTKey]) -> None:
        raise NotImplementedError

    async def delete_by_index(self, indices: Iterable[TIndex]) -> None:
        raise NotImplementedError

    async def mask_new(self, keys: Iterable[GTKey]) -> Iterable[bool]:
        raise NotImplementedError


####################################################################################################
# Vector Storage
####################################################################################################


@dataclass
class BaseVectorStorage(BaseStorage, Generic[GTId, GTEmbedding]):
    embedding_dim: int = field(default=0)

    @property
    def size(self) -> int:
        raise NotImplementedError

    async def get_knn(
        self, embeddings: Iterable[GTEmbedding], top_k: int
    ) -> Tuple[Iterable[Iterable[GTId]], Iterable[Iterable[TScore]]]:
        raise NotImplementedError

    async def upsert(
        self,
        ids: Iterable[GTId],
        embeddings: Iterable[GTEmbedding],
        metadata: Union[Iterable[Dict[str, Any]], None] = None,
    ) -> None:
        raise NotImplementedError

    async def score_all(
        self, embeddings: Iterable[GTEmbedding], top_k: int = 1, threshold: Optional[float] = None
    ) -> csr_matrix:
        """Score all embeddings against the given queries.

        Return a (#queries, #all_embeddings) matrix containing the relevancy scores of each embedding given each query.
        """
        raise NotImplementedError


####################################################################################################
# Graph Storage
####################################################################################################


@dataclass
class BaseGraphStorage(BaseStorage, Generic[GTNode, GTEdge, GTId]):
    async def save_graphml(self, path: str) -> None:
        raise NotImplementedError

    async def node_count(self) -> int:
        raise NotImplementedError

    async def edge_count(self) -> int:
        raise NotImplementedError

    async def get_edge_ids(self) -> Iterable[GTId]:
        raise NotImplementedError

    async def get_node(self, node: Union[GTNode, GTId]) -> Union[Tuple[GTNode, TIndex], Tuple[None, None]]:
        raise NotImplementedError

    async def get_all_edges(self) -> Iterable[GTEdge]:
        raise NotImplementedError

    async def get_edges(
        self, source_node: Union[GTId, TIndex], target_node: Union[GTId, TIndex]
    ) -> Iterable[Tuple[GTEdge, TIndex]]:
        raise NotImplementedError

    async def get_edge_indices(
        self, source_node: Union[GTId, TIndex], target_node: Union[GTId, TIndex]
    ) -> Iterable[TIndex]:
        raise NotImplementedError

    async def get_node_by_index(self, index: TIndex) -> Union[GTNode, None]:
        raise NotImplementedError

    async def get_edge_by_index(self, index: TIndex) -> Union[GTEdge, None]:
        raise NotImplementedError

    async def upsert_node(self, node: GTNode, node_index: Union[TIndex, None]) -> TIndex:
        raise NotImplementedError

    async def upsert_edge(self, edge: GTEdge, edge_index: Union[TIndex, None]) -> TIndex:
        raise NotImplementedError

    async def insert_edges(
        self,
        edges: Optional[Iterable[GTEdge]] = None,
        indices: Optional[Iterable[Tuple[TIndex, TIndex]]] = None,
        attrs: Optional[Mapping[str, Iterable[Any]]] = None,
    ) -> List[TIndex]:
        raise NotImplementedError

    async def are_neighbours(self, source_node: Union[GTId, TIndex], target_node: Union[GTId, TIndex]) -> bool:
        raise NotImplementedError

    async def delete_edges_by_index(self, indices: Iterable[TIndex]) -> None:
        raise NotImplementedError

    async def get_entities_to_relationships_map(self) -> csr_matrix:
        raise NotImplementedError

    async def get_relationships_to_chunks_map(
        self, key: str, key_to_index_fn: Callable[[Iterable[GTKey]], Awaitable[Iterable[TIndex]]], num_chunks: int
    ) -> csr_matrix:
        raise NotImplementedError

    async def get_relationships_attrs(self, key: str) -> List[List[Any]]:
        raise NotImplementedError

    async def score_nodes(self, initial_weights: Optional[csr_matrix]) -> csr_matrix:
        """Score nodes based on the initial weights."""
        raise NotImplementedError
