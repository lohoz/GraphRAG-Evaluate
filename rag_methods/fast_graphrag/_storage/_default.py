__all__ = [
    "DefaultVectorStorage",
    "DefaultVectorStorageConfig",
    "DefaultBlobStorage",
    "DefaultIndexedKeyValueStorage",
    "DefaultGraphStorage",
    "DefaultGraphStorageConfig",
]

from .._storage._blob_pickle import PickleBlobStorage
from .._storage._gdb_igraph import IGraphStorage, IGraphStorageConfig
from .._storage._ikv_pickle import PickleIndexedKeyValueStorage
from .._storage._vdb_hnswlib import HNSWVectorStorage, HNSWVectorStorageConfig
from .._types import GTBlob, GTEdge, GTEmbedding, GTId, GTKey, GTNode, GTValue


# Storage
class DefaultVectorStorage(HNSWVectorStorage[GTId, GTEmbedding]):
    pass
class DefaultVectorStorageConfig(HNSWVectorStorageConfig):
    pass
class DefaultBlobStorage(PickleBlobStorage[GTBlob]):
    pass
class DefaultIndexedKeyValueStorage(PickleIndexedKeyValueStorage[GTKey, GTValue]):
    pass
class DefaultGraphStorage(IGraphStorage[GTNode, GTEdge, GTId]):
    pass
class DefaultGraphStorageConfig(IGraphStorageConfig[GTNode, GTEdge]):
    pass
