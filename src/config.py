from dataclasses import dataclass
from typing import Optional


@dataclass
class SpladeConfig:
    """Configuration for SPLADE sparse embedding provider."""
    model: str = "naver/splade-cocondenser-ensembledistil"
    device: str = "cpu"


@dataclass
class SparseEmbeddingConfig:
    """Configuration for sparse embedding providers."""
    provider: str = "splade"
    splade: Optional[SpladeConfig] = None