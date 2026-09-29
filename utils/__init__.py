"""
GSD 工具模块
"""

from .seed import set_seed, worker_init_fn
from .signature import (
    RandomProjection,
    SignQuantizer,
    BCHEncoder,
    HMACBinder,
    SignatureGenerator,
    compute_ber,
    compute_hamming_distance
)
from .metrics import (
    compute_ber,
    compute_normalized_correlation,
    compute_tpr_at_far,
    compute_eer,
    compute_auc,
    SignatureEvaluator,
    EmbeddingEvaluator
)

__all__ = [
    'set_seed',
    'worker_init_fn',
    'RandomProjection',
    'SignQuantizer',
    'BCHEncoder',
    'HMACBinder',
    'SignatureGenerator',
    'compute_ber',
    'compute_hamming_distance',
    'compute_normalized_correlation',
    'compute_tpr_at_far',
    'compute_eer',
    'compute_auc',
    'SignatureEvaluator',
    'EmbeddingEvaluator'
]

