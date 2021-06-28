# -*- coding: utf-8 -*-

"""
pydelta library
---------------

Stylometrics in Python
"""
import warnings

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError: # < Python 3.8
    import importlib_metadata

__title__ = 'delta'
try:
    __version__ = importlib_metadata.version(__name__)
except Exception as e:
    __version__ = None
    warnings.warn(f'Could not determine version: {e}', ImportWarning, source=e)
__author__ = 'Fotis Jannidis, Thorsten Vitt'

from warnings import warn
from delta.corpus import Corpus, FeatureGenerator, LETTERS_PATTERN, WORD_PATTERN
from delta.deltas import registry as functions, normalization, Normalization, \
        DeltaFunction, PDistDeltaFunction, MetricDeltaFunction, \
        CompositeDeltaFunction
from delta.cluster import Clustering, FlatClustering

from delta.features import get_rfe_features
from delta.graphics import Dendrogram
from delta.util import compare_pairwise, Metadata, TableDocumentDescriber

__all__ = [ "Corpus", "FeatureGenerator", "LETTERS_PATTERN", "WORD_PATTERN",
           "functions", "Normalization", "normalization",
           "DeltaFunction", "PDistDeltaFunction",
           "MetricDeltaFunction", "CompositeDeltaFunction",
           "Clustering", "FlatClustering",
           "get_rfe_features", "Dendrogram",
           "compare_pairwise", "Metadata", "TableDocumentDescriber" ]

try:
        from delta.cluster import KMedoidsClustering
        __all__.append("KMedoidsClustering")
except (ImportError, NameError):
        warn("KMedoidsClustering not available")
