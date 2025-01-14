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
        CompositeDeltaFunction, DistanceMatrix
from delta.cluster import Clustering, FlatClustering, evaluate_distances

from delta.features import get_rfe_features
from delta.graphics import Dendrogram
from delta.util import compare_pairwise, get_triangle_values, ComposerDescriber, DocumentDescriber, EpochDescriber, Metadata, ModeDescriber, TableDocumentDescriber, TsvDocumentDescriber

__all__ = [ "Corpus", "FeatureGenerator", "LETTERS_PATTERN", "WORD_PATTERN",
           "functions", "Normalization", "normalization",
           "DeltaFunction", "PDistDeltaFunction",
           "MetricDeltaFunction", "CompositeDeltaFunction",
           "Clustering", "FlatClustering",
           "get_rfe_features", "Dendrogram",
           "compare_pairwise", "Metadata", "get_triangle_values", "ComposerDescriber", "DocumentDescriber", "EpochDescriber", "ModeDescriber", "TsvDocumentDescriber"]

try:
        from delta.cluster import KMedoidsClustering, KMedoidsClustering_distances
        __all__.append("KMedoidsClustering")
except (ImportError, NameError):
        warn("KMedoidsClustering not available")
