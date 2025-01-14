# -*- encoding: utf-8 -*-
"""
This module contains the actual delta measures.

Normalizations
==============

A *normalization* is a function that works on a :class:`Corpus` and returns a
somewhat normalized version of that corpus. Each normalization has the
following additional attributes:

* name – an identifier for the normalization, usually the function name
* title – an optional, human-readable name for the normalization

Each normalization leaves its name in the 'normalizations' field of the corpus'
:class:`Metadata`.

All available normalizations need to be registered to the normalization
registry.


Delta Functions
===============

A *delta function* takes a :class:`Corpus` and creates a :class:`Distances`
table from that. Each delta function has the following properties:

* descriptor – a systematic descriptor of the distance function. For simple
    delta functions (see below), this is simply the name. For composite distance
    functions, this starts with the name of a simple delta function and is followed
    by a list of normalizations (in order) that are applied to the corpus before
    applying the distance function.
* name – a unique name for the distance function
* title – an optional, human-readable name for the distance function.


Simple Delta Functions
----------------------

Simple delta functions are functions that

"""
from __future__ import annotations
import logging
from enum import Enum
from typing import Tuple

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
from scipy import linalg
try:
    from scipy.misc import comb
except ImportError:
    from scipy.special import comb
from functools import update_wrapper, cached_property
from .util import Metadata, compare_pairwise, append_dataframe_to_zip, get_triangle_values, map_to_index_levels
from .corpus import Corpus
from textwrap import dedent
from sklearn.metrics import pairwise_distances

sep = '-'   # separates parts of a descriptor



class _FunctionRegistry:
    """
    The registry of normalizations and delta functions.

    Usually, functions register themselves when they are created using one of the
    base classes or decorators (see below), they can be accessed using the registry's
    methods :meth:`normalization` and :meth:`delta`, or using subscription or
    attribute access.
    """

    def __init__(self):
        self.normalizations = {}    # name -> normalization function
        self.deltas = {}            # descriptor -> normalization function
        self.aliases = {}           # name -> normalization function

    @staticmethod
    def get_name(f):
        try:
            return f.name
        except:
            return f.__name__

    def add_normalization(self, f):
        """
        Registers the normalization _f_.

        Args:
            f (Normalization): The normalization to register.
        """
        name = self.get_name(f)
        if name in self.normalizations:
            logger.warning("Registering %s as %s, replacing existing function with this name", f, name)
        self.normalizations[name] = f

    def add_delta(self, f):
        """
        Registers the given Delta function.

        Args:
            f (DeltaFunction): The delta function to register.
        """
        self.deltas[f.descriptor] = f
        if f.name != f.descriptor:
            self.aliases[f.name] = f


    def normalization(self, name):
        """
        Returns the normalization identified by the name, or raises an :class:`IndexError` if
        it has not been registered.

        Args:
            name (str): The name of the normalization to retrieve.
        Returns:
            Normalization
        """
        return self.normalizations[name]

    def delta(self, descriptor, register=False):
        """
        Returns the delta function identified by the given descriptor or alias.
        If you pass in a composite descriptor and the delta function has _not_
        been registered yet, this tries to create a
        :class:`CompositeDeltaFunction` from the descriptor.

        Args:
            descriptor (str): Descriptor for the delta function to retrieve or create.
            register (bool): When creating a composite delta function,register
                it for future access.
        Returns:
            DeltaFunction: The requested delta function.
        """
        if descriptor in self.deltas:
            return self.deltas[descriptor]
        elif descriptor in self.aliases:
            return self.aliases[descriptor]
        elif sep in descriptor:
            return CompositeDeltaFunction(descriptor, register=register)
        else:
            raise IndexError("Delta function '%s' has not been found, "
                    "and it is not a composite descriptor, either." % descriptor)

    def __getitem__(self, index):
        try:
            return self.normalization(index)
        except KeyError:
            return self.delta(index)

    def __getattr__(self, index):
        try:
            return self[index]
        except IndexError as error:
            raise AttributeError(error)

    def __dir__(self):
        attributes = list(super().__dir__())
        attributes.extend(self.normalizations.keys())
        attributes.extend((name for name in self.deltas.keys() if name.isidentifier()))
        attributes.extend(self.aliases.keys())
        return attributes

    def __str__(self):
        return dedent(
            """
            {} Delta Functions:
            ------------------
            {}

            {} Normalizations:
            -----------------
            {}
            """).format(len(self.deltas),
                    '\n'.join(str(d) for d in self.deltas.values()),
                    len(self.normalizations),
                    '\n'.join(str(n) for n in self.normalizations.values()))

    def _repr_html_(self):
        return "<h4>Delta Functions</h4><ol><li>" + \
            '</li><li>'.join(d._repr_html_() for d in self.deltas.values()) + \
            '</ol><h4>Normalizations</h4><ol><li>' + \
            '</li><li>'.join(str(n) for n in self.normalizations.values())


registry = _FunctionRegistry()

class Normalization:
    """
    Wrapper for normalizations.
    """

    def __init__(self, f, name=None, title=None, register=True):
        self.normalize = f
        if name is None:
            name = f.__name__
        if title is None:
            title = name
        self.name = name
        self.title = title
        update_wrapper(self, f)

        if register:
            registry.add_normalization(self)

    def __call__(self, corpus, *args, **kwargs):
        return Corpus(self.normalize(corpus, *args, **kwargs),
                document_describer=corpus.document_describer,
                metadata=corpus.metadata, normalization=(self.name,))

    def __str__(self):
        result = self.name
        if self.title != self.name:
            result += ' ('+self.title+')'
        # add docstring?
        return result

    def _repr_html_(self):
        result = '<code>{}</code>'.format(self.name)
        if self.title != self.name:
            result += ' <em>{}</em>'.format(self.title)
        return result


def normalization(*args, **kwargs):
    """
    Decorator that creates a :class:`Normalization` from a function or
    (callable) object. Can be used without or with keyword arguments:

        name (str): Name (identifier) for the normalization. By default, the function's name is used.
        title (str): Human-readable title for the normalization.
    """
    name = kwargs['name']   if 'name' in kwargs  else None
    title = kwargs['title'] if 'title' in kwargs else None

    def createNormalization(f):
        wrapper = Normalization(f, name=name, title=title)
        return update_wrapper(wrapper, f)
    if args and callable(args[0]):
        return createNormalization(args[0])
    else:
        return createNormalization

class DeltaFunction:
    """
    Abstract base class of a delta function.

    To define a delta function, you have various options:

    1. subclass DeltaFunction and override its :meth:`__call__` method with something that directly handles a :class:`Corpus`.
    2. subclass DeltaFunction and override its :meth:`distance` method with a distance function
    3. instantiate DeltaFunction and pass it a distance function, or use the :func:`delta` decorator
    4. use one of the subclasses
    """

    def __init__(self, f=None, descriptor=None, name=None, title=None, register=True):
        """
        Creates a custom delta function.

        Args:
            f (function): a distance function that calculates the difference
                between two feature vectors and returns a float. If passed,
                this will be used for the implementation.
            name (str): The name/id of this function. Can be inferred from
                `f` or `descriptor`.
            descriptor (str): The descriptor to identify this function.
            title (str): A human-readable title for this function.
            register (bool): If true (default), register this delta function
                with the function registry on instantiation.
        """
        if f is not None:
            if name is None:
                name = f.__name__
            self.distance = f
            update_wrapper(self, f)

        if name is None:
            if descriptor is None:
                name = type(self).__name__
            else:
                name = descriptor

        if descriptor is None:
            descriptor = name

        if title is None:
            title = name

        self.name = name
        self.descriptor = descriptor
        self.title = title
        logger.debug("Created a %s with name=%s, descriptor=%s, title=%s",
                type(self), name, descriptor, title)
        if register:
            self.register()

    def __str__(self):
        result = self.name
        if self.title != self.name:
            result += ' "'+self.title+'"'
        if self.descriptor != self.name:
            result += ' = ' + self.descriptor
        return result

    def _repr_html_(self):
        result = '<code>{}</code>'.format(self.name)
        if self.title != self.name:
            result += ' <em>{}</em>'.format(self.title)
        if self.descriptor != self.name:
            result += ' = ' + self.descriptor
        return result

    @staticmethod
    def distance(u, v, *args, **kwargs):
        """
        Calculate a distance between two feature vectors.

        This is an abstract method, you must either inherit from DeltaFunction
        and override distance or assign a function in order to use this.

        Args:
            u, v (pandas.Series): The documents to compare.
            *args, **kwargs: Passed through from the caller
        Returns:
            float: Distance between the documents.
        Raises:
            NotImplementedError if no implementation is provided.
        """
        raise NotImplementedError("You need to either override DeltaFunction"
                                  "and override distance or assign a function"
                                  "to distance")

    def register(self):
        """Registers this delta function with the global function registry."""
        registry.add_delta(self)

    # def iterate_distance(self, corpus, *args, **kwargs):
    #     """
    #     Calculates the distance matrix for the given corpus.
    #
    #     The default implementation will iterate over all pairwise combinations
    #     of the documents in the given corpus and call :meth:`distance` on each
    #     pair, passing on the additional arguments.
    #
    #     Clients may want to use :meth:`__call__` instead, i.e. they want to call
    #     this object as a function.
    #
    #     Args:
    #         corpus (Corpus): feature matrix for which to calculate the distance
    #         *args, **kwargs: further arguments for the matrix
    #     Returns:
    #         pandas.DataFrame: square dataframe containing pairwise distances.
    #             The default implementation will return a matrix that has zeros
    #             on the diagonal and the lower triangle a mirror of the upper
    #             triangle.
    #     """
    #     df = pd.DataFrame(index=corpus.index, columns=corpus.index)
    #     for a, b in combinations(df.index, 2):
    #         delta = self.distance(corpus.loc[a,:], corpus.loc[b,:], *args, **kwargs)
    #         df.at[a, b] = delta
    #         df.at[b, a] = delta
    #     return df.fillna(0)

    def pairwise_distance(self, corpus, **kwargs):
        distances_long = ssd.pdist(corpus, self.distance, **kwargs)
        distances_square = ssd.squareform(distances_long)
        return pd.DataFrame(distances_square, index=corpus.index, columns=corpus.index)

    def create_result(self, df, corpus):
        """
        Wraps a square dataframe to a DistanceMatrix, adding appropriate
        metadata from corpus and this delta function.

        Args:
            df (pandas.DataFrame): Distance matrix like created by :meth:`iterate_distance`
            corpus (Corpus): source feature matrix
        Returns:
            DistanceMatrix: df as values, appropriate metadata
        """
        return DistanceMatrix(df, metadata=corpus.metadata, corpus=corpus,
                              delta=self.name,
                              delta_descriptor=self.descriptor,
                              delta_title=self.title)

    def __call__(self, corpus):
        """
        Calculates the distance matrix.

        Args:
            corpus (Corpus): The feature matrix that is the basis for the distance
        Returns:
            DistanceMatrix: Pairwise distances between the documents
        """
        return self.create_result(self.pairwise_distance(corpus), corpus)


    def prepare(self, corpus):
        """
        Return the corpus prepared for the metric, if applicable.

        Many delta functions consist of a preparation step that normalizes
        the corpus in some way and a relatively standard distance metric
        that is one of the built-in distance metrics of scikit-learn or
        scipy.

        If a specific delta variant supports this, it should expose a metric
        attribute set to a string or a callable that implements the metric,
        and possibly override this method in order to perform the preparation
        steps.

        The default implementation simply returns the corpus as-is.

        Raises:
            NotImplementedError if there is no metric
        """
        if hasattr(self, 'metric'):
            return corpus
        else:
            raise NotImplementedError("This delta function does not support a standard metric.")


class _LinearDelta(DeltaFunction):

    @staticmethod
    def diversity(values):
        """
        calculates the spread or diversity (wikipedia) of a laplace distribution of values
        see Argamon's Interpreting Burrow's Delta p. 137 and
        http://en.wikipedia.org/wiki/Laplace_distribution
        couldn't find a ready-made solution in the python libraries

        :param values: a pd.Series of values
        """
        return np.abs(values - values.median()).sum() / values.size

    @staticmethod
    def distance(u, v, *args, diversities=None):
        dist = (np.abs(u - v) / diversities).sum()
        return dist

    def __call__(self, corpus):
        diversities = corpus.apply(_LinearDelta.diversity)
        matrix = self.pairwise_distance(corpus, diversities=diversities)
        return self.create_result(matrix, corpus)


class PreprocessingDeltaFunction(DeltaFunction):

    def __init__(self, distance_function, prep_function, descriptor=None,
                 name=None, title=None, register=True):
        super().__init__(f=distance_function, descriptor=descriptor, name=name,
                         title=title, register=register)
        self.prep_function = prep_function

    @staticmethod
    def prep_function(corpus):
        return dict()

    def __call__(self, corpus):
        kwargs = self.prep_function(corpus)
        logger.info("Preprocessor delivered %s", kwargs)
        matrix = self.pairwise_distance(corpus, **kwargs)
        return self.create_result(matrix, corpus)

_LinearDelta(descriptor="linear", name="Linear Delta")

def _prep_linear(corpus):
    return { 'diversities': corpus.apply(_LinearDelta.diversity) }

PreprocessingDeltaFunction(_LinearDelta.distance, _prep_linear, descriptor="linear2")

def _classic_delta(a, b, stds, n
                   ):
    """
    Burrow's Classic Delta, from pydelta 0.1
    """
    return (np.abs(a - b) / stds).sum() / n
def _prep_classic_delta(corpus):
    return { 'stds': corpus.std(), 'n': corpus.columns.size }
PreprocessingDeltaFunction(_classic_delta, _prep_classic_delta, 'burrows2')

class CompositeDeltaFunction(DeltaFunction):
    """
    A composite delta function consists of a *basis* (which is another delta
    function) and a list of *normalizations*. It first transforms the corpus
    via all the given normalizations in order, and then runs the basis on the
    result.
    """

    def __init__(self, descriptor, name=None, title=None, register=True):
        """
        Creates a new composite delta function.

        Args:
            descriptor (str): Formally defines this delta function. First the
                name of an existing, registered distance function, then, separated
                by ``-``, the names of normalizations to run, in order.
            name (str): Name by which this delta function is registered, in
                addition to the descriptor
            title (str): human-readable title
            register (bool): If true (the default), register this delta
                function on creation
        """
        items = descriptor.split(sep)
        self.basis = registry.deltas[items[0]]
        if hasattr(self.basis, 'metric'):
            self.metric = self.basis.metric
        else:
            self.metric = self.basis.distance_function
        del items[0]
        self.normalizations = [registry.normalizations[n] for n in items]
        super().__init__(self, descriptor, name, title, register)

    def prepare(self, corpus):
        for normalization in self.normalizations:
            corpus = normalization(corpus)
        return corpus

    def __call__(self, corpus):
        return self.create_result(self.basis(self.prepare(corpus)), corpus)


class PDistDeltaFunction(DeltaFunction):
    """
    Wraps one of the metrics implemented by :func:`ssd.pdist` as a delta function.

    Warning:
        You should use MetricDeltaFunction instead.
    """
    def __init__(self, metric, name=None, title=None, register=True, scale=False, **kwargs):
        """
        Args:
            metric (str):  The metric that should be called via ssd.pdist
            name (str):    Name / Descriptor for the delta function, if None, metric is used
            title (str):   Human-Readable Title
            register (bool): If false, don't register this with the registry
            **kwargs:      passed on to :func:`ssd.pdist`
        """
        logger.warning("Prefer MetricsDeltaFunction to PDistDeltaFunction.")

        self.metric = metric
        self.kwargs = kwargs
        if name is None:
            name = metric
        if title is None:
            title = name.title() + " Distance"
        self.scale = scale

        super().__init__(descriptor=name, name=name, title=title, register=register)


    def __call__(self, corpus):
        df = pd.DataFrame(index=corpus.index, columns=corpus.index,
                          data=ssd.squareform(ssd.pdist(corpus, self.metric,
                                                        **self.kwargs)))
        if self.scale:
            df = df / corpus.columns.size
        return self.create_result(df, corpus)


class MetricDeltaFunction(DeltaFunction):
    """
    Distance functions based on scikit-learn's :func:`sklearn.metric.pairwise_distances`.
    """

    def __init__(self, metric, name=None, title=None, register=True, scale=False, fix_symmetry=True, **kwargs):
        """
        Args:
            metric (str):  The metric that should be called via sklearn.metric.pairwise_distances
            name (str):    Name / Descriptor for the delta function, if None, metric is used
            title (str):   Human-Readable Title
            register (bool): If false, don't register this with the registry
            scale (bool):  Scale by number of features
            fix_symmetry:  Force the resulting matrix to be symmetric
            **kwargs:      passed on to :func:`ssd.pdist`

        Note:
            :func:`sklearn.metric.pairwise_distances` fast, but the result may
            not be exactly symmetric. The `fix_symmetry` option enforces
            symmetry by mirroring the lower-left triangle after calculating
            distances so, e.g., scipy clustering won't complain.
        """
        self.metric = metric
        self.scale = scale
        self.fix_symmetry = fix_symmetry
        self.kwargs = kwargs
        if name is None:
            name = metric
        if title is None:
            title = name.title() + " Distance"
        super().__init__(descriptor=name, name=name, title=title, register=register)

    def __call__(self, corpus):
        dm = pairwise_distances(corpus, metric=self.metric, n_jobs=-1, **self.kwargs)
        if self.fix_symmetry:
            dm = np.tril(dm, -1)
            dm = dm + dm.T
        df = pd.DataFrame(data=dm, index=corpus.index, columns=corpus.index)
        if self.scale:
            df = df / corpus.columns.size
        np.fill_diagonal(df.values, 0)   # rounding errors may lead to validation bugs
        return self.create_result(df, corpus)

class DeltaColumns(Enum):
    """
    Columns in the delta table.
    """
    DELTA = 'Delta'
    AUTHOR1 = 'Composer1'
    AUTHOR2 = 'Composer2'
    TEXT1 = 'Document1'
    TEXT2 = 'Document2'

class DistanceMatrix(pd.DataFrame):

    _metadata = ['metadata']

    """
    A distance matrix is the result of applying a :class:`DeltaFunction` to a
    :class:`Corpus`.

    Args:
        df (pandas.DataFrame): Values for the distance matrix to be created
        copy_from (DistanceMatrix): copy metadata etc. from this distance matrix.
            If ``df`` is a DistanceMatrix, it will be used as copy_from value
        metadata (Metadata): Metadata record to start with
        document_describer (DocumentDescriber): Describes the documents, i.e.,
            labels and ground truth
        corpus (Corpus): Try to take document describer from here
        **kwargs: Additional metadata
    """

    def __init__(self, df, copy_from=None, metadata=None, corpus=None,
                 document_describer=None, **kwargs):
        super().__init__(df)
        self.document_describer = None
        if isinstance(df, DistanceMatrix) and copy_from is None:
            copy_from = df
        if copy_from is not None:
            self.document_describer = copy_from.document_describer
            self.metadata = copy_from.metadata
            if metadata is not None:
                self.metadata.update(metadata)
            self.metadata.update(kwargs)
        else:
            self.metadata = Metadata(metadata, **kwargs)
        if document_describer is not None:
            self.document_describer = document_describer
        elif corpus is not None:
            self.document_describer = corpus.document_describer

    @cached_property
    def group_index_level(self) -> pd.Series:
        """Converts the index into a Series of group names."""
        group_a, = map_to_index_levels(self.index, self.document_describer.group_name)
        return group_a.rename('group')

    @cached_property
    def item_index_level(self) -> pd.Series:
        """Converts the index into a Series of item names."""
        group_a, = map_to_index_levels(self.index, self.document_describer.item_name)
        return group_a.rename('item')

    @cached_property
    def long_format(self) -> pd.Series:
        """Like calling :meth:`delta_values`."""
        return self.delta_values()

    @cached_property
    def long_format_group_index_levels(self) -> Tuple[pd.Series, pd.Series]:
        """Returns each of the two index levels of :meth:`delta_values` as a series with the same index but where the
        values of a given level have been replaced with their group (i.e. corpus) name.
        """
        deltas = self.long_format
        group_a, group_b = map_to_index_levels(deltas.index, self.document_describer.group_name)
        return group_a, group_b

    @cached_property
    def long_format_item_index_levels(self) -> Tuple[pd.Series, pd.Series]:
        """Returns each of the two index levels of :meth:`delta_values` as a series with the same index but where the
        values of a given level have been replaced with their group (i.e. corpus) name.
        """
        deltas = self.long_format
        item_a, item_b = map_to_index_levels(deltas.index, self.document_describer.item_name)
        return item_a, item_b

    @classmethod
    def from_csv(cls, filename):
        """
        Loads a distance matrix from a cross-table style csv file.
        """
        df = pd.DataFrame.from_csv(filename)
        md = Metadata.load(filename)
        return cls(df, md)

    def save(self, filename):
        self.to_csv(filename)
        self.metadata.save(filename)

    def save_to_zip(self, filename, zip_path):
        """Append the corpus to the ZIP file under the given path."""
        append_dataframe_to_zip(self, filename, zip_path)
        self.metadata.save_to_zip(filename, zip_path)

    def _remove_duplicates(self, transpose=False, check=True):
        """
        Returns a DistanceMatrix that has only the upper right triangle filled,
        ie contains only the unique meaningful values.
        """
        df = self.T if transpose else self
        result = DistanceMatrix(df.where(np.triu(np.ones(self.shape, dtype=bool), k=1)), copy_from=self)
        if check and result.isna().sum().sum() > 0 and self.notna().sum().sum() > 0:
            return self._remove_duplicates(transpose=not transpose, check=False)
        return result

    def _delta_values(self, transpose=False, check=True):
        r"""
        Converts the given n×n Delta matrix to a :math:`\binom{n}{2}` long
        series of distinct delta values – i.e. duplicates from the upper
        triangle and zeros from the diagonal are removed.

        Args:
            transpose: if True, transpose the dataframe first, i.e. use the upper right triangle
            check: if True and if the result does not contain any non-null value, try the other
                   option for transpose.
        """
        result = self._remove_duplicates(transpose, check).stack()
        result.name = self.metadata.get('delta')
        return result

    def delta_values(self, transpose=False, check=None, without_diagonal=True) -> pd.Series:
        df = self.T if transpose else self
        return get_triangle_values(df, offset=int(without_diagonal), name=self.metadata.get('delta'))

    def new_data(self, data, **metadata):
        """
        Wraps the given `DataFrame` with metadata from this DistanceMatrix object.

        Args:
            data (pandas.DataFrame): Raw data that is derived by, e.g., pandas filter operations
            **metadata: Metadata fields that should be changed / modified
        """
        return DistanceMatrix(data, copy_from=self, **metadata)





    def delta_values_df(self):
        """
        Returns an stacked form of the given delta table along with
        additional metadata. Assumes delta is symmetric.

        The dataframe returned has the columns Author1, Author2, Text1, Text2,
        and Delta, it has an entry for every unique combination of texts
        """
        deltas = self.long_format
        group_a, group_b = self.long_format_group_index_levels
        item_a, item_b = self.long_format_item_index_levels
        result = pd.concat([
            deltas,
            group_a.rename(DeltaColumns.AUTHOR1.value),
            group_b.rename(DeltaColumns.AUTHOR2.value),
            item_a.rename(DeltaColumns.TEXT1.value),
            item_b.rename(DeltaColumns.TEXT2.value)
        ], axis=1)
        return result.rename(columns={result.columns[0]: DeltaColumns.DELTA.value})

    def f_ratio(self):
        """
        Calculates the (normalized) F-ratio over the distance matrix, according
        to Heeringa et al.

        Checks whether the distances within a group (i.e., texts with the same
        author) are much smaller thant the distances between groups
        """
        return f_ratio(self)

    def fisher_ld(self):
        """
        Calculates Fisher's Linear Discriminant for the distance matrix.

        cf. Heeringa et al.
        """
        return fisher_ld(self)

    def z_scores(self):
        """
        Returns a distance matrix with the distances standardized using z-scores
        """
        deltas = self.long_format
        return DistanceMatrix((self - deltas.mean()) / np.std(deltas, axis=0),
                              metadata=self.metadata,
                              document_describer=self.document_describer,
                              distance_normalization='z-score')

    def partition(self, square=False):
        """
        Splits this distance matrix into two sparse halves: the first contains
        only the differences between documents that are in the same group
        ('in-group'), the second only the differences between documents that
        are in different groups.

        Group associations are created according to the
        :class:`DocumentDescriber`.

        Returns:
            (DistanceMatrix, DistanceMatrix): (in_group, out_group)
        """
        if square:
            raise NotImplementedError("partition available only in (non-redundant) long format")
        deltas = self.long_format
        group_a, group_b = self.long_format_group_index_levels
        in_group = group_a == group_b
        same = deltas[in_group]
        diff = deltas[~in_group]
        return (same, diff)

    def simple_score(self):
        """
        Simple delta quality score for the given delta matrix:

        The difference between the means of the standardized differences
        between works of different authors and works of the same author; i.e.
        different authors are considered *score* standard deviations more
        different than equal authors.
        """
        return simple_score(self)


    def evaluate(self):
        """
        Returns:
            pandas.Series: All scores implemented for distance matrixes
        """
        result = pd.Series(dtype='float64')
        result["F-Ratio"] = self.f_ratio()
        result["Fisher's LD"] = self.fisher_ld()
        result["Simple Score"] = self.simple_score()
        return result

    def compare_with(self, doc_metadata, comparisons=None, join='inner'):
        """
        Compare the distance matrix value with values calculated from the given document metadata table.

        Args:
            doc_metadata (pd.DataFrame): a dataframe with one row per document and arbitrary columns
            comparisons: see `compare_pairwise`
            join (str): inner (the default) or outer, if outer, keep pairs for which we have neither metadata nor comparisons.

        Returns:
            a dataframe with a row for each pairwise document combination (as in `DistanceMatrix.delta_values`).
            The first column will contain the delta values, subsequent columns the metadata comparisons.
        """
        return pd.concat([self.long_format, compare_pairwise(doc_metadata, comparisons)], join=join, axis=1)

    def __hash__(self):
        return hash(tuple(pd.util.hash_pandas_object(self)))



################# Now a bunch of normalizations:

@normalization(title="Z-Score")
def z_score(corpus):
    """Normalizes the corpus to the z-scores."""
    return (corpus - corpus.mean()) / np.std(corpus, axis=0)

@normalization
def eder_std(corpus):
    """
    Returns a copy of this corpus that is normalized using Eder's normalization.
    This multiplies each entry with :math:`\\frac{n-n_i+1}{n}`
    """
    n = corpus.columns.size
    ed = pd.Series(range(n, 0, -1), index=corpus.columns) / n
    return corpus.apply(lambda f: f*ed, axis=1)

@normalization
def binarize(corpus):
    """
    Returns a copy of this corpus in which the word frequencies are
    normalized to be either 0 (word is not present in the document) or 1.
    """
    df = corpus.copy()
    df[df > 0] = 1
    metadata = corpus.metadata
    del metadata["frequencies"]
    metadata["binarized"] = True
    return Corpus(corpus=df, metadata=metadata)

@normalization
def length_normalized(corpus):
    """
    Returns a copy of this corpus in which the frequency vectors
    have been length-normalized.
    """
    return corpus / corpus.apply(linalg.norm)

@normalization
def diversity_scaled(corpus):
    """
    Returns a copy of this corpus which has been scaled by the diversity argument from the Laplace distribution.
    """
    def diversity(values):
        return np.abs(values - values.median()).sum() / values.size
    return corpus / corpus.apply(diversity)

@normalization
def sqrt(corpus):
    return corpus.apply(np.sqrt)

@normalization
def clamp(corpus, lower_bound=-1, upper_bound=1):
    clamped = corpus.copy()
    clamped[clamped < lower_bound] = lower_bound
    clamped[clamped > upper_bound] = upper_bound
    return clamped

@normalization
def ternarize(corpus, lower_bound=-0.43, upper_bound=0.43):
    ternarized = corpus.copy()
    lower = corpus < lower_bound
    ternarized[lower] = -1
    ternarized[~lower & (corpus < upper_bound)] = 0
    ternarized[corpus > upper_bound] = +1
    return ternarized

################ Here come the deltas

MetricDeltaFunction("cityblock", "manhattan", title="Manhattan Distance", scale=True)
MetricDeltaFunction("euclidean")
MetricDeltaFunction("sqeuclidean", title="Squared Euclidean Distance")
MetricDeltaFunction("cosine")
MetricDeltaFunction("canberra")
MetricDeltaFunction("braycurtis", title="Bray-Curtis Distance")
MetricDeltaFunction("correlation")
MetricDeltaFunction("chebyshev")

CompositeDeltaFunction("manhattan-z_score", "burrows", "Burrows' Delta")
CompositeDeltaFunction("sqeuclidean-z_score", "quadratic", "Quadratic Delta")
CompositeDeltaFunction("manhattan-z_score-eder_std", "eder", "Eder's Delta")
CompositeDeltaFunction("manhattan-sqrt", "eder_simple", "Eder's Simple")
CompositeDeltaFunction("cosine-z_score", "cosine_delta", "Cosine Delta")

# TODO hoover # rotated # pielström


def _f_ratio(delta_values_df):

    def ratio(group: pd.DataFrame):
        in_group = group[DeltaColumns.AUTHOR1.value] == group[DeltaColumns.AUTHOR2.value]
        out_group = ~in_group
        n_in, n_out = in_group.sum(), out_group.sum()
        if not (n_in and n_out):
            return np.nan
        squared_deltas = group[squared_delta_col]
        within = squared_deltas[in_group].mean()
        without = squared_deltas[out_group].mean()
        return within / without

    squared_delta_col = DeltaColumns.DELTA.value + "_squared"
    delta_values_df[squared_delta_col] = delta_values_df[DeltaColumns.DELTA.value] ** 2
    ratios = delta_values_df.groupby(DeltaColumns.AUTHOR1.value).apply(ratio)
    result = ratios.mean()
    return result

def f_ratio(distance_matrix: DistanceMatrix):
    """
    Calculates the (normalized) F-ratio over the distance matrix, according
    to Heeringa et al.

    Checks whether the distances within a group (i.e., texts with the same
    author) are much smaller thant the distances between groups
    """
    delta_values_df = distance_matrix.delta_values_df()
    result = _f_ratio(delta_values_df)
    return result

def _fisher_ld(matrix_groups_tuple):
    distance_matrix, group_index = matrix_groups_tuple
    groups = group_index.unique().to_list()
    gpb = distance_matrix.groupby(group_index)
    groupwise_means = gpb.mean().to_numpy()
    groupwise_vars = gpb.var().replace(0.0, np.nan).to_numpy()
    total_discrimination = 0.0
    for i in range(1, len(groups)):
        means_i, means_j = groupwise_means[i:], groupwise_means[:-i]
        var_i, var_j = groupwise_vars[i:], groupwise_vars[:-i]
        discrimination = ((means_i - means_j) ** 2) / (var_i + var_j)
        total_discrimination += np.nansum(discrimination)
    return total_discrimination / (sum(range(1, len(groups))) * len(group_index))



def fisher_ld(distance_matrix: DistanceMatrix):
    """
    Calculates Fisher's Linear Discriminant for the distance matrix.

    cf. Heeringa et al. (2002), Validating Dialect Comparison Methods, p. 4
    """
    group_index = distance_matrix.columns.map(distance_matrix.document_describer.group_name)
    return _fisher_ld((distance_matrix, group_index))



def _simple_score(partition: Tuple[pd.Series, pd.Series]):
    in_group, out_group = partition
    score = out_group.mean() - in_group.mean()
    return score

def simple_score(distance_matrix: DistanceMatrix):
    """
    Simple delta quality score for the given delta matrix:

    The difference between the means of the standardized differences
    between works of different authors and works of the same author; i.e.
    different authors are considered *score* standard deviations more
    different than equal authors.
    """
    partition = distance_matrix.z_scores().partition(square=False)
    return _simple_score(partition)