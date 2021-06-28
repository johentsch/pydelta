import pandas as pd
from pytest import approx
from scipy.special import comb

import delta as d
import pytest

from delta.deltas import DistanceMatrix


@pytest.fixture(scope='module')
def c1000(corpus):
    return corpus.get_mfw_table(1000)


def fn_id(fn):
    return fn.name if isinstance(fn, d.DeltaFunction) else None

@pytest.mark.parametrize("function,expected_distance", [(d.functions.burrows, 0.7538867972199293),
                                                        (d.functions.linear, 1149.434663563308),
                                                        (d.functions.quadratic, 1102.845003724634),
                                                        (d.functions.eder, 0.3703309813454142),
                                                        (d.functions.cosine_delta, 0.6156353166442046)],
                         ids=fn_id)
def test_distance(function, expected_distance, c1000):
    distances = function(c1000)
    sample = distances.at['Fontane,-Theodor_Der-Stechlin',
                          'Fontane,-Theodor_Effi-Briest']
    assert sample == approx(expected_distance, rel=1e-2)


def test_simple_score(distances):
    assert distances.simple_score() > 0


def test_composite_metric(c1000):
    mcosine = d.MetricDeltaFunction('cosine', 'mcosine')
    assert mcosine.fix_symmetry == True, "fix_symmetry is False!?"
    mcd = d.CompositeDeltaFunction('mcosine-z_score', 'metric_cosine_delta')
    assert mcd.basis.fix_symmetry == True, "basis.fix_symmetry is False!?"
    test_distance(d.functions.metric_cosine_delta, 0.6156353166442046, c1000)


def test_delta_values_shape(distances: DistanceMatrix):
    dv = distances.delta_values()
    assert len(dv) == comb(distances.shape[0], 2)

def test_compare_with_shape(distances: DistanceMatrix, testdir):
    metadata = pd.read_csv(testdir + '.csv', index_col=0)
    comparison = distances.compare_with(metadata)
    assert comparison.shape[0] == comb(distances.shape[0], 2)