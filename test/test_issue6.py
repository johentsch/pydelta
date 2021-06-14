import delta
import pytest


def test_issue_6(distances):
    clustering = delta.Clustering(distances)
    clusters = clustering.fclustering()
    print("\nFLAT CLUSTERS\n", clusters.describe())
    print("\nATTEMPTING distances.evaluate()\n")
    assert distances.evaluate() is not None


@pytest.mark.skipif('KMedoidsClustering' not in dir(delta), reason='KMedoidsClustering not available')
def test_kmedoids(corpus):
    clusters = delta.KMedoidsClustering(corpus, delta.functions.cosine_delta)
    clusters.describe()
    assert clusters.cluster_errors() == 0


def test_dm_zscore(distances):
    z_scores = distances.z_scores()
    assert z_scores is not None


def test_dm_operation(distances):
    assert distances - 1 is not None