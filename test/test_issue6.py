import delta


def test_issue_6(distances):
    clustering = delta.Clustering(distances)
    clusters = clustering.fclustering()
    print("\nFLAT CLUSTERS\n", clusters.describe())
    print("\nATTEMPTING distances.evaluate()\n")
    assert distances.evaluate() is not None


def test_dm_zscore(distances):
    z_scores = distances.z_scores()
    assert z_scores is not None


def test_dm_operation(distances):
    assert distances - 1 is not None