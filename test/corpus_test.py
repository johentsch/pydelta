from pytest import approx

import delta as d
import os
import pytest


@pytest.fixture
def feature_generator() -> d.FeatureGenerator:
    return d.FeatureGenerator()


def test_tokenize():
    assert list(d.FeatureGenerator().tokenize(["This is a", "simple test"])) \
           == ["This", "is", "a", "simple", "test"]


def test_tokenize_letters():
    fg1 = d.FeatureGenerator(token_pattern=d.LETTERS_PATTERN)
    assert list(fg1.tokenize(["I don't like mondays."])) \
           == ["I", "don", "t", "like", "mondays"]


def test_tokenize_words():
    fg1 = d.FeatureGenerator(token_pattern=d.WORD_PATTERN)
    assert list(fg1.tokenize(["I don't like mondays."])) \
           == ["I", "don't", "like", "mondays"]


def test_count_tokens(feature_generator):
    result = feature_generator.count_tokens(
            ["this is a test", "testing this generator"])
    assert result["this"] == 2
    assert result["generator"] == 1
    assert result.sum() == 7


def test_get_name(feature_generator):
    assert feature_generator.get_name('foo/bar.baz.txt') == 'bar.baz'


def test_call_fg(feature_generator, testdir):
    df = feature_generator(os.fspath(testdir))
    assert df.loc[:, 'und'].sum() == approx(25738.0)


## Corpus

@pytest.fixture(scope='module')
def corpus(testdir):
    return d.Corpus(testdir)

def test_corpus_parse(corpus):
    assert corpus.und.sum() == approx(25738.0)

def test_corpus_mfw(corpus):
    rel_corpus = corpus.get_mfw_table(0)
    assert rel_corpus.sum(axis=1).sum() == approx(9)


def test_integration_cluster(corpus):
    # FIXME
    top1000 = corpus.get_mfw_table(1000)
    deltas = d.functions.cosine_delta(top1000)
    hclust = d.Clustering(deltas)
    fclust = hclust.fclustering()
    print(fclust.describe())
    print(fclust.evaluate())
    assert fclust.adjusted_rand_index() == 1


def test_table_describer(testdir):
    corpus = d.Corpus(testdir,
                      document_describer=d.util.TableDocumentDescriber(testdir + '.csv', 'Author', 'Title'))
    assert corpus.document_describer.group_name(corpus.index[-1]) in {'Raabe', 'Marlitt', 'Fontane'}
