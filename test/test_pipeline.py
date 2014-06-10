import numpy as np
from mixtape.markovstatemodel import MarkovStateModel
from mixtape.cluster import KMeans
from sklearn.pipeline import Pipeline

def test_1():
    pipeline = Pipeline([
        ('cluster', KMeans(n_clusters=100)),
        ('msm', MarkovStateModel(n_states=100))
    ])
    pipeline.fit([np.random.RandomState(0).randn(1000,2)])

test_1()
