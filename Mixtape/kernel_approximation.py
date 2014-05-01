import numpy as np
from sklearn.kernel_approximation import Nystroem as _Nystroem


class Nystroem(_Nystroem):
    def fit(self, sequences, y=None):
        return super(Nystroem, self).fit(np.concatenate(sequences))

    def transform(self, sequences):
        trans = super(Nystroem, self).transform
        y = [trans(X) for X in sequences]
        return y
        
