"""Test-only stand-in for TabPFN. See ../README.md."""

from sklearn.linear_model import LogisticRegression, Ridge


class _Base:
    @classmethod
    def create_default_for_version(cls, version=None, device="cpu", **k):
        return cls()

    def __init__(self, *a, **k):
        self._m = None


class TabPFNClassifier(_Base):
    def fit(self, X, y, sample_weight=None):
        self._m = LogisticRegression(max_iter=500).fit(X, y)
        self.classes_ = self._m.classes_
        return self

    def predict(self, X):
        return self._m.predict(X)

    def predict_proba(self, X):
        return self._m.predict_proba(X)


class TabPFNRegressor(_Base):
    def fit(self, X, y, sample_weight=None):
        self._m = Ridge().fit(X, y)
        return self

    def predict(self, X, output_type=None):
        return self._m.predict(X)
