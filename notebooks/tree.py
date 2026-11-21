"""Ядро дерева из ЛР3 для ансамблей в ЛР4–5."""

import numpy as np


class DecisionTreeNumPy:
    def __init__(self, classification=True, max_depth=None, min_samples_leaf=1,
                 max_features=1.0, random_state=None):
        self.classification = classification
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        self._rng = np.random.default_rng(self.random_state)
        self.tree = self._grow(np.asarray(X, dtype=float), np.asarray(y, dtype=float), depth=0)
        return self

    def _grow(self, X, y, depth):
        n = len(y)
        leaf = float(y.mean())
        minimum = self.min_samples_leaf
        if n < 2 * minimum or (self.max_depth is not None and depth >= self.max_depth):
            return leaf

        centered = y - leaf
        factor = 2 if self.classification else 1
        best_loss = factor * np.dot(centered, centered)
        if best_loss <= 1e-12:
            return leaf

        split = None
        n_features = X.shape[1]
        count = int(np.sqrt(n_features)) if self.max_features == "sqrt" else int(self.max_features * n_features)
        features = range(n_features) if count == n_features else self._rng.choice(n_features, size=max(1, count), replace=False)
        for feature in features:
            order = np.argsort(X[:, feature], kind="stable")
            values = X[order, feature]
            candidates = np.flatnonzero(values[:-1] < values[1:]) + 1
            counts = candidates[(candidates >= minimum) & (n - candidates >= minimum)]
            if not len(counts):
                continue

            targets = centered[order]
            sums = np.cumsum(targets)
            squares = np.cumsum(targets ** 2)
            left_sum, left_sq = sums[counts - 1], squares[counts - 1]
            right_sum, right_sq = sums[-1] - left_sum, squares[-1] - left_sq
            left_loss = left_sq - left_sum ** 2 / counts
            right_loss = right_sq - right_sum ** 2 / (n - counts)
            losses = factor * (left_loss + right_loss)
            i = np.argmin(losses)
            if losses[i] < best_loss - 1e-12:
                best_loss = losses[i]
                pos = counts[i]
                split = (feature, (values[pos - 1] + values[pos]) / 2)

        if split is None:
            return leaf
        feature, threshold = split
        left = X[:, feature] <= threshold
        return (feature, threshold, self._grow(X[left], y[left], depth + 1), self._grow(X[~left], y[~left], depth + 1))

    def predict_leaf(self, X):
        """Среднее в листе: прогноз регрессии или доля класса 1."""
        predictions = []
        for row in np.asarray(X, dtype=float):
            node = self.tree
            while isinstance(node, tuple):
                feature, threshold, left, right = node
                node = left if row[feature] <= threshold else right
            predictions.append(node)
        return np.asarray(predictions)

    def predict(self, X):
        result = self.predict_leaf(X)
        return (result > 0.5).astype(int) if self.classification else result
