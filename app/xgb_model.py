"""Inferensi XGBoost tanpa pustaka XGBoost/SHAP saat runtime.

Berkas model JSON berisi struktur pohon yang sudah dilatih. Modul ini
menghitung probabilitas kelas dan kontribusi Tree SHAP dengan algoritma
interventional Tree SHAP. Hasil kontribusinya setara dengan `pred_contribs`
pada XGBoost, tetapi hanya memerlukan Python standar.
"""
import json
import math
from pathlib import Path


def _sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1 + exp)


def _prepare_tree(raw):
    left = list(raw["left_children"])
    right = list(raw["right_children"])
    default_left = list(raw["default_left"])
    is_leaf = [child == -1 for child in left]

    features = list(raw["split_indices"])
    conditions = [float(value) for value in raw["split_conditions"]]
    thresholds = [
        0.0 if leaf else math.nextafter(condition, -math.inf)
        for leaf, condition in zip(is_leaf, conditions)
    ]
    values = [condition if leaf else 0.0 for leaf, condition in zip(is_leaf, conditions)]
    weights = [max(float(value), 1e-6) for value in raw["sum_hessian"]]

    def compute_expectation(node):
        if right[node] == -1:
            return 0
        depth_left = compute_expectation(left[node])
        depth_right = compute_expectation(right[node])
        left_weight = weights[left[node]]
        right_weight = weights[right[node]]
        values[node] = (
            left_weight * values[left[node]] + right_weight * values[right[node]]
        ) / (left_weight + right_weight)
        return max(depth_left, depth_right) + 1

    compute_expectation(0)
    return {
        "left": left,
        "right": right,
        "default_left": default_left,
        "features": features,
        "thresholds": thresholds,
        "values": values,
        "weights": weights,
        "expected": values[0],
    }


def _extend_path(fi, zf, of, pw, depth, zero, one, feature):
    fi[depth] = feature
    zf[depth] = zero
    of[depth] = one
    pw[depth] = 1.0 if depth == 0 else 0.0
    for i in range(depth - 1, -1, -1):
        pw[i + 1] += one * pw[i] * (i + 1.0) / (depth + 1.0)
        pw[i] = zero * pw[i] * (depth - i) / (depth + 1.0)


def _unwind_path(fi, zf, of, pw, depth, index):
    one = of[index]
    zero = zf[index]
    next_one = pw[depth]
    for i in range(depth - 1, -1, -1):
        if one != 0:
            previous = pw[i]
            pw[i] = next_one * (depth + 1.0) / ((i + 1.0) * one)
            next_one = previous - pw[i] * zero * (depth - i) / (depth + 1.0)
        else:
            pw[i] = (pw[i] * (depth + 1)) / (zero * (depth - i))
    for i in range(index, depth):
        fi[i] = fi[i + 1]
        zf[i] = zf[i + 1]
        of[i] = of[i + 1]


def _unwound_path_sum(fi, zf, of, pw, depth, index):
    one = of[index]
    zero = zf[index]
    next_one = pw[depth]
    total = 0.0
    for i in range(depth - 1, -1, -1):
        if one != 0:
            value = next_one * (depth + 1.0) / ((i + 1.0) * one)
            total += value
            next_one = pw[i] - value * zero * ((depth - i) / (depth + 1.0))
        else:
            total += (pw[i] / zero) / ((depth - i) / (depth + 1.0))
    return total


def _recurse(tree, row, phi, node, depth, pfi, pzf, pof, ppw, pzero, pone, pfeature):
    fi, zf, of, pw = [0.0] * len(phi), [0.0] * len(phi), [0.0] * len(phi), [0.0] * len(phi)
    fi[: depth + 1] = pfi[: depth + 1]
    zf[: depth + 1] = pzf[: depth + 1]
    of[: depth + 1] = pof[: depth + 1]
    pw[: depth + 1] = ppw[: depth + 1]
    _extend_path(fi, zf, of, pw, depth, pzero, pone, pfeature)

    left = tree["left"]
    right = tree["right"]
    features = tree["features"]
    split = features[node]

    if right[node] == -1:
        for i in range(1, depth + 1):
            weight = _unwound_path_sum(fi, zf, of, pw, depth, i)
            phi[int(fi[i])] += weight * (of[i] - zf[i]) * tree["values"][node]
        return

    value = row[split]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        hot = left[node] if tree["default_left"][node] == 1 else right[node]
    elif value < tree["thresholds"][node]:
        hot = left[node]
    else:
        hot = right[node]
    cold = right[node] if hot == left[node] else left[node]

    node_weight = tree["weights"][node]
    hot_zero = tree["weights"][hot] / node_weight
    cold_zero = tree["weights"][cold] / node_weight
    incoming_zero = 1.0
    incoming_one = 1.0

    path_index = 0
    while path_index <= depth:
        if fi[path_index] == split:
            break
        path_index += 1
    if path_index != depth + 1:
        incoming_zero = zf[path_index]
        incoming_one = of[path_index]
        _unwind_path(fi, zf, of, pw, depth, path_index)
        depth -= 1

    _recurse(tree, row, phi, hot, depth + 1, fi, zf, of, pw,
             hot_zero * incoming_zero, incoming_one, split)
    _recurse(tree, row, phi, cold, depth + 1, fi, zf, of, pw,
             cold_zero * incoming_zero, 0.0, split)


class XGBModel:
    """Model biner XGBoost yang dimuat dari berkas JSON."""

    def __init__(self, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        learner = payload["learner"]
        trees = learner["gradient_booster"]["model"]["trees"]
        base_score = float(learner["learner_model_param"]["base_score"].strip("[]"))
        self.base_score = math.log(base_score / (1 - base_score))
        self.trees = [_prepare_tree(tree) for tree in trees]

    def raw_score(self, values):
        score = self.base_score
        for tree in self.trees:
            node = 0
            left = tree["left"]
            right = tree["right"]
            while right[node] != -1:
                value = values[tree["features"][node]]
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    go_left = tree["default_left"][node] == 1
                else:
                    go_left = value < tree["thresholds"][node]
                node = left[node] if go_left else right[node]
            score += tree["values"][node]
        return score

    def predict_proba(self, values):
        """Probabilitas kelas 1 (legitimate)."""
        return _sigmoid(self.raw_score(values))

    def predict(self, values):
        return 1 if self.predict_proba(values) >= 0.5 else 0

    def shap_values(self, values):
        """Kontribusi Tree SHAP untuk setiap fitur; elemen terakhir adalah bias."""
        phi = [0.0] * (len(values) + 1)
        for tree in self.trees:
            phi[-1] += tree["expected"]
            _recurse(tree, values, phi, 0, 0,
                     [0] * len(phi), [0.0] * len(phi), [0.0] * len(phi), [0.0] * len(phi),
                     1.0, 1.0, -1)
        phi[-1] += self.base_score
        return phi
