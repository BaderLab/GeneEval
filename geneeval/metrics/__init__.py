from functools import partial
from sklearn import metrics

f1_micro_score = partial(metrics.f1_score, average="micro")
f1_micro_score.__name__ = "f1_micro_score"
