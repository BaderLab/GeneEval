from functools import partial

from sklearn.metrics import f1_score

f1_micro_score: f1_score = partial(f1_score, average="micro")
f1_micro_score.__name__ = "f1_micro_score"
