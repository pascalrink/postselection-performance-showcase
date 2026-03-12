import numpy as np

from mabt import mabt_ci


# %%
def softmax(logits):
    logits = np.asarray(logits)
    shifted = logits - logits.max(axis=1)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1)
