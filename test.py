from sklearn.metrics import cohen_kappa_score
import numpy as np

a = "1 4 5 0 4 0 1 2 0 0 3 3 1 5 0 2 1 3 5 1 1 5 1 2 0 1 1 1 5 4 1 0 5 0 3 5 0 4 2 0 1 0 0 2 1 1 4 1 2 1 4 1 0 1 3 0 1 2 1 5 3 0 4 2".split(" ")
a = [int(i) for i in a]

b = "1 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5".split(" ")
b = [int(i) for i in b]

print(cohen_kappa_score(a, b, weights="quadratic"))
