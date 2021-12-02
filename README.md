# Approximate k-NN search with locality-sensitive hashing


```python
import numpy as np
```


```python
rs = np.random.default_rng(0)
```


```python
m = 1000  # number of data points in the
n = 16    # number of features in each data point
```


```python
X = rs.normal(size=(m, n))  # random dataset
q = rs.normal(size=n)       # query vector
```

## Vanilla k-NN search


```python
def knn_search(query, data, k=5):
    assert k <= len(data)
    dists = np.sqrt(np.sum((data - query) ** 2, axis=1)) # euclidean distance
    inds = np.argsort(dists)                             # sorted in ascending order
    inds_k = inds[:k]                                    # top k closest data points
    # NOTE: optionally, if the argumet dataset has a set of labels, we can also
    # associate the query vector with a label (i.e., classification).
    return data[inds_k], dists[inds_k]
```


```python
neighbors, dists = knn_search(q, X)

print("query =", q)
print()

for i, (neighbor, dist) in enumerate(zip(neighbors, dists)):
    print(f"top {i + 1}:")
    print("neighbor =", neighbor)
    print("dist =", dist)
    print()
```

    query = [ 0.74597801  0.10798961  0.96640122  0.15459683 -0.52637468  1.29406665
      0.15440615 -0.16837241 -1.25924732 -0.29044147 -1.6855752  -1.01036298
     -0.05422822 -0.90894185 -0.83150002 -1.18530497]
    
    top 1:
    neighbor = [ 0.79580696 -0.66135363  0.06473675  1.00379453 -0.4779662  -0.08503506
     -0.50306338  0.22048967 -0.2966549   0.16166078 -1.55502652 -0.52165985
     -0.95881421 -0.24767401 -1.38765319  0.44639838]
    dist = 3.198502734555067
    
    top 2:
    neighbor = [ 0.17507704 -0.46295949  0.21952077 -0.25228969  0.60720812  1.05960923
     -0.65088179  0.6286331  -0.91197297  0.71001593 -0.8652757   0.34580728
     -0.72779903 -1.01855172  0.39106477 -0.23056385]
    dist = 3.2410639159872017
    
    top 3:
    neighbor = [ 0.62507323  0.65184664  1.06381321 -1.53810049 -0.14029532  1.01364623
      0.18528485  0.0226172   0.11771915  0.08777791 -0.50787174 -1.73716936
     -1.02854233  0.16894379 -0.76239505 -2.26023546]
    dist = 3.2674043491121405
    
    top 4:
    neighbor = [-0.2143315   1.03290702  0.11692843  0.60510131 -0.28401206  1.30303073
     -0.65291775 -0.80419847 -1.31714793  0.29695359 -0.13561754  0.84169419
     -0.4736822  -1.57655737  0.0424373  -0.89523907]
    dist = 3.3863724337304166
    
    top 5:
    neighbor = [ 0.35903809 -0.21348963  1.07878313  0.65185984  0.1173584   0.20223162
     -0.28008264 -0.34814262  0.36218495 -0.55756653 -0.86449782  0.5983379
     -0.19086241  0.14229764 -1.86871093 -0.09926336]
    dist = 3.4179111585451505
    


## Approximate k-NN search


```python
def locality_sensitive_hash(data, hyperplanes):
    b = hyperplanes.shape[0]                  # number of hyperplanes (i.e., number of bits in each code)
    hamm_codes = (data @ hyperplanes.T) >= 0  # hamming codes
    hash_vals = hamm_codes @ np.array([2 ** i for i in range(b)], dtype=int)
    hash_table = {}
    for i, v in enumerate(hash_vals):
        if v not in hash_table:
            hash_table[v] = []
        hash_table[v].append(i)
    return hash_table
```


```python
hyperplanes = rs.normal(size=(3, X.shape[1]))  # hyperplanes represented as their normal vectors
hash_table = locality_sensitive_hash(X, hyperplanes)

for k, v in hash_table.items():
    print(k, len(v))
```

    5 111
    7 143
    2 94
    1 144
    6 137
    0 159
    4 106
    3 106



```python
def approx_knn_search(query, data, k=5):
    hyperplanes = 
```
