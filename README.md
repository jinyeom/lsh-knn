# Approximate k-NN with locality-sensitive hashing


```python
import numpy as np
```


```python
rng = np.random.default_rng(0)
```


```python
m = 10000  # number of data points in the
n = 128    # number of features in each data point
```


```python
X = rng.normal(size=(m, n))  # random dataset
q = rng.normal(size=n)       # query vector
```

## Vanilla k-NN search


```python
def knn_search(query, data, k=5, debug=False):
    assert k <= len(data)
    dists = np.sqrt(np.sum((data - query) ** 2, axis=1))  # euclidean distance
    if debug:
        print("[DEBUG] max dist =", np.max(dists))
        print("[DEBUG] min dist =", np.min(dists))
        print("[DEBUG] mean dist =", np.mean(dists))
    inds = np.argsort(dists)  # sorted in ascending order
    inds_k = inds[:k]         # top k closest data points
    # NOTE: optionally, if the argumet dataset has a set of labels, we can also
    # associate the query vector with a label (i.e., classification).
    return data[inds_k], dists[inds_k]
```


```python
neighbors, dists = knn_search(q, X, debug=True)
for i, (neighbor, dist) in enumerate(zip(neighbors, dists)):
    print(f"top {i + 1}: dist = {dist}")
```

    [DEBUG] max dist = 19.576730097361967
    [DEBUG] min dist = 12.72876479759339
    [DEBUG] mean dist = 16.136399135895942
    top 1: dist = 12.72876479759339
    top 2: dist = 12.980832845009397
    top 3: dist = 13.109098301375685
    top 4: dist = 13.178447300861382
    top 5: dist = 13.248307679497904


## Approximate k-NN search


```python
def hamming_hash(data, hyperplanes):
    b = hyperplanes.shape[0]
    hash_key = (data @ hyperplanes.T) >= 0
    bin_vals = np.array([2 ** i for i in range(b)], dtype=int)
    hash_key = hash_key @ bin_vals
    return hash_key
```


```python
def generate_hyperplanes(data, bucket_size=16):
    m = data.shape[0]            # number of data points
    n = data.shape[1]            # number of features in a data point
    b = m // bucket_size         # desired number of hash buckets
    h = int(np.log2(b))          # desired number of hyperplanes
    H = rng.normal(size=(h, n))  # hyperplanes as their normal vectors
    return H
```


```python
def locality_sensitive_hash(data, hyperplanes):
    hash_vals = hamming_hash(data, hyperplanes)
    hash_table = {}
    for i, v in enumerate(hash_vals):
        if v not in hash_table:
            hash_table[v] = set()
        hash_table[v].add(i)
    return hash_table
```


```python
hyperplanes = generate_hyperplanes(X)
hash_table = locality_sensitive_hash(X, hyperplanes)
avg_bucket_size = np.mean([len(v) for v in hash_table.values()])
print("avg_bucket_size =", avg_bucket_size)
```

    avg_bucket_size = 19.53125



```python
def approx_knn_search(query, data, k=5, bucket_size=16, repeat=10, debug=False):
    candidates = set()
    for i in range(repeat):
        hyperplanes = generate_hyperplanes(data)
        hash_table = locality_sensitive_hash(data, hyperplanes)
        if debug:
            avg_bucket_size = np.mean([len(v) for v in hash_table.values()])
            print(f"[DEBUG] i = {i}, avg_bucket_size = {avg_bucket_size}")
        query_hash = hamming_hash(query, hyperplanes)
        if query_hash in hash_table:
            candidates = candidates.union(hash_table[query_hash])
    candidates = np.stack([data[i] for i in candidates], axis=0)
    return knn_search(query, candidates, k=k, debug=debug)
```


```python
neighbors, dists = approx_knn_search(q, X, repeat=24, debug=True)
for i, (neighbor, dist) in enumerate(zip(neighbors, dists)):
    print(f"top {i + 1}: dist = {dist}")
```

    [DEBUG] i = 0, avg_bucket_size = 19.53125
    [DEBUG] i = 1, avg_bucket_size = 19.53125
    [DEBUG] i = 2, avg_bucket_size = 19.53125
    [DEBUG] i = 3, avg_bucket_size = 19.53125
    [DEBUG] i = 4, avg_bucket_size = 19.53125
    [DEBUG] i = 5, avg_bucket_size = 19.53125
    [DEBUG] i = 6, avg_bucket_size = 19.53125
    [DEBUG] i = 7, avg_bucket_size = 19.53125
    [DEBUG] i = 8, avg_bucket_size = 19.53125
    [DEBUG] i = 9, avg_bucket_size = 19.53125
    [DEBUG] i = 10, avg_bucket_size = 19.53125
    [DEBUG] i = 11, avg_bucket_size = 19.53125
    [DEBUG] i = 12, avg_bucket_size = 19.53125
    [DEBUG] i = 13, avg_bucket_size = 19.569471624266146
    [DEBUG] i = 14, avg_bucket_size = 19.53125
    [DEBUG] i = 15, avg_bucket_size = 19.53125
    [DEBUG] i = 16, avg_bucket_size = 19.53125
    [DEBUG] i = 17, avg_bucket_size = 19.53125
    [DEBUG] i = 18, avg_bucket_size = 19.53125
    [DEBUG] i = 19, avg_bucket_size = 19.53125
    [DEBUG] i = 20, avg_bucket_size = 19.53125
    [DEBUG] i = 21, avg_bucket_size = 19.53125
    [DEBUG] i = 22, avg_bucket_size = 19.53125
    [DEBUG] i = 23, avg_bucket_size = 19.53125
    [DEBUG] max dist = 18.035577554152816
    [DEBUG] min dist = 12.980832845009397
    [DEBUG] mean dist = 15.752405187018653
    top 1: dist = 12.980832845009397
    top 2: dist = 13.315716707872218
    top 3: dist = 13.599262272317079
    top 4: dist = 13.77326305995105
    top 5: dist = 13.810928200015331

