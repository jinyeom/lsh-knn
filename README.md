# Approximate k-NN with locality-sensitive hashing

**Locality-sensitive hashing (LSH)** is a method that hashes similar data points to have a high chance of being grouped together. This makes LSH a good way to optimize similarity search: by comparing the query data only with a subset of presumably similar items, you can avoid having to sort the entire dataset by some distance metric. The trade-off here is that there is no guarantee for correctness.

### Why is LSH exciting?

**Generally, LSH offers a way to narrow the scope of search without any prior knowledge about the target dataset (i.e., data-independent).** This property benefits cases outside of methods like k-NN for information retrieval. For example, one such place is _inside_ of a [Transformer](https://arxiv.org/abs/1706.03762) model: the dot-product attention mechanism. Kitaev, Kaiser and Levskaya proposed an alternative architecture called [Reformer](https://arxiv.org/abs/2001.04451), which reduces the complexity of the attention mechanism from O(N^2) to O(NlogN) (where N is the number of input items) by introducing LSH to each attention layer.


```python
import numpy as np
```


```python
rng = np.random.default_rng(0)
```

For the purpose of this demonstration, we will generate a random "dataset" of Gaussian samples and a query vector from the same distribution.


```python
m = 10000  # number of data points in the
n = 128    # number of features in each data point
```


```python
X = rng.normal(size=(m, n))  # random dataset
q = rng.normal(size=n)       # query vector
```

## Vanilla k-NN search

As a baseline, we implement a vanilla k-nearest neighbors (k-NN) search algorithm.


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
neighbors, dists = knn_search(q, X, debug=False)  # set debug=True for additional information
for i, (neighbor, dist) in enumerate(zip(neighbors, dists)):
    print(f"top {i + 1}: dist = {dist}")
```

    top 1: dist = 12.72876479759339
    top 2: dist = 12.980832845009397
    top 3: dist = 13.109098301375685
    top 4: dist = 13.178447300861382
    top 5: dist = 13.248307679497904


## Approximate k-NN search

In this section, we will implement locality-sensitive hashing (LSH) which maps a numerical vector to a hash value (an integer) with a set of random hyperplanes. The main idea of the technique is to split the input data space with a plane and determine whether the data point belongs above (1) or below (0) the plane. By repeating this technique $b$ times, each data point can be encoded to a binary string of length $b$. For the purpose of building a hash table, we convert these binary strings to decimal values.


```python
def hamming_hash(data, hyperplanes):
    b = len(hyperplanes)
    hash_key = (data @ hyperplanes.T) >= 0
    dec_vals = np.array([2 ** i for i in range(b)], dtype=int)
    hash_key = hash_key @ dec_vals
    return hash_key
```

Below, we write a small function that generates random hyperplanes, which determines the number of hyperplanes based on a desired expected number of elements in each bucket.


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


Now, we implement a k-NN search algorithm that incorporates LSH. Note that we can repeat the search with more than one set of hyperplanes to boost the accuracy. Feel free to experiment by tweaking the argument `repeat=10`.


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
neighbors, dists = approx_knn_search(q, X, repeat=24, debug=False)  # set debug=True for additional information
for i, (neighbor, dist) in enumerate(zip(neighbors, dists)):
    print(f"top {i + 1}: dist = {dist}")
```

    top 1: dist = 12.980832845009397
    top 2: dist = 13.315716707872218
    top 3: dist = 13.599262272317079
    top 4: dist = 13.77326305995105
    top 5: dist = 13.810928200015331

