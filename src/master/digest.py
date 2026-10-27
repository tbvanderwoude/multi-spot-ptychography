import numpy as np
import xxhash


def make_array_digest(x):
    # https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array/31829201#31829201
    h = xxhash.xxh64()
    h.update(np.asarray(x))
    digest = h.intdigest()
    h.reset()
    return digest
