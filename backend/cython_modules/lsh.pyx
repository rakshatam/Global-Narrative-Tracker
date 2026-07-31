# distutils: language = c++
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.time cimport time

# Initialize random seed
srand(time(NULL))

cdef class MinHashLSH:
    cdef int num_permutations
    cdef int num_bands
    cdef int rows_per_band
    cdef float[:, :] random_vectors # For cosine similarity approximation
    
    # Simple hash table for LSH buckets: band_index -> { hash_val -> [doc_ids] }
    # Since we can't easily expose C++ unordered_map to python without extra wrapping,
    # we'll manage the buckets in a standard Python dictionary for simplicity, 
    # but the heavy hashing math is done in C.
    cdef object buckets 
    
    def __init__(self, int vector_dim=384, int num_permutations=128, int num_bands=32):
        """
        vector_dim: The size of the incoming embedding (384 for all-MiniLM-L6-v2)
        num_permutations: Number of random hyperplanes for Cosine Sim approximation
        """
        self.num_permutations = num_permutations
        self.num_bands = num_bands
        
        if num_permutations % num_bands != 0:
            raise ValueError("num_permutations must be divisible by num_bands")
            
        self.rows_per_band = num_permutations // num_bands
        
        # Initialize random vectors for random projection (Cosine Similarity MinHash/SimHash)
        # We generate random standard normal vectors
        cdef cnp.ndarray[cnp.float32_t, ndim=2] rv = np.random.randn(num_permutations, vector_dim).astype(np.float32)
        self.random_vectors = rv
        
        # Initialize buckets
        self.buckets = [{} for _ in range(self.num_bands)]
        
    cdef int[:] compute_signature(self, float[:] embedding):
        """
        Computes the MinHash/SimHash signature for a continuous vector
        using Random Projection.
        Returns a binary signature of length num_permutations.
        """
        cdef int[:] signature = np.zeros(self.num_permutations, dtype=np.int32)
        cdef int i, j
        cdef float dot_product
        
        cdef int v_dim = embedding.shape[0]
        
        # Matrix multiplication / dot products
        for i in range(self.num_permutations):
            dot_product = 0.0
            for j in range(v_dim):
                dot_product += self.random_vectors[i, j] * embedding[j]
            
            # If dot product > 0, hash is 1, else 0
            if dot_product > 0:
                signature[i] = 1
            else:
                signature[i] = 0
                
        return signature

    def insert(self, str doc_id, cnp.ndarray[cnp.float32_t, ndim=1] embedding):
        """
        Hashes an embedding and inserts the doc_id into the appropriate LSH buckets.
        """
        cdef int[:] signature = self.compute_signature(embedding)
        cdef int b, r, offset
        
        for b in range(self.num_bands):
            # Compute a single hash for the band
            offset = b * self.rows_per_band
            # Create a tuple of the signature bits for this band to use as a dict key
            band_sig = tuple([signature[offset + r] for r in range(self.rows_per_band)])
            
            if band_sig not in self.buckets[b]:
                self.buckets[b][band_sig] = []
            
            self.buckets[b][band_sig].append(doc_id)

    def query(self, cnp.ndarray[cnp.float32_t, ndim=1] embedding):
        """
        Finds all candidate doc_ids that might be similar to the given embedding.
        """
        cdef int[:] signature = self.compute_signature(embedding)
        cdef int b, r, offset
        
        candidates = set()
        
        for b in range(self.num_bands):
            offset = b * self.rows_per_band
            band_sig = tuple([signature[offset + r] for r in range(self.rows_per_band)])
            
            if band_sig in self.buckets[b]:
                for doc_id in self.buckets[b][band_sig]:
                    candidates.add(doc_id)
                    
        return list(candidates)
