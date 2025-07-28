#!/usr/bin/env python3


# -----------------------------------------------------------------------------
# Utility: Generate a Unified Taxonomy Indicator Matrix (Sparse)
# -----------------------------------------------------------------------------
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import numpy as np
import math

def generate_unified_taxonomy_matrix(genome_labels, taxonomic_levels):
    """
    Generate a unified sparse indicator matrix for all specified taxonomic levels.
    
    Parameters:
      genome_labels: list where genome_labels[i] is a dict of taxonomy information for candidate i,
                     e.g. {"domain": "bacteria", "family": "FamA", ...}
      taxonomic_levels: list of strings (e.g., ["domain", "family"]) to include in the matrix.
      
    Returns:
      T_unified: csr_matrix of shape (n_total_keys, num_candidates) with binary indicators.
      unified_taxon_to_index: dict mapping a key "level-taxon" to the corresponding row index.
    """
    num_candidates = len(genome_labels)
    # First, determine all unique keys.
    unique_keys = set()
    for i in range(num_candidates):
        for level in taxonomic_levels:
            key = f"{level}-{genome_labels[i][level]}"
            unique_keys.add(key)
    unique_keys = sorted(list(unique_keys))
    unified_taxon_to_index = {key: idx for idx, key in enumerate(unique_keys)}
    
    rows = []
    cols = []
    data = []
    for i in range(num_candidates):
        for level in taxonomic_levels:
            key = f"{level}-{genome_labels[i][level]}"
            row = unified_taxon_to_index[key]
            rows.append(row)
            cols.append(i)
            data.append(1)
    
    # Build a COO matrix and then convert to CSR for efficient row slicing.
    T_unified = coo_matrix((data, (rows, cols)), shape=(len(unique_keys), num_candidates), dtype=int)
    return T_unified.tocsr(), unified_taxon_to_index


def determine_domain_constraints(group_size, proportions=[0.85, 0.1, 0.05], labels=["bacteria", "fungi", "archaea"]):
    # Step 1: Compute initial lower limits using ceiling
    initial_limits = [math.ceil(group_size * p) for p in proportions]
    total = sum(initial_limits)
    
    # Step 2: Adjust to match the total group size
    while total > group_size:
        # Find the category with the largest excess
        for i in sorted(range(len(initial_limits)), key=lambda x: proportions[x]):
            if initial_limits[i] > group_size * proportions[i]:
                initial_limits[i] -= 1
                break
        total = sum(initial_limits)
    
    constraints = dict(zip(labels, initial_limits))
    constraints = {k: {"exact":v} for k, v in constraints.items()}
    
    return constraints


def clr_transform(v: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Compute the centered log-ratio (CLR) transformation for a vector v.
    
    CLR(v)_i = log((v_i + epsilon) / g(v + epsilon))
    where g(v + epsilon) is the geometric mean of (v + epsilon).
    
    Parameters
    ----------
    v : np.ndarray
        Input abundance vector.
    epsilon : float, optional
        Small constant to avoid log(0).
        
    Returns
    -------
    np.ndarray
        CLR-transformed vector.
    
    Notes
    -----
    The CLR transform is computed as:
    
        CLR(v)_i = log((v_i + epsilon) / g(v + epsilon))
    
    where g(v + epsilon) is the geometric mean of (v + epsilon).
    """
    v_shift = v + epsilon
    # Compute geometric mean. Using log-average exp.
    geom_mean = np.exp(np.mean(np.log(v_shift)))
    return np.log(v_shift / geom_mean)