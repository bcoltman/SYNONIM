import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union
import logging

# Import your custom classes. Adjust these imports according to your project structure.
from ..core import Model, Feature, Profile

logger = logging.getLogger(__name__)

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def model_from_frames(
    genomes_binary: Optional[pd.DataFrame] = None,
    genomes_abundance: Optional[pd.DataFrame] = None,
    metagenomes_binary: Optional[pd.DataFrame] = None,
    metagenomes_abundance: Optional[pd.DataFrame] = None,
    genomes_info: Optional[pd.DataFrame] = None,
    taxonomy_cols: Optional[List[str]] = None,
    model_id: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Model:
    
    gb = genomes_binary.astype(int) if genomes_binary is not None else None
    ga = genomes_abundance.astype(float) if genomes_abundance is not None else None
    mb = metagenomes_binary.astype(int) if metagenomes_binary is not None else None
    ma = metagenomes_abundance.astype(float) if metagenomes_abundance is not None else None
    
    # 1b) check taxonomy columns
    if genomes_info is not None and taxonomy_cols:
        missing = set(taxonomy_cols) - set(genomes_info.columns)
        if missing:
            logger.warning(f"Ignoring unknown taxonomy_cols: {missing!r}")
    
    # 2) master feature index
    feature_idx = pd.Index([])
    for df in (gb, ga, mb, ma):
        if df is not None:
            feature_idx = feature_idx.union(df.index)
    logger.info(f"Master feature count: {len(feature_idx)}")
    
    # 3) determine sample IDs
    genome_samples = []
    if gb is not None and ga is not None:
        if not gb.empty and not ga.empty:
            if gb.columns.tolist() == ga.columns.tolist():
                genome_samples = gb.columns.tolist()
            else:
                logger.info("Columns of genome binary and abundance are not the same.")
    elif gb is not None and not gb.empty:
        genome_samples = gb.columns.tolist()
    elif ga is not None and not ga.empty:
        genome_samples = ga.columns.tolist()
    else:
        logger.info("Both Genome DataFrames are either None or empty.")
        
    meta_samples = []
    if mb is not None and ma is not None:
        if not mb.empty and not ma.empty:
            if mb.columns.tolist() == ma.columns.tolist():
                meta_samples = mb.columns.tolist()
            else:
                logger.info("Columns of metagenome binary and abundance are not the same.")
    elif mb is not None and not mb.empty:
        meta_samples = mb.columns.tolist()
    elif ma is not None and not ma.empty:
        meta_samples = ma.columns.tolist()
    else:
        logging.info("Both Metagenome DataFrames are either None or empty.")
    
    # 4) reindex all matrices to full shape
    def reindex(df, cols):
        return df.reindex(index=feature_idx, columns=cols, fill_value=0)
    
    gb = reindex(gb, genome_samples) if gb is not None else None
    ga = reindex(ga, genome_samples) if ga is not None else None
    mb = reindex(mb, meta_samples)   if mb is not None else None
    ma = reindex(ma, meta_samples)   if ma is not None else None
        
    # 5) build canonical feature objects
    sorted_feats = feature_idx.sort_values()
    canonical = {fid: Feature(id=str(fid), name=str(fid)) for fid in sorted_feats}
    
    def make_profiles(samples, bin_df, ab_df, prof_type):
        out = []
        for s in samples:
            # Check if both bin_df and ab_df are None
            if bin_df is None and ab_df is None:
                logger.warning(f"Both binary and abundance are None for sample {s}. Skipping.")
                continue
            
            # Determine presence and mask
            if bin_df is not None:
                pres = (bin_df[s] != 0).astype(int)
                mask = pres.astype(bool)
            else:
                pres = (~ab_df[s].isna()).astype(int)
                mask = (~ab_df[s].isna())
                
            abund = ab_df[s] if ab_df is not None else None
            
             # Prepare features to add
            features_to_add = {
                canonical[fid]: (
                    {"presence": pres_val, "abundance": abund_val}
                    if abund is not None else {"presence": pres_val}
                )
                for fid, pres_val, abund_val in zip(sorted_feats, pres, abund if abund is not None else [None] * len(pres))
                if mask.loc[fid]
            }
            
            # Prepare metadata and taxonomy
            meta, tax = {}, {}
            if prof_type == "genome" and genomes_info is not None and s in genomes_info.index:
                meta = genomes_info.loc[s].to_dict()
                tax = {c: meta[c] for c in taxonomy_cols if c in meta} if taxonomy_cols else {}
                
                    
            p = Profile(id=str(s), name=str(s), profile_type=prof_type,
                        metadata=meta, taxonomy=tax)
            p.add_features(features_to_add)
            out.append(p)
            logger.debug(f"Built {prof_type} profile {s!r} w/ {len(features_to_add)} feats.")
        return out
        
    genome_profiles    = make_profiles(genome_samples, gb, ga, "genome")
    metagenome_profiles= make_profiles(meta_samples,   mb, ma, "metagenome")
    
    # 6) assemble model
    model = Model(id_or_model=model_id, name=model_name)
    model.add_features(list(canonical.values()))
    model.add_profiles(genome_profiles + metagenome_profiles)
    
    logger.info(f"Model has {len(genome_profiles)} genomes, {len(metagenome_profiles)} metas.")
    return model
