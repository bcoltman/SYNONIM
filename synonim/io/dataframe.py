import pandas as pd
from typing import List, Optional
import logging

from ..core import Model, Feature, Profile

logger = logging.getLogger(__name__)


def model_from_frames(
    genomes_binary: Optional[pd.DataFrame] = None,
    metagenomes_binary: Optional[pd.DataFrame] = None,
    genomes_info: Optional[pd.DataFrame] = None,
    taxonomy_cols: Optional[List[str]] = None,
    model_id: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Model:
    """
    Build a SYNONIM model from feature-by-sample data frames.

    Binary data frames must use feature IDs as their index and sample IDs as columns.
    If ``genomes_info`` is provided, it must be indexed by genome sample ID so
    metadata and taxonomy can be attached to the matching genome profiles.
    """
    
    gb = genomes_binary.astype(int) if genomes_binary is not None else None
    mb = metagenomes_binary.astype(int) if metagenomes_binary is not None else None
    if gb is None and mb is None:
        raise ValueError("At least one binary data frame must be provided.")
    
    # 1b) check taxonomy columns
    if genomes_info is not None and taxonomy_cols:
        missing = set(taxonomy_cols) - set(genomes_info.columns)
        if missing:
            logger.warning(f"Ignoring unknown taxonomy_cols: {missing!r}")
    
    # 2) master feature index
    feature_idx = pd.Index([])
    for df in (gb, mb):
        if df is not None:
            feature_idx = feature_idx.union(df.index)
    logger.info(f"Master feature count: {len(feature_idx)}")
    
    # 3) determine sample IDs
    genome_samples = []
    if gb is not None and not gb.empty:
        genome_samples = gb.columns.tolist()
    else:
        logger.info("Genome binary DataFrame is either None or empty.")
        
    meta_samples = []
    if mb is not None and not mb.empty:
        meta_samples = mb.columns.tolist()
    else:
        logging.info("Metagenome binary DataFrame is either None or empty.")
    
    # 4) reindex all matrices to full shape
    def reindex(df, cols):
        return df.reindex(index=feature_idx, columns=cols, fill_value=0)
    
    gb = reindex(gb, genome_samples) if gb is not None else None
    mb = reindex(mb, meta_samples)   if mb is not None else None
        
    # 5) build canonical feature objects
    sorted_feats = feature_idx.sort_values()
    canonical = {fid: Feature(id=str(fid), name=str(fid)) for fid in sorted_feats}
    
    def make_profiles(samples, bin_df, prof_type):
        out = []
        for s in samples:
            if bin_df is None:
                logger.warning(f"Binary data is None for sample {s}. Skipping.")
                continue
            
            pres = (bin_df[s] != 0).astype(int)
            mask = pres.astype(bool)
            
            # Prepare features to add
            features_to_add = {
                canonical[fid]: {"presence": pres_val}
                for fid, pres_val in zip(sorted_feats, pres)
                if mask.loc[fid]
            }
            
            # Prepare metadata and taxonomy
            meta, tax = {}, {}
            if prof_type == "genome" and genomes_info is not None:
                if s in genomes_info.index:
                    meta = genomes_info.loc[s].to_dict()
                    tax = {c: meta[c] for c in taxonomy_cols if c in meta} if taxonomy_cols else {}
                else:
                    logger.warning(
                        "No genomes_info row found for genome sample %r; "
                        "genomes_info must be indexed by sample ID.",
                        s,
                    )
                
                    
            p = Profile(id=str(s), name=str(s), profile_type=prof_type,
                        metadata=meta, taxonomy=tax)
            p.add_features(features_to_add)
            out.append(p)
            logger.debug(f"Built {prof_type} profile {s!r} w/ {len(features_to_add)} feats.")
        return out
        
    genome_profiles = make_profiles(genome_samples, gb, "genome")
    metagenome_profiles = make_profiles(meta_samples, mb, "metagenome")
    
    # 6) assemble model
    model = Model(id_or_model=model_id, name=model_name)
    model.add_features(list(canonical.values()))
    model.add_profiles(genome_profiles + metagenome_profiles)
    
    logger.info(f"Model has {len(genome_profiles)} genomes, {len(metagenome_profiles)} metas.")
    return model
