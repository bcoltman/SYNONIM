import pandas as pd
from typing import List, Optional
from synonim.profile import Profile
from synonim.feature import Feature

def generate_profiles_from_genomes(
    genomes_df: pd.DataFrame,
    genome_info_df: pd.DataFrame,
    profile_type: str = "genome",
    taxonomy_cols: Optional[List[str]] = None
) -> List[Profile]:
    """
    Generate a list of Profile objects from a genomes DataFrame and associated metadata.
    
    Parameters
    ----------
    genomes_df : pd.DataFrame
        A DataFrame where each column corresponds to an isolate and each row to a feature.
        The index should be feature identifiers.
    genome_info_df : pd.DataFrame
        A DataFrame of metadata for isolates, indexed by the isolate ID (formatted_accession).
    profile_type : str, optional
        The type of profile, e.g. "genome" or "metagenome". Default is "genome".
    taxonomy_cols : List[str], optional
        A list of column names from genome_info_df to extract as taxonomy information.
        If None, taxonomy will be left as an empty dict.
        
    Returns
    -------
    List[Profile]
        A list of Profile objects, one per isolate (column in genomes_df).
    """
    profiles = []
    # Iterate over each isolate (column) in the genomes DataFrame.
    for isolate in genomes_df.columns:
        # Get the feature vector for the isolate.
        feature_vector = genomes_df[isolate]
        
        # Build a list of Feature objects.
        features = []
        for feat_id, value in feature_vector.items():
            # Here, we assume value is binary (0 or 1). You could extend this to incorporate abundance.
            feat = Feature(
                id=str(feat_id),
                name=str(feat_id),  # Alternatively, you might look up a descriptive name.
                presence=int(value),
                abundance=None
            )
            features.append(feat)
        
        # Extract taxonomy info if requested.
        taxonomy = {}
        if taxonomy_cols is not None and isolate in genome_info_df.index:
            taxonomy = genome_info_df.loc[isolate, taxonomy_cols].to_dict()
        
        # Include all metadata available for the isolate.
        metadata = {}
        if isolate in genome_info_df.index:
            metadata = genome_info_df.loc[isolate].to_dict()
        
        # Create the Profile object.
        profile = Profile(
            id=str(isolate),
            name=str(isolate),
            features=features,
            profile_type=profile_type,
            taxonomy=taxonomy,
            metadata=metadata
        )
        profiles.append(profile)
    return profiles
