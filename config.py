# config.py
"""
Configuration parameters for the automatic dubbing system.
Modify these values to adjust the system behavior.
"""

# Sentence matcher parameters
MATCHER_CONFIG = {
    "method": "sequence_preserving",  # Options: "greedy", "hungarian", "relaxed", "sequence_preserving"
    "similarity_threshold": 0.2,       # Minimum similarity threshold for matches (lower = more matches)
    "force_all_matches": True,         # Force matching all segments even with low similarity
    "preserve_order": True,            # Preserve the original sequence order in matching
    "max_position_shift": 2            # Maximum position shift allowed when preserve_order is True
}

# Prosodic aligner parameters
ALIGNER_CONFIG = {
    "min_silence": 0.3,                # Minimum silence duration in seconds
    "use_relaxation": True,            # Whether to use time boundary relaxation
    "feature_weights": {               # Weights for alignment scoring features
        "lm": 0.2,                     # Language model score weight
        "cm": 0.3,                     # Cross-lingual semantic match weight
        "sv": 0.15,                    # Speaking rate variation weight
        "sm": 0.25,                    # Speaking rate match weight
        "is": 0.1                      # Isochrony score weight
    },
    "relaxation_factor_on_screen": 0.5,    # Max relaxation factor for on-screen segments (fraction of min_silence)
    "relaxation_factor_off_screen": 2.0,   # Max relaxation factor for off-screen segments (fraction of min_silence)
}

# TTS parameters
TTS_CONFIG = {
    "engine": "neural",                # TTS engine: "neural", "amazon", "google"
    "voice_style": "neutral",          # Voice style: "neutral", "conversational", "formal"
    "speaking_rate": 1.0,              # Speaking rate multiplier (1.0 = normal)
    "pitch": 0.0,                      # Pitch adjustment (-10.0 to 10.0)
    "volume": 1.0                      # Volume (0.0 to 2.0)
}

# Audio renderer parameters
RENDERER_CONFIG = {
    "use_source_separation": True,     # Extract background noise from original audio
    "add_reverberation": True,         # Add reverberation matching original audio
    "background_volume": 0.2,          # Background noise volume (0.0 to 1.0)
    "output_sample_rate": 22050,       # Output sample rate in Hz
    "normalize_audio": True            # Normalize audio levels
}

# Evaluator parameters
EVALUATOR_CONFIG = {
    "evaluation_weights": {            # Weights for overall score calculation
        "isochrony": 0.3,              # Weight for isochrony score
        "smoothness": 0.2,             # Weight for smoothness score
        "fluency": 0.3,                # Weight for fluency score
        "intelligibility": 0.2         # Weight for intelligibility score
    }
}

# Embedding model parameters
EMBEDDER_CONFIG = {
    "model_name": "LASER",             # Options: "LASER", "LaBSE", "SBERT"
    "cache_embeddings": True           # Cache embeddings to improve performance
}

# Default processing settings
DEFAULT_CONFIG = {
    "src_lang": "ko",                  # Source language code
    "tgt_lang": "en",                  # Target language code
    "default_on_screen": False,        # Default value for on-screen segments if not specified
    "output_format": "wav",            # Output audio format
    "save_intermediate_files": True    # Save intermediate files for debugging
}