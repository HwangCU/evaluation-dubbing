# embedder.py
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class SentenceEmbedder:
    """Class for generating multilingual sentence embeddings."""
    
    def __init__(self, model_name: str = "LASER"):
        """
        Initialize the sentence embedder.
        
        Args:
            model_name: Name of the embedding model to use ('LASER', 'LaBSE', 'SBERT')
        """
        self.model_name = model_name.upper()
        logger.info(f"Initializing {self.model_name} sentence embedder")
        
        # Initialize the appropriate embedding model
        if self.model_name == "LASER":
            self._init_laser()
        elif self.model_name == "LABSE":
            self._init_labse()
        elif self.model_name == "SBERT":
            self._init_sbert()
        else:
            raise ValueError(f"Unsupported embedding model: {model_name}. Use 'LASER', 'LaBSE', or 'SBERT'")
    
    def _init_laser(self):
        """Initialize LASER embedding model."""
        try:
            from laserembeddings import Laser
            
            # Initialize LASER model
            self.model = Laser()
            logger.info("LASER model initialized successfully")
        except ImportError:
            logger.error("Failed to import laserembeddings. Install with: pip install laserembeddings")
            raise
    
    def _init_labse(self):
        """Initialize LaBSE embedding model."""
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            
            # Load LaBSE model from TF Hub
            self.model = hub.load('https://tfhub.dev/google/LaBSE/1')
            logger.info("LaBSE model initialized successfully")
        except ImportError:
            logger.error("Failed to import tensorflow or tensorflow_hub. Install with: pip install tensorflow tensorflow-hub")
            raise
    
    def _init_sbert(self):
        """Initialize Sentence-BERT embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Initialize multilingual Sentence-BERT model
            self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            logger.info("Sentence-BERT model initialized successfully")
        except ImportError:
            logger.error("Failed to import sentence_transformers. Install with: pip install sentence-transformers")
            raise
    
    def encode_sentences(self, sentences: List[str], lang: str) -> np.ndarray:
        """
        Encode the given sentences into embeddings.
        
        Args:
            sentences: List of sentences to encode
            lang: Language code of the sentences
        
        Returns:
            Array of embeddings with shape (len(sentences), embedding_dim)
        """
        if not sentences:
            logger.warning("Empty sentence list provided for encoding")
            return np.array([])
        
        logger.info(f"Encoding {len(sentences)} sentences in {lang} using {self.model_name}")
        
        if self.model_name == "LASER":
            # LASER requires language code for encoding
            embeddings = self.model.embed_sentences(sentences, lang=lang)
        
        elif self.model_name == "LABSE":
            # Preprocess sentences for LaBSE
            sentences_preprocessed = [s.lower().strip() for s in sentences]
            embeddings = self.model.signatures['serving_default'](
                tf.constant(sentences_preprocessed)
            )['default'].numpy()
            
            # Normalize embeddings to unit length
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        
        elif self.model_name == "SBERT":
            # Encode with Sentence-BERT
            embeddings = self.model.encode(sentences, convert_to_numpy=True)
        
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings