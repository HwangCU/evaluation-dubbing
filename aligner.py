# aligner.py (수정 버전)
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import math

logger = logging.getLogger(__name__)

class ProsodicAligner:
    """
    Class for prosodic alignment of translated sentences to match the
    temporal structure of source speech.
    
    Implements the algorithms from:
    - "Improvements to Prosodic Alignment for Automatic Dubbing"
    - "Prosodic Alignment for Off-Screen Automatic Dubbing"
    """
    
    def __init__(
        self,
        src_lang: str = "ko",
        tgt_lang: str = "en",
        min_silence: float = 0.3,
        use_relaxation: bool = True,
        feature_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the prosodic aligner.
        
        Args:
            src_lang: Source language code
            tgt_lang: Target language code
            min_silence: Minimum silence duration in seconds
            use_relaxation: Whether to use time boundary relaxation
            feature_weights: Weights for the scoring features
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.min_silence = min_silence
        self.use_relaxation = use_relaxation
        
        # Default feature weights based on the papers
        self.feature_weights = feature_weights or {
            "lm": 0.2,         # Language model score
            "cm": 0.3,         # Cross-lingual semantic match
            "sv": 0.15,        # Speaking rate variation
            "sm": 0.25,        # Speaking rate match
            "is": 0.1          # Isochrony score
        }
        
        # Possible relaxation values as fractions of min_silence
        self.relaxation_values = [0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75, -1.0]
        
        # Isochrony penalty parameter (penalizes left relaxations more than right)
        self.alpha = 0.8
        
        logger.info(f"Initializing prosodic aligner: {src_lang} → {tgt_lang}, min_silence={min_silence}s")
        
        # Initialize language models for breakpoint scoring
        self._init_language_models()
    
    def _init_language_models(self):
        """Initialize language models for scoring breakpoints."""
        # This would typically involve loading pre-trained language models
        # For now, we'll use a simple n-gram based approach with pre-computed statistics
        
        # Simple POS transition probabilities for breakpoint scoring
        # These would typically be learned from a large corpus
        self.pos_transitions = {
            "en": {
                ("DET", "NOUN"): 0.05,   # Low probability of break between determiner and noun
                ("NOUN", "VERB"): 0.3,    # Medium probability of break between noun and verb
                ("VERB", "PUNCT"): 0.9,   # High probability of break before punctuation
                ("PUNCT", "DET"): 0.8,    # High probability of break after punctuation
                # Default for unknown transitions
                "default": 0.1
            },
            "fr": {
                # Similar transitions for French
                "default": 0.1
            },
            "de": {
                # Similar transitions for German
                "default": 0.1
            },
            "es": {
                # Similar transitions for Spanish
                "default": 0.1
            },
            "it": {
                # Similar transitions for Italian
                "default": 0.1
            },
            "ko": {
                # Similar transitions for Korean
                "default": 0.1
            }
        }
        
        logger.info("Language models initialized")
    
    def align(
        self,
        src_segments: List[Dict[str, Any]],
        tgt_segments: List[Dict[str, Any]],
        on_screen: Optional[List[bool]] = None
    ) -> List[Dict[str, Any]]:
        """
        Align target segments to match the prosodic structure of source segments.
        This version works with pre-existing target segments with timing information.
        
        Args:
            src_segments: List of source segments with timing information
            tgt_segments: List of target segments with timing information
            on_screen: List indicating whether each segment is on-screen
                      If None, all segments are considered off-screen
        
        Returns:
            List of aligned target segments with timing information
        """
        if len(src_segments) != len(tgt_segments):
            logger.warning(
                f"Number of source segments ({len(src_segments)}) does not match "
                f"number of target segments ({len(tgt_segments)}). "
                f"Some segments may be dropped or duplicated."
            )
        
        # If on_screen not provided, assume all segments are off-screen
        if on_screen is None:
            on_screen = [False] * len(src_segments)
        
        aligned_segments = []
        
        # Process each segment pair
        for i, (src_segment, tgt_segment, is_on_screen) in enumerate(
            zip(src_segments, tgt_segments, on_screen)
        ):
            logger.info(f"Aligning segment {i+1}/{len(src_segments)}: on_screen={is_on_screen}")
            
            # Extract source and target timing
            src_start, src_end = src_segment["start"], src_segment["end"]
            src_duration = src_end - src_start
            
            tgt_text = tgt_segment["text"]
            
            # Calculate new timing based on source timing and on_screen status
            if is_on_screen:
                # For on-screen segments, maintain isochrony
                aligned_start = src_start
                aligned_end = src_end
                aligned_duration = src_duration
            else:
                # For off-screen segments, we can be more flexible
                # Use target segment's natural duration if available, otherwise use source
                tgt_duration = tgt_segment.get("duration", src_duration)
                
                # Apply relaxation if enabled
                if self.use_relaxation:
                    # Allow more flexibility for off-screen segments
                    # Calculate relaxation based on speaking rates
                    tgt_natural_duration = self._estimate_natural_duration(tgt_text, self.tgt_lang)
                    speaking_rate_ratio = tgt_natural_duration / src_duration
                    
                    if speaking_rate_ratio > 1.2:
                        # Target needs more time, extend duration
                        extended_duration = min(tgt_natural_duration, src_duration * 1.5)
                        extra_time = extended_duration - src_duration
                        
                        # Add extra time at the end, unless there's not enough silence
                        aligned_start = src_start
                        aligned_end = src_end + extra_time
                        aligned_duration = aligned_end - aligned_start
                    elif speaking_rate_ratio < 0.8:
                        # Target needs less time, shorten duration
                        aligned_start = src_start
                        aligned_end = src_start + max(tgt_natural_duration, src_duration * 0.8)
                        aligned_duration = aligned_end - aligned_start
                    else:
                        # Speaking rates are close enough, use source timing
                        aligned_start = src_start
                        aligned_end = src_end
                        aligned_duration = src_duration
                else:
                    # No relaxation, use source timing
                    aligned_start = src_start
                    aligned_end = src_end
                    aligned_duration = src_duration
            
            # Create aligned segment
            aligned_segment = {
                "text": tgt_text,
                "start": aligned_start,
                "end": aligned_end,
                "duration": aligned_duration,
                "on_screen": is_on_screen
            }
            
            aligned_segments.append(aligned_segment)
        
        logger.info(f"Alignment completed: {len(aligned_segments)} target segments aligned")
        return aligned_segments
    
    def align_from_text(
        self,
        src_segments: List[Dict[str, Any]],
        tgt_sentences: List[str],
        on_screen: Optional[List[bool]] = None
    ) -> List[Dict[str, Any]]:
        """
        Align target sentences to match the prosodic structure of source segments.
        This version works with target text only (no pre-existing timing).
        
        Args:
            src_segments: List of source segments with timing information
            tgt_sentences: List of target sentences to align
            on_screen: List indicating whether each segment is on-screen
                      If None, all segments are considered off-screen
        
        Returns:
            List of aligned target segments with timing information
        """
        if len(src_segments) != len(tgt_sentences):
            logger.warning(
                f"Number of source segments ({len(src_segments)}) does not match "
                f"number of target sentences ({len(tgt_sentences)})"
            )
        
        # If on_screen not provided, assume all segments are off-screen
        if on_screen is None:
            on_screen = [False] * len(src_segments)
        
        aligned_segments = []
        
        # Process each segment and corresponding target sentence
        for i, (src_segment, tgt_sentence, is_on_screen) in enumerate(
            zip(src_segments, tgt_sentences, on_screen)
        ):
            logger.info(f"Aligning segment {i+1}/{len(src_segments)}: on_screen={is_on_screen}")
            
            # 1. Tokenize the target sentence
            tgt_tokens = self._tokenize(tgt_sentence, self.tgt_lang)
            
            # 2. Extract source segment timing information
            src_breaks = self._extract_breaks(src_segment)
            
            # 3. Compute optimal segmentation (breakpoints) for target
            if len(src_breaks) <= 1:
                # No internal breaks, just copy the timing
                tgt_breaks = [(0, len(tgt_tokens))]
            else:
                # Compute optimal segmentation using dynamic programming
                tgt_breaks = self._compute_optimal_segmentation(
                    src_segment=src_segment,
                    src_breaks=src_breaks,
                    tgt_tokens=tgt_tokens,
                    is_on_screen=is_on_screen
                )
            
            # 4. Convert token indices to character positions and apply time relaxation
            tgt_segments = self._convert_to_segments(
                tgt_tokens=tgt_tokens,
                tgt_breaks=tgt_breaks,
                src_breaks=src_breaks,
                src_segment=src_segment,
                is_on_screen=is_on_screen
            )
            
            aligned_segments.extend(tgt_segments)
        
        logger.info(f"Alignment completed: {len(aligned_segments)} target segments created")
        return aligned_segments
    
    def _tokenize(self, sentence: str, lang: str) -> List[str]:
        """
        Tokenize a sentence into words or sub-word units.
        
        Args:
            sentence: The sentence to tokenize
            lang: Language code
        
        Returns:
            List of tokens
        """
        # Simple space-based tokenization for demonstration
        # In a real implementation, this would use a language-specific tokenizer
        tokens = sentence.split()
        return tokens
    
    def _extract_breaks(self, segment: Dict[str, Any]) -> List[Tuple[float, float]]:
        """
        Extract breakpoints from source segment.
        
        Args:
            segment: Source segment with timing information
        
        Returns:
            List of (start_time, end_time) tuples for each phrase
        """
        # Check if the segment already has break information
        if "breaks" in segment:
            return segment["breaks"]
        
        # If no break information, assume the segment is a single phrase
        start_time = segment.get("start", 0.0)
        end_time = segment.get("end", 0.0)
        
        # Check if there are word-level timings
        if "words" in segment:
            words = segment["words"]
            breaks = []
            current_phrase_start = start_time
            
            # Find breaks based on silence between words
            for i in range(len(words) - 1):
                current_word_end = words[i].get("end", 0.0)
                next_word_start = words[i + 1].get("start", 0.0)
                
                # If silence duration exceeds min_silence, this is a break
                if next_word_start - current_word_end >= self.min_silence:
                    breaks.append((current_phrase_start, current_word_end))
                    current_phrase_start = next_word_start
            
            # Add the final phrase
            breaks.append((current_phrase_start, end_time))
            return breaks
        
        # If no word-level timing, return the entire segment as one phrase
        return [(start_time, end_time)]
    
    def _compute_optimal_segmentation(
        self,
        src_segment: Dict[str, Any],
        src_breaks: List[Tuple[float, float]],
        tgt_tokens: List[str],
        is_on_screen: bool
    ) -> List[Tuple[int, int]]:
        """
        Compute optimal segmentation for target tokens using dynamic programming.
        
        Args:
            src_segment: Source segment information
            src_breaks: List of (start_time, end_time) tuples for source phrases
            tgt_tokens: List of target tokens
            is_on_screen: Whether the segment is on-screen
        
        Returns:
            List of (start_idx, end_idx) tuples for target phrases
        """
        n_src_phrases = len(src_breaks)
        n_tgt_tokens = len(tgt_tokens)
        
        # Dynamic programming table: Q[j][t] represents the best score for
        # segmenting the first j tokens into t phrases
        Q = np.full((n_tgt_tokens + 1, n_src_phrases + 1), float('-inf'))
        backpointers = np.zeros((n_tgt_tokens + 1, n_src_phrases + 1), dtype=int)
        
        # Base case: empty target with empty source has score 0
        Q[0][0] = 0
        
        # Fill the DP table
        for j in range(1, n_tgt_tokens + 1):
            for t in range(1, min(j + 1, n_src_phrases + 1)):
                for j_prev in range(t - 1, j):
                    score = Q[j_prev][t - 1] + self._score_phrase(
                        tgt_tokens=tgt_tokens,
                        start_idx=j_prev,
                        end_idx=j,
                        src_phrase_idx=t - 1,
                        src_breaks=src_breaks,
                        src_segment=src_segment
                    )
                    
                    if score > Q[j][t]:
                        Q[j][t] = score
                        backpointers[j][t] = j_prev
        
        # Reconstruct the optimal segmentation
        segmentation = []
        j = n_tgt_tokens
        t = n_src_phrases
        
        while t > 0:
            j_prev = backpointers[j][t]
            segmentation.append((j_prev, j))
            j = j_prev
            t -= 1
        
        # Reverse to get phrases in the correct order
        segmentation.reverse()
        
        return segmentation
    
    def _score_phrase(
        self,
        tgt_tokens: List[str],
        start_idx: int,
        end_idx: int,
        src_phrase_idx: int,
        src_breaks: List[Tuple[float, float]],
        src_segment: Dict[str, Any]
    ) -> float:
        """
        Score a potential target phrase assignment.
        
        Args:
            tgt_tokens: Target tokens
            start_idx: Start index of target phrase
            end_idx: End index of target phrase
            src_phrase_idx: Index of the corresponding source phrase
            src_breaks: Source breakpoints
            src_segment: Source segment information
            
        Returns:
            Score for this phrase assignment
        """
        # Extract target phrase
        tgt_phrase = tgt_tokens[start_idx:end_idx]
        tgt_phrase_text = " ".join(tgt_phrase)
        
        # Extract source phrase
        src_start, src_end = src_breaks[src_phrase_idx]
        src_duration = src_end - src_start
        
        # Extract source phrase text based on timing
        src_phrase_text = self._get_source_phrase_text(src_segment, src_start, src_end)
        
        # Calculate various scores
        scores = {}
        
        # 1. Language model score - probability of break at this position
        scores["lm"] = self._language_model_score(tgt_tokens, start_idx, end_idx)
        
        # 2. Cross-lingual semantic match score - similarity between source and target phrases
        scores["cm"] = self._cross_lingual_semantic_match(src_phrase_text, tgt_phrase_text)
        
        # 3. Speaking rate variation score - consistency of speaking rate across phrases
        # For first phrase, use a default value
        if src_phrase_idx == 0:
            scores["sv"] = 1.0
        else:
            prev_src_start, prev_src_end = src_breaks[src_phrase_idx - 1]
            prev_src_duration = prev_src_end - prev_src_start
            prev_src_phrase_text = self._get_source_phrase_text(src_segment, prev_src_start, prev_src_end)
            
            # For simplicity, estimate speaking rate based on character count / duration
            curr_src_rate = len(src_phrase_text) / max(src_duration, 0.1)
            prev_src_rate = len(prev_src_phrase_text) / max(prev_src_duration, 0.01)
            
            # Speaking rate variation - penalize large changes
            rate_ratio = curr_src_rate / max(prev_src_rate, 0.01)
            scores["sv"] = 1.0 - min(abs(1.0 - rate_ratio), 1.0)
        
        # 4. Speaking rate match score - match between source and target phrases speaking rates
        # Estimate source speaking rate
        src_char_per_sec = len(src_phrase_text) / max(src_duration, 0.1)
        
        # Estimate target speaking rate (assuming a default rate for the target language)
        # This is typically measured using a TTS system, but we'll use a heuristic here
        typical_src_rate = 10.0  # characters per second for source language
        typical_tgt_rate = 8.0   # characters per second for target language
        
        # Scaling factor between languages
        rate_factor = typical_tgt_rate / typical_src_rate
        
        # Expected target rate
        expected_tgt_rate = src_char_per_sec * rate_factor
        
        # Actual target chars for this phrase
        tgt_chars = len(tgt_phrase_text)
        
        # Expected target duration
        expected_tgt_duration = tgt_chars / expected_tgt_rate
        
        # Speaking rate match - penalize if target duration deviates from source
        duration_ratio = expected_tgt_duration / max(src_duration, 0.1)
        scores["sm"] = 1.0 - min(abs(1.0 - duration_ratio), 1.0)
        
        # 5. Isochrony score - always 1.0 during segmentation phase
        scores["is"] = 1.0
        
        # Compute weighted score
        weighted_score = sum(
            self.feature_weights[feature] * score
            for feature, score in scores.items()
        )
        
        return weighted_score
    
    def _get_source_phrase_text(
        self, src_segment: Dict[str, Any], start_time: float, end_time: float
    ) -> str:
        """
        Extract source phrase text based on timing.
        
        Args:
            src_segment: Source segment information
            start_time: Start time of the phrase
            end_time: End time of the phrase
            
        Returns:
            Text of the source phrase
        """
        # If segment has word-level timings, extract words that fall within the phrase
        if "words" in src_segment:
            words = []
            for word_info in src_segment["words"]:
                word_start = word_info.get("start", 0.0)
                word_end = word_info.get("end", 0.0)
                
                # Include word if it overlaps with the phrase
                if word_end > start_time and word_start < end_time:
                    words.append(word_info.get("word", ""))
            
            return " ".join(words)
        
        # If no word-level timing, return the entire segment text
        return src_segment.get("text", "")
    
    def _language_model_score(self, tokens: List[str], start_idx: int, end_idx: int) -> float:
        """
        Calculate language model score for a breakpoint.
        
        Args:
            tokens: List of tokens
            start_idx: Start index of phrase
            end_idx: End index of phrase
            
        Returns:
            Language model score for breaking at this position
        """
        # For phrase-initial position
        if start_idx == 0:
            return 1.0
        
        # For simplicity, return a fixed score based on POS tags
        # In a real implementation, this would use a proper language model
        return 0.8
    
    def _cross_lingual_semantic_match(self, src_phrase: str, tgt_phrase: str) -> float:
        """
        Calculate cross-lingual semantic match score.
        
        Args:
            src_phrase: Source phrase text
            tgt_phrase: Target phrase text
            
        Returns:
            Semantic similarity score
        """
        # In a real implementation, this would use a multilingual embedding model
        # For now, just return a heuristic score based on length ratio
        src_len = len(src_phrase)
        tgt_len = len(tgt_phrase)
        
        if src_len == 0 or tgt_len == 0:
            return 0.0
        
        ratio = min(src_len, tgt_len) / max(src_len, tgt_len)
        return ratio
    
    def _convert_to_segments(
        self,
        tgt_tokens: List[str],
        tgt_breaks: List[Tuple[int, int]],
        src_breaks: List[Tuple[float, float]],
        src_segment: Dict[str, Any],
        is_on_screen: bool
    ) -> List[Dict[str, Any]]:
        """
        Convert token-level breakpoints to character-level segments with timing.
        
        Args:
            tgt_tokens: Target tokens
            tgt_breaks: Target token-level breakpoints
            src_breaks: Source time breakpoints
            src_segment: Source segment information
            is_on_screen: Whether the segment is on-screen
            
        Returns:
            List of target segments with timing information
        """
        segments = []
        
        # Process each target phrase
        for i, ((start_idx, end_idx), (src_start, src_end)) in enumerate(zip(tgt_breaks, src_breaks)):
            # Get text for this phrase
            phrase_tokens = tgt_tokens[start_idx:end_idx]
            phrase_text = " ".join(phrase_tokens)
            
            # Calculate base duration
            base_duration = src_end - src_start
            
            # Apply relaxation if enabled
            if self.use_relaxation:
                # Relaxation depends on whether the segment is on-screen
                if is_on_screen:
                    # On-screen segments: limited relaxation to maintain isochrony
                    delta_left, delta_right = self._compute_relaxation_on_screen(
                        phrase_tokens=phrase_tokens,
                        base_duration=base_duration,
                        src_phrase_idx=i,
                        src_breaks=src_breaks,
                        src_segment=src_segment
                    )
                else:
                    # Off-screen segments: more relaxation allowed
                    delta_left, delta_right = self._compute_relaxation_off_screen(
                        phrase_tokens=phrase_tokens,
                        base_duration=base_duration,
                        src_phrase_idx=i,
                        src_breaks=src_breaks,
                        src_segment=src_segment
                    )
                
                # Apply relaxations to timing
                relaxed_start = src_start - delta_left * self.min_silence
                relaxed_end = src_end + delta_right * self.min_silence
                
                # Ensure timing is valid
                relaxed_start = max(relaxed_start, 0)
                if i > 0:
                    # Ensure start time is after previous segment
                    prev_end = segments[-1]["end"]
                    relaxed_start = max(relaxed_start, prev_end)
                
                relaxed_end = max(relaxed_end, relaxed_start + 0.1)  # Minimum duration
                
                duration = relaxed_end - relaxed_start
            else:
                # No relaxation, use source timing
                relaxed_start = src_start
                relaxed_end = src_end
                duration = base_duration
            
            segments.append({
                "text": phrase_text,
                "start": relaxed_start,
                "end": relaxed_end,
                "duration": duration,
                "on_screen": is_on_screen
            })
        
        return segments
    
    def _compute_relaxation_on_screen(
        self,
        phrase_tokens: List[str],
        base_duration: float,
        src_phrase_idx: int,
        src_breaks: List[Tuple[float, float]],
        src_segment: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Compute relaxation values for on-screen segments.
        
        Args:
            phrase_tokens: Target phrase tokens
            base_duration: Base duration from source
            src_phrase_idx: Index of source phrase
            src_breaks: Source breaks
            src_segment: Source segment information
            
        Returns:
            Tuple of (delta_left, delta_right) relaxation values
        """
        # For on-screen segments, use a more conservative relaxation approach
        # to maintain isochrony
        
        # Estimate target phrase duration using TTS
        phrase_text = " ".join(phrase_tokens)
        char_count = len(phrase_text)
        
        # Simple heuristic: assume 10 chars/sec for target language
        estimated_duration = char_count / 10.0
        
        # Calculate expected speaking rate
        speaking_rate = estimated_duration / base_duration
        
        # No relaxation needed if speaking rate is close to 1.0
        if 0.9 <= speaking_rate <= 1.1:
            return 0.0, 0.0
        
        # For high speaking rates (> 1.1), extend the duration
        if speaking_rate > 1.1:
            # Calculate how much extra time needed
            extra_time_needed = estimated_duration - base_duration
            
            # Convert to relaxation units
            relaxation_units = min(extra_time_needed / self.min_silence, 1.0)
            
            # Distribute between left and right, with preference for right
            delta_right = min(relaxation_units * 0.7, 0.75)
            delta_left = min(relaxation_units - delta_right, 0.25)
            
            return delta_left, delta_right
        
        # For low speaking rates (< 0.9), reduce the duration
        if speaking_rate < 0.9:
            # Calculate how much time to remove
            time_to_remove = base_duration - estimated_duration
            
            # Convert to relaxation units (negative for contraction)
            relaxation_units = -min(time_to_remove / self.min_silence, 1.0)
            
            # Distribute between left and right, with preference for right
            delta_right = max(relaxation_units * 0.7, -0.75)
            delta_left = max(relaxation_units - delta_right, -0.25)
            
            return delta_left, delta_right
        
        return 0.0, 0.0
    
    def _compute_relaxation_off_screen(
        self,
        phrase_tokens: List[str],
        base_duration: float,
        src_phrase_idx: int,
        src_breaks: List[Tuple[float, float]],
        src_segment: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Compute relaxation values for off-screen segments.
        
        Args:
            phrase_tokens: Target phrase tokens
            base_duration: Base duration from source
            src_phrase_idx: Index of source phrase
            src_breaks: Source breaks
            src_segment: Source segment information
            
        Returns:
            Tuple of (delta_left, delta_right) relaxation values
        """
        # For off-screen segments, we can apply more aggressive relaxation
        # to optimize for speech naturalness
        
        # Estimate target phrase duration using TTS
        phrase_text = " ".join(phrase_tokens)
        char_count = len(phrase_text)
        
        # Simple heuristic: assume 10 chars/sec for target language
        estimated_duration = char_count / 10.0
        
        # Calculate speaking rate
        speaking_rate = estimated_duration / base_duration
        
        # Scoring function from the paper: prefer speaking rates <= 1.0
        if speaking_rate <= 1.0:
            # Speaking rate is good, minimal relaxation needed
            return 0.0, 0.0
        elif 1.0 < speaking_rate <= 2.0:
            # Linear penalty for rates between 1.0 and 2.0
            penalty = speaking_rate - 1.0
            
            # Calculate how much extra time needed
            extra_time_needed = estimated_duration - base_duration
            
            # Convert to relaxation units, limited by available silence
            max_relaxation = min(4.0, extra_time_needed / self.min_silence)
            
            # Distribute with preference for right relaxation
            if src_phrase_idx < len(src_breaks) - 1:
                # Not the last phrase, can relax right boundary
                delta_right = min(max_relaxation * 0.7, 4.0)
                delta_left = min(max_relaxation - delta_right, 1.0)
            else:
                # Last phrase, focus on left boundary
                delta_left = min(max_relaxation, 1.0)
                delta_right = 0.0
            
            return delta_left, delta_right
        else:
            # Speaking rate > 2.0 is too fast, apply maximum relaxation
            if src_phrase_idx < len(src_breaks) - 1:
                return 1.0, 4.0
            else:
                return 1.0, 0.0
    
    def _estimate_natural_duration(self, text: str, lang: str) -> float:
        """
        Estimate natural duration of text when spoken.
        
        Args:
            text: Text to estimate duration for
            lang: Language code
            
        Returns:
            Estimated duration in seconds
        """
        from utils import SpeakingRateEstimator
        
        # Create estimator
        estimator = SpeakingRateEstimator()
        
        # Estimate duration
        return estimator.estimate_duration(text, lang)

    def align_segment(
        self,
        src_segment: Dict[str, Any],
        tgt_segment: Dict[str, Any],
        is_on_screen: bool
    ) -> Dict[str, Any]:
        """
        Align a single target segment to match the prosodic structure of a source segment.
        This method aligns one segment at a time rather than processing lists.
        
        Args:
            src_segment: Source segment with timing information
            tgt_segment: Target segment with timing information
            is_on_screen: Whether the segment is on-screen
        
        Returns:
            Aligned target segment with timing information
        """
        logger.info(f"Aligning single segment: on_screen={is_on_screen}")
        
        # Extract source and target timing
        src_start, src_end = src_segment["start"], src_segment["end"]
        src_duration = src_end - src_start
        
        tgt_text = tgt_segment["text"]
        
        # Calculate new timing based on source timing and on_screen status
        if is_on_screen:
            # For on-screen segments, maintain isochrony
            aligned_start = src_start
            aligned_end = src_end
            aligned_duration = src_duration
        else:
            # For off-screen segments, we can be more flexible
            # Use target segment's natural duration if available, otherwise use source
            tgt_duration = tgt_segment.get("duration", src_duration)
            
            # Apply relaxation if enabled
            if self.use_relaxation:
                # Allow more flexibility for off-screen segments
                # Calculate relaxation based on speaking rates
                tgt_natural_duration = self._estimate_natural_duration(tgt_text, self.tgt_lang)
                speaking_rate_ratio = tgt_natural_duration / src_duration
                
                if speaking_rate_ratio > 1.2:
                    # Target needs more time, extend duration
                    extended_duration = min(tgt_natural_duration, src_duration * 1.5)
                    extra_time = extended_duration - src_duration
                    
                    # Add extra time at the end, unless there's not enough silence
                    aligned_start = src_start
                    aligned_end = src_end + extra_time
                    aligned_duration = aligned_end - aligned_start
                elif speaking_rate_ratio < 0.8:
                    # Target needs less time, shorten duration
                    aligned_start = src_start
                    aligned_end = src_start + max(tgt_natural_duration, src_duration * 0.8)
                    aligned_duration = aligned_end - aligned_start
                else:
                    # Speaking rates are close enough, use source timing
                    aligned_start = src_start
                    aligned_end = src_end
                    aligned_duration = src_duration
            else:
                # No relaxation, use source timing
                aligned_start = src_start
                aligned_end = src_end
                aligned_duration = src_duration
        
        # Create aligned segment
        aligned_segment = {
            "text": tgt_text,
            "start": aligned_start,
            "end": aligned_end,
            "duration": aligned_duration,
            "on_screen": is_on_screen
        }
        
        return aligned_segment
        
    def align_from_text_segment(
        self,
        src_segment: Dict[str, Any],
        tgt_sentence: str,
        is_on_screen: bool
    ) -> Dict[str, Any]:
        """
        Align a target sentence to match the prosodic structure of a source segment.
        This method aligns one segment at a time from text rather than processing lists.
        
        Args:
            src_segment: Source segment with timing information
            tgt_sentence: Target sentence to align
            is_on_screen: Whether the segment is on-screen
        
        Returns:
            Aligned target segment with timing information
        """
        logger.info(f"Aligning single segment from text: on_screen={is_on_screen}")
        
        # 1. Tokenize the target sentence
        tgt_tokens = self._tokenize(tgt_sentence, self.tgt_lang)
        
        # 2. Extract source segment timing information
        src_breaks = self._extract_breaks(src_segment)
        
        # 3. Compute optimal segmentation (breakpoints) for target
        if len(src_breaks) <= 1:
            # No internal breaks, just copy the timing
            tgt_breaks = [(0, len(tgt_tokens))]
        else:
            # Compute optimal segmentation using dynamic programming
            tgt_breaks = self._compute_optimal_segmentation(
                src_segment=src_segment,
                src_breaks=src_breaks,
                tgt_tokens=tgt_tokens,
                is_on_screen=is_on_screen
            )
        
        # 4. Convert token indices to character positions and apply time relaxation
        tgt_segments = self._convert_to_segments(
            tgt_tokens=tgt_tokens,
            tgt_breaks=tgt_breaks,
            src_breaks=src_breaks,
            src_segment=src_segment,
            is_on_screen=is_on_screen
        )
        
        # For a single segment alignment, we expect only one result
        if tgt_segments:
            return tgt_segments[0]
        else:
            # If no segments were created, return a placeholder
            return {
                "text": tgt_sentence,
                "start": src_segment["start"],
                "end": src_segment["end"],
                "duration": src_segment["end"] - src_segment["start"],
                "on_screen": is_on_screen
            }