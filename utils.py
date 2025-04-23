# utils.py
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class TextGridProcessor:
    """Class for processing TextGrid files to extract timing information."""
    
    def __init__(self, word_tier_name: str = "words", phone_tier_name: str = "phones"):
        """
        Initialize the TextGrid processor.
        
        Args:
            word_tier_name: Name of the tier containing words
            phone_tier_name: Name of the tier containing phones
        """
        self.word_tier_name = word_tier_name
        self.phone_tier_name = phone_tier_name
        logger.info(f"Initializing TextGrid processor with word tier '{word_tier_name}' and phone tier '{phone_tier_name}'")
    
    def process_textgrid(self, textgrid_path: str) -> List[Dict[str, Any]]:
        """
        Process a TextGrid file to extract segments with timing information.
        
        Args:
            textgrid_path: Path to the TextGrid file
        
        Returns:
            List of segments with timing information
        """
        logger.info(f"Processing TextGrid file: {textgrid_path}")
        
        try:
            # Try to import textgrid module
            try:
                import textgrid
            except ImportError:
                logger.error("Failed to import textgrid. Install with: pip install textgrid")
                raise
            
            # Load the TextGrid file
            tg = textgrid.TextGrid.fromFile(textgrid_path)
            
            # Find the relevant tiers
            word_tier = None
            phone_tier = None
            
            for tier in tg.tiers:
                if tier.name == self.word_tier_name:
                    word_tier = tier
                elif tier.name == self.phone_tier_name:
                    phone_tier = tier
            
            if word_tier is None:
                logger.warning(f"Word tier '{self.word_tier_name}' not found in TextGrid")
                word_tier = tg.tiers[0]  # Use the first tier as fallback
            
            # Extract sentences based on punctuation or long silences
            segments = self._extract_segments(word_tier)
            
            # Add word-level information to each segment
            for segment in segments:
                segment["words"] = self._extract_words(
                    word_tier, segment["start"], segment["end"]
                )
            
            logger.info(f"Extracted {len(segments)} segments from TextGrid")
            return segments
            
        except Exception as e:
            logger.error(f"Error processing TextGrid file: {e}")
            # Return a minimal segment covering the entire file
            return [{
                "start": 0.0,
                "end": 60.0,  # Assume 1 minute as default
                "text": "",
                "words": []
            }]
    
    def _extract_segments(self, word_tier) -> List[Dict[str, Any]]:
        """
        Extract sentence-level segments from the word tier.
        
        Args:
            word_tier: The word tier from the TextGrid
        
        Returns:
            List of segment dictionaries with start/end times and text
        """
        segments = []
        current_segment_start = 0.0
        current_segment_words = []
        
        # Punctuation that typically ends a sentence
        sentence_end_markers = [".", "!", "?", "。", "！", "？"]
        
        # Minimum silence duration to consider as sentence boundary (in seconds)
        min_silence = 0.5
        
        for i, interval in enumerate(word_tier):
            word = interval.mark.strip() if interval.mark else ""
            
            # Skip empty intervals
            if not word:
                # Check if this is a long silence that should break the segment
                if interval.duration() >= min_silence and current_segment_words:
                    # End the current segment before this silence
                    segment_text = " ".join(current_segment_words)
                    segments.append({
                        "start": current_segment_start,
                        "end": interval.minTime,
                        "text": segment_text,
                        "duration": interval.minTime - current_segment_start
                    })
                    
                    # Start a new segment after this silence
                    current_segment_start = interval.maxTime
                    current_segment_words = []
                
                continue
            
            # Add the word to the current segment
            current_segment_words.append(word)
            
            # Check if this word ends with sentence-final punctuation
            if any(word.endswith(marker) for marker in sentence_end_markers):
                # End the current segment
                segment_text = " ".join(current_segment_words)
                segments.append({
                    "start": current_segment_start,
                    "end": interval.maxTime,
                    "text": segment_text,
                    "duration": interval.maxTime - current_segment_start
                })
                
                # Start a new segment
                if i < len(word_tier) - 1:
                    current_segment_start = interval.maxTime
                    current_segment_words = []
        
        # Add the last segment if not empty
        if current_segment_words:
            last_interval = word_tier[-1]
            segment_text = " ".join(current_segment_words)
            segments.append({
                "start": current_segment_start,
                "end": last_interval.maxTime,
                "text": segment_text,
                "duration": last_interval.maxTime - current_segment_start
            })
        
        return segments
    
    def _extract_words(
        self, word_tier, start_time: float, end_time: float
    ) -> List[Dict[str, Any]]:
        """
        Extract word-level information within a segment.
        
        Args:
            word_tier: The word tier from the TextGrid
            start_time: Start time of the segment
            end_time: End time of the segment
        
        Returns:
            List of word dictionaries with timing information
        """
        words = []
        
        for interval in word_tier:
            # Skip intervals outside the segment time range
            if interval.maxTime <= start_time or interval.minTime >= end_time:
                continue
            
            word = interval.mark.strip() if interval.mark else ""
            
            # Skip empty intervals
            if not word:
                continue
            
            words.append({
                "word": word,
                "start": interval.minTime,
                "end": interval.maxTime,
                "duration": interval.duration()
            })
        
        return words


class SyllableCounter:
    """Utility class for counting syllables in different languages."""
    
    def __init__(self):
        """Initialize the syllable counter."""
        # Language-specific patterns for syllable counting
        self.patterns = {
            "en": self._count_english_syllables,
            "fr": self._count_french_syllables,
            "es": self._count_spanish_syllables,
            "de": self._count_german_syllables,
            "it": self._count_italian_syllables,
            "ko": self._count_korean_syllables
        }
    
    def count_syllables(self, text: str, lang: str) -> int:
        """
        Count syllables in the given text for the specified language.
        
        Args:
            text: Input text
            lang: Language code
        
        Returns:
            Number of syllables
        """
        if lang in self.patterns:
            return self.patterns[lang](text)
        else:
            # Fallback: count vowel sequences
            return self._count_vowel_sequences(text)
    
    def _count_english_syllables(self, text: str) -> int:
        """Count syllables in English text."""
        text = text.lower()
        text = re.sub(r'[^a-z]', ' ', text)
        words = text.split()
        
        count = 0
        for word in words:
            word = word.strip()
            if not word:
                continue
            
            # Count vowel groups
            vowel_groups = re.findall(r'[aeiouy]+', word)
            
            # Adjust for common patterns
            if word.endswith('e'):
                if len(vowel_groups) > 0:
                    count += len(vowel_groups) - 1
                else:
                    count += 1
            else:
                count += len(vowel_groups)
            
            # Ensure at least one syllable per word
            if count == 0 or len(vowel_groups) == 0:
                count += 1
        
        return count
    
    def _count_french_syllables(self, text: str) -> int:
        """Count syllables in French text."""
        text = text.lower()
        text = re.sub(r'[^a-zàâäæçéèêëîïôœùûüÿ]', ' ', text)
        
        # Replace specific patterns
        text = re.sub(r'([aeiouàâäæéèêëîïôœùûüÿ])e([aeiouàâäæéèêëîïôœùûüÿ])', r'\1\2', text)
        
        # Count vowel groups
        vowel_groups = re.findall(r'[aeiouàâäæéèêëîïôœùûüÿ]+', text)
        
        return max(1, len(vowel_groups))
    
    def _count_spanish_syllables(self, text: str) -> int:
        """Count syllables in Spanish text."""
        text = text.lower()
        text = re.sub(r'[^a-záéíóúüñ]', ' ', text)
        
        # Handle diphthongs and triphthongs
        text = re.sub(r'([aeiouáéíóú])([iuy])([aeiouáéíóú])', r'\1 \2 \3', text)
        
        # Count vowel groups
        vowel_groups = re.findall(r'[aeiouáéíóúü]+', text)
        
        return max(1, len(vowel_groups))
    
    def _count_german_syllables(self, text: str) -> int:
        """Count syllables in German text."""
        text = text.lower()
        text = re.sub(r'[^a-zäöüß]', ' ', text)
        
        # Count vowel groups
        vowel_groups = re.findall(r'[aeiouäöü]+', text)
        
        return max(1, len(vowel_groups))
    
    def _count_italian_syllables(self, text: str) -> int:
        """Count syllables in Italian text."""
        text = text.lower()
        text = re.sub(r'[^a-zàèéìíîòóùú]', ' ', text)
        
        # Handle specific patterns
        text = re.sub(r'([aeiouàèéìíîòóùú])i([aeiouàèéìíîòóùú])', r'\1 \2', text)
        
        # Count vowel groups
        vowel_groups = re.findall(r'[aeiouàèéìíîòóùú]+', text)
        
        return max(1, len(vowel_groups))
    
    def _count_korean_syllables(self, text: str) -> int:
        """
        Count syllables in Korean text.
        In Korean, each syllable block (Hangul character) represents one syllable.
        """
        # Count Hangul characters (Unicode range)
        hangul_count = len(re.findall(r'[\uAC00-\uD7A3]', text))
        
        # Add non-Hangul vowels (for mixed text)
        vowel_groups = len(re.findall(r'[aeiouAEIOU]+', text))
        
        return max(1, hangul_count + vowel_groups)
    
    def _count_vowel_sequences(self, text: str) -> int:
        """Generic syllable counter for unsupported languages."""
        text = text.lower()
        text = re.sub(r'[^a-z\u00C0-\u00FF]', ' ', text)
        
        # Count vowel groups (considering common vowels in European languages)
        vowel_groups = re.findall(r'[aeiouàáâäæèéêëìíîïòóôöøùúûüÿ]+', text)
        
        return max(1, len(vowel_groups))


class SpeakingRateEstimator:
    """Utility class for estimating speaking rates."""
    
    def __init__(self):
        """Initialize the speaking rate estimator."""
        self.syllable_counter = SyllableCounter()
        
        # Typical speaking rates in syllables per second for different languages
        self.typical_rates = {
            "en": 4.0,  # English: ~4 syllables per second
            "fr": 4.2,  # French: ~4.2 syllables per second
            "de": 3.8,  # German: ~3.8 syllables per second
            "es": 4.5,  # Spanish: ~4.5 syllables per second
            "it": 4.6,  # Italian: ~4.6 syllables per second
            "ko": 4.0   # Korean: ~4.0 syllables per second
        }
    
    def estimate_duration(self, text: str, lang: str) -> float:
        """
        Estimate the duration of the text when spoken.
        
        Args:
            text: Input text
            lang: Language code
        
        Returns:
            Estimated duration in seconds
        """
        # Count syllables
        syllable_count = self.syllable_counter.count_syllables(text, lang)
        
        # Get typical speaking rate for this language (or default to English)
        rate = self.typical_rates.get(lang, self.typical_rates["en"])
        
        # Calculate duration
        duration = syllable_count / rate
        
        # Add a small pause at the end
        duration += 0.2
        
        return duration
    
    def calculate_speaking_rate(self, text: str, duration: float, lang: str) -> float:
        """
        Calculate speaking rate for given text and duration.
        
        Args:
            text: Input text
            duration: Actual duration in seconds
            lang: Language code
        
        Returns:
            Speaking rate (syllables per second)
        """
        # Count syllables
        syllable_count = self.syllable_counter.count_syllables(text, lang)
        
        # Calculate rate
        if duration > 0:
            rate = syllable_count / duration
        else:
            rate = 0
        
        return rate
    
    def is_speaking_rate_natural(self, rate: float, lang: str) -> bool:
        """
        Check if the speaking rate is within natural range for the language.
        
        Args:
            rate: Speaking rate in syllables per second
            lang: Language code
        
        Returns:
            True if the rate is natural, False otherwise
        """
        # Get typical rate for this language
        typical_rate = self.typical_rates.get(lang, self.typical_rates["en"])
        
        # Define acceptable range (70% to 130% of typical rate)
        min_rate = typical_rate * 0.7
        max_rate = typical_rate * 1.3
        
        return min_rate <= rate <= max_rate