# evaluator.py
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class DubbingEvaluator:
    """Class for evaluating the quality of automatic dubbing."""
    
    def __init__(self):
        """Initialize the dubbing evaluator."""
        logger.info("Initializing dubbing evaluator")
    
    def evaluate(
        self,
        src_audio_path: str,
        src_segments: List[Dict[str, Any]],
        tgt_audio_path: str,
        aligned_segments: List[Dict[str, Any]],
        tts_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the quality of the dubbed audio.
        
        Args:
            src_audio_path: Path to source audio file
            src_segments: Source segments with timing information
            tgt_audio_path: Path to target audio file
            aligned_segments: Aligned target segments
            tts_results: Results from TTS synthesis
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating dubbing quality")
        
        # Calculate individual metrics
        isochrony_score = self._evaluate_isochrony(src_segments, aligned_segments)
        smoothness_score = self._evaluate_smoothness(aligned_segments)
        fluency_score = self._evaluate_fluency(aligned_segments)
        intelligibility_score = self._evaluate_intelligibility(tts_results)
        
        # Calculate overall score (weighted average)
        weights = {
            "isochrony": 0.3,
            "smoothness": 0.2,
            "fluency": 0.3,
            "intelligibility": 0.2
        }
        
        overall_score = (
            weights["isochrony"] * isochrony_score +
            weights["smoothness"] * smoothness_score +
            weights["fluency"] * fluency_score +
            weights["intelligibility"] * intelligibility_score
        )
        
        # Create results dictionary
        results = {
            "aligned": {
                "isochrony": round(isochrony_score, 3),
                "smoothness": round(smoothness_score, 3),
                "fluency": round(fluency_score, 3),
                "intelligibility": round(intelligibility_score, 3),
                "overall": round(overall_score, 3)
            }
        }
        
        # Compare with original (without alignment) if possible
        if all("target_duration" in result for result in tts_results):
            original_scores = self._evaluate_original(src_segments, tts_results)
            results["original"] = original_scores
        
        # Save results to file
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        with open(output_dir / "evaluation.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    def _evaluate_isochrony(
        self, 
        src_segments: List[Dict[str, Any]],
        aligned_segments: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate isochrony (temporal alignment) between source and target.
        
        Args:
            src_segments: Source segments
            aligned_segments: Aligned target segments
        
        Returns:
            Isochrony score (0-1)
        """
        # Match source and target segments
        matched_segments = self._match_segments(src_segments, aligned_segments)
        
        scores = []
        for src_seg, tgt_seg in matched_segments:
            # Calculate temporal overlap
            src_start, src_end = src_seg["start"], src_seg["end"]
            tgt_start, tgt_end = tgt_seg["start"], tgt_seg["end"]
            
            # Duration of the segments
            src_duration = src_end - src_start
            tgt_duration = tgt_end - tgt_start
            
            # Calculate overlap
            overlap_start = max(src_start, tgt_start)
            overlap_end = min(src_end, tgt_end)
            
            if overlap_end <= overlap_start:
                # No overlap
                overlap_duration = 0
            else:
                overlap_duration = overlap_end - overlap_start
            
            # Calculate overlap ratio
            overlap_ratio = overlap_duration / max(src_duration, tgt_duration)
            
            # Calculate timing error
            start_error = abs(src_start - tgt_start)
            end_error = abs(src_end - tgt_end)
            duration_error = abs(src_duration - tgt_duration)
            
            # Normalized error (as a fraction of source duration)
            normalized_error = (start_error + end_error) / (2 * src_duration)
            
            # Calculate score (1 - normalized error)
            score = max(0, 1 - normalized_error)
            
            # Store for on-screen segments (which need better isochrony)
            scores.append(score)
        
        # Calculate average score
        if scores:
            return sum(scores) / len(scores)
        else:
            return 0.0
    
    def _evaluate_smoothness(
        self, 
        aligned_segments: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate speaking rate smoothness across segments.
        
        Args:
            aligned_segments: Aligned target segments
        
        Returns:
            Smoothness score (0-1)
        """
        if len(aligned_segments) <= 1:
            return 1.0  # Perfect smoothness for single segment
        
        # Calculate speaking rate for each segment
        speaking_rates = []
        for segment in aligned_segments:
            duration = segment["duration"]
            text = segment["text"]
            
            # Simple speaking rate estimate: characters per second
            speaking_rate = len(text) / max(duration, 0.1)
            speaking_rates.append(speaking_rate)
        
        # Calculate speaking rate variation between consecutive segments
        variations = []
        for i in range(1, len(speaking_rates)):
            prev_rate = speaking_rates[i-1]
            curr_rate = speaking_rates[i]
            
            # Variation ratio (should be close to 1 for smooth speech)
            ratio = curr_rate / max(prev_rate, 0.01)
            
            # Penalize large variations
            variation = abs(1 - ratio)
            variations.append(variation)
        
        # Calculate average variation
        avg_variation = sum(variations) / len(variations) if variations else 0
        
        # Convert to score (1 - variation, capped at 0)
        smoothness_score = max(0, 1 - avg_variation)
        
        return smoothness_score
    
    def _evaluate_fluency(
        self, 
        aligned_segments: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate speech fluency within segments.
        
        Args:
            aligned_segments: Aligned target segments
        
        Returns:
            Fluency score (0-1)
        """
        # For fluency, we need to estimate if the speaking rate is natural
        scores = []
        
        # Import speaking rate estimator
        from utils import SpeakingRateEstimator
        rate_estimator = SpeakingRateEstimator()
        
        for segment in aligned_segments:
            text = segment["text"]
            duration = segment["duration"]
            
            # Calculate speaking rate in characters per second
            char_per_second = len(text) / max(duration, 0.1)
            
            # Penalize very high or very low speaking rates
            # Assuming normal range is ~10-20 chars/sec for English
            if 8 <= char_per_second <= 22:
                # Within normal range
                score = 1.0
            else:
                # Outside normal range, calculate penalty
                if char_per_second < 8:
                    # Too slow
                    score = char_per_second / 8
                else:
                    # Too fast
                    score = max(0, 1 - (char_per_second - 22) / 10)
            
            scores.append(score)
        
        # Calculate average score
        if scores:
            return sum(scores) / len(scores)
        else:
            return 0.0
    
    def _evaluate_intelligibility(
        self, 
        tts_results: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate speech intelligibility based on TTS results.
        
        Args:
            tts_results: Results from TTS synthesis
        
        Returns:
            Intelligibility score (0-1)
        """
        try:
            # If we have an actual speech recognition system available,
            # we can evaluate intelligibility by comparing ASR results to the original text
            # For now, we'll use a simple heuristic based on speaking rate
            
            scores = []
            
            for result in tts_results:
                text = result["text"]
                duration = result["duration"]
                
                # Calculate speaking rate in syllables per second
                # Rough approximation: 1 syllable per 3 characters on average
                char_count = len(text)
                syllable_count = char_count / 3
                syllables_per_second = syllable_count / max(duration, 0.1)
                
                # Penalize very high speaking rates (reduce intelligibility)
                if syllables_per_second <= 4.5:
                    # Normal range for clear speech
                    score = 1.0
                elif syllables_per_second <= 7:
                    # Faster but still intelligible
                    score = 1.0 - (syllables_per_second - 4.5) / 5
                else:
                    # Too fast to be intelligible
                    score = max(0, 1.0 - (syllables_per_second - 7) / 3)
                
                scores.append(score)
            
            # Calculate average score
            if scores:
                return sum(scores) / len(scores)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error evaluating intelligibility: {e}")
            return 0.5  # Default score
    
    def _evaluate_original(
        self, 
        src_segments: List[Dict[str, Any]],
        tts_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate quality metrics without alignment (original TTS).
        
        Args:
            src_segments: Source segments
            tts_results: TTS results with target durations
        
        Returns:
            Dictionary with evaluation metrics
        """
        # For original TTS (without alignment), we cannot calculate isochrony
        # We'll use natural TTS durations for smoothness and fluency
        
        # Create mock aligned segments using natural TTS durations
        mock_segments = []
        current_time = 0.0
        
        for result in tts_results:
            text = result["text"]
            duration = result.get("target_duration", 1.0)  # Use natural duration
            
            mock_segments.append({
                "text": text,
                "start": current_time,
                "end": current_time + duration,
                "duration": duration,
                "on_screen": False  # Assume off-screen for original
            })
            
            current_time += duration
        
        # Calculate metrics
        smoothness_score = self._evaluate_smoothness(mock_segments)
        fluency_score = self._evaluate_fluency(mock_segments)
        intelligibility_score = self._evaluate_intelligibility(tts_results)
        
        # Use a placeholder for isochrony
        isochrony_score = 0.0  # Cannot measure isochrony for original TTS
        
        # Calculate overall score
        weights = {
            "isochrony": 0.3,
            "smoothness": 0.2,
            "fluency": 0.3,
            "intelligibility": 0.2
        }
        
        overall_score = (
            weights["isochrony"] * isochrony_score +
            weights["smoothness"] * smoothness_score +
            weights["fluency"] * fluency_score +
            weights["intelligibility"] * intelligibility_score
        )
        
        return {
            "isochrony": round(isochrony_score, 3),
            "smoothness": round(smoothness_score, 3),
            "fluency": round(fluency_score, 3),
            "intelligibility": round(intelligibility_score, 3),
            "overall": round(overall_score, 3)
        }
    
    def _match_segments(
        self,
        src_segments: List[Dict[str, Any]],
        tgt_segments: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Match source and target segments based on timing.
        
        Args:
            src_segments: Source segments
            tgt_segments: Target segments
        
        Returns:
            List of (src_segment, tgt_segment) pairs
        """
        # Simple matching based on temporal overlap
        matched_pairs = []
        
        for src_seg in src_segments:
            src_start, src_end = src_seg["start"], src_seg["end"]
            best_match = None
            best_overlap = 0
            
            for tgt_seg in tgt_segments:
                tgt_start, tgt_end = tgt_seg["start"], tgt_seg["end"]
                
                # Calculate overlap
                overlap_start = max(src_start, tgt_start)
                overlap_end = min(src_end, tgt_end)
                
                if overlap_end > overlap_start:
                    overlap = overlap_end - overlap_start
                    
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = tgt_seg
            
            if best_match is not None:
                matched_pairs.append((src_seg, best_match))
        
        return matched_pairs