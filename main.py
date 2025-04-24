# main.py (improved version)
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import os
# Import modules
from embedder import SentenceEmbedder
from matcher import SentenceMatcher
from aligner import ProsodicAligner
from evaluator import DubbingEvaluator
from renderer import AudioRenderer
from utils import TextGridProcessor
from config import DEFAULT_CONFIG, MATCHER_CONFIG, ALIGNER_CONFIG, EMBEDDER_CONFIG, TTS_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutomaticDubbingPipeline:
    """Main pipeline for automatic dubbing."""
    
    def __init__(
        self, 
        src_lang: str = None, 
        tgt_lang: str = None,
        embedding_model: str = None,
        use_relaxation: bool = None,
        min_silence: float = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the automatic dubbing pipeline.
        
        Args:
            src_lang: Source language code (overrides config)
            tgt_lang: Target language code (overrides config)
            embedding_model: Model for embeddings (overrides config)
            use_relaxation: Whether to use relaxation (overrides config)
            min_silence: Minimum silence duration (overrides config)
            config: Configuration dictionary (if None, uses defaults from config.py)
        """
        # Load configuration
        self.config = config or {
            **DEFAULT_CONFIG,
            "matcher": MATCHER_CONFIG,
            "aligner": ALIGNER_CONFIG,
            "embedder": EMBEDDER_CONFIG,
            "tts": TTS_CONFIG
        }
        
        # Override config with any provided parameters
        self.src_lang = src_lang or self.config.get("src_lang")
        self.tgt_lang = tgt_lang or self.config.get("tgt_lang")
        self.min_silence = min_silence or self.config.get("aligner", {}).get("min_silence")
        self.use_relaxation = use_relaxation if use_relaxation is not None else self.config.get("aligner", {}).get("use_relaxation")
        embedding_model = embedding_model or self.config.get("embedder", {}).get("model_name")
        
        # Initialize components
        self.embedder = SentenceEmbedder(model_name=embedding_model)
        self.matcher = SentenceMatcher(config=self.config.get("matcher"))
        self.aligner = ProsodicAligner(
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            min_silence=self.min_silence,
            use_relaxation=self.use_relaxation,
            feature_weights=self.config.get("aligner", {}).get("feature_weights")
        )
        self.renderer = AudioRenderer()
        self.evaluator = DubbingEvaluator()
        self.textgrid_processor = TextGridProcessor()
        
        logger.info(f"Automatic Dubbing Pipeline initialized: {self.src_lang} → {self.tgt_lang}")
    
    def validate_config(self) -> None:
        """
        Validate that configuration parameters are properly loaded and applied.
        Logs config details to help with debugging.
        """
        logger.info("=== Configuration Validation ===")
        
        # Core settings
        logger.info(f"Source language: {self.src_lang} (from config: {self.config.get('src_lang')})")
        logger.info(f"Target language: {self.tgt_lang} (from config: {self.config.get('tgt_lang')})")
        logger.info(f"Min silence: {self.min_silence} (from config: {self.config.get('aligner', {}).get('min_silence')})")
        logger.info(f"Use relaxation: {self.use_relaxation} (from config: {self.config.get('aligner', {}).get('use_relaxation')})")
        
        # Matcher settings
        matcher_config = self.config.get("matcher", {})
        logger.info(f"Matcher method: {matcher_config.get('method')}")
        logger.info(f"Similarity threshold: {matcher_config.get('similarity_threshold')}")
        logger.info(f"Force all matches: {matcher_config.get('force_all_matches')}")
        logger.info(f"Preserve order: {matcher_config.get('preserve_order')}")
        
        # Aligner settings
        aligner_config = self.config.get("aligner", {})
        logger.info(f"Feature weights: {aligner_config.get('feature_weights')}")
        logger.info(f"Relaxation factor (on-screen): {aligner_config.get('relaxation_factor_on_screen')}")
        logger.info(f"Relaxation factor (off-screen): {aligner_config.get('relaxation_factor_off_screen')}")
        
        # TTS settings
        tts_config = self.config.get("tts", {})
        logger.info(f"TTS engine: {tts_config.get('engine')}")
        logger.info(f"Voice style: {tts_config.get('voice_style')}")
        logger.info(f"Speaking rate: {tts_config.get('speaking_rate')}")
        
        # Confirm config is passed to components
        logger.info(f"Matcher config loaded: {bool(self.matcher.config)}")
        logger.info(f"Matcher similarity threshold: {self.matcher.similarity_threshold}")
        
        # Check if feature weights were passed to aligner
        logger.info(f"Aligner feature weights: {self.aligner.feature_weights}")
        
        logger.info("=== End Configuration Validation ===")

    def process(
        self, 
        src_audio_path: str,
        src_textgrid_path: str,
        tgt_audio_path: str,
        tgt_textgrid_path: str,
        output_dir: Path,
        on_screen_segments: Optional[List[bool]] = None
    ) -> Dict:
        """
        Process the complete dubbing pipeline using pre-synthesized target audio.
        
        Args:
            src_audio_path: Path to source audio file
            src_textgrid_path: Path to source TextGrid file
            tgt_audio_path: Path to pre-synthesized target audio file
            tgt_textgrid_path: Path to target TextGrid file
            output_dir: Directory to save outputs
            on_screen_segments: List indicating whether each segment is on-screen
                               If None, all segments are considered off-screen
        
        Returns:
            Dict containing results and evaluation metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Process source TextGrid to get timing information
        src_segments = self.textgrid_processor.process_textgrid(src_textgrid_path)
        src_text = [seg['text'] for seg in src_segments]
        
        # Process target TextGrid
        tgt_segments = self.textgrid_processor.process_textgrid(tgt_textgrid_path)
        tgt_text = [seg['text'] for seg in tgt_segments]
        
        logger.info(f"Source segments: {len(src_segments)}")
        logger.info(f"Target segments: {len(tgt_segments)}")
        
        # If on_screen_segments not provided, use default from config
        if on_screen_segments is None:
            default_on_screen = self.config.get("default_on_screen", False)
            on_screen_segments = [default_on_screen] * len(src_segments)
        
        # Generate embeddings for source and target sentences
        src_embeddings = self.embedder.encode_sentences(src_text, lang=self.src_lang)
        tgt_embeddings = self.embedder.encode_sentences(tgt_text, lang=self.tgt_lang)
        
        # Match sentences based on semantic similarity
        matches = self.matcher.match_sentences(
            src_embeddings=src_embeddings,
            tgt_embeddings=tgt_embeddings,
            src_texts=src_text,
            tgt_texts=tgt_text
        )
        
        # Check if all source segments are matched
        matched_src_indices = set(match['src_idx'] for match in matches)
        if len(matched_src_indices) < len(src_segments):
            logger.warning(f"Not all source segments were matched. "
                          f"Matched {len(matched_src_indices)}/{len(src_segments)}")
        
        # Create a mapping from source index to corresponding target segment
        src_to_tgt_segment = {}
        for match in matches:
            src_idx = match['src_idx']
            tgt_idx = match['tgt_idx']
            
            if tgt_idx < len(tgt_segments):
                src_to_tgt_segment[src_idx] = tgt_segments[tgt_idx]
            else:
                logger.warning(f"Target index {tgt_idx} out of range")
        
        # Create aligned_segments with the same order as src_segments
        aligned_segments = []
        for i, src_segment in enumerate(src_segments):
            # If this source segment was matched, use the corresponding target
            if i in src_to_tgt_segment:
                tgt_segment = src_to_tgt_segment[i]
                is_on_screen = on_screen_segments[i] if i < len(on_screen_segments) else False
                
                # Apply alignment
                aligned_segment = self.aligner.align_segment(
                    src_segment=src_segment,
                    tgt_segment=tgt_segment,
                    is_on_screen=is_on_screen
                )
                
                aligned_segments.append(aligned_segment)
            else:
                # If not matched, create a placeholder with empty text
                # This ensures all source segments have a corresponding aligned segment
                logger.warning(f"Source segment {i} has no match, creating placeholder")
                aligned_segments.append({
                    "text": "",
                    "start": src_segment["start"],
                    "end": src_segment["end"],
                    "duration": src_segment["end"] - src_segment["start"],
                    "on_screen": on_screen_segments[i] if i < len(on_screen_segments) else False
                })
        
        # Prepare TTS results from existing audio
        tts_results = self._prepare_tts_results_from_audio(
            tgt_audio_path=tgt_audio_path,
            aligned_segments=aligned_segments
        )
        
        # Render final audio with background from original
        final_audio_path = output_dir / "dubbed_audio.wav"
        self.renderer.render(
            src_audio_path=src_audio_path,
            tts_audio_paths=[res['audio_path'] for res in tts_results],
            segment_timings=[(seg['start'], seg['end']) for seg in aligned_segments],
            output_path=final_audio_path
        )
        
        # Evaluate the dubbed audio
        evaluation = self.evaluator.evaluate(
            src_audio_path=src_audio_path,
            src_segments=src_segments,
            tgt_audio_path=final_audio_path,
            aligned_segments=aligned_segments,
            tts_results=tts_results
        )
        
        # Save alignment and evaluation results
        results = {
            "alignment": [
                {
                    "src_idx": match['src_idx'],
                    "tgt_idx": match['tgt_idx'],
                    "similarity": match['similarity'],
                    "src_text": src_text[match['src_idx']],
                    "tgt_text": tgt_text[match['tgt_idx']] if match['tgt_idx'] < len(tgt_text) else ""
                }
                for match in matches
            ],
            "segments": [
                {
                    "start": seg['start'],
                    "end": seg['end'],
                    "text": seg['text'],
                    "on_screen": seg.get('on_screen', False)
                }
                for seg in aligned_segments
            ],
            "evaluation": evaluation
        }
        
        # Save results to JSON
        with open(output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dubbing completed. Results saved to {output_dir}")
        return results
    
    def _prepare_tts_results_from_audio(
        self,
        tgt_audio_path: str,
        aligned_segments: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Prepare TTS results using existing target audio file.
        
        Args:
            tgt_audio_path: Path to target audio file
            aligned_segments: Aligned target segments
            
        Returns:
            List of dictionaries similar to TTS output
        """
        import librosa
        import soundfile as sf
        from pathlib import Path
        import tempfile
        
        logger.info(f"Processing pre-synthesized target audio: {tgt_audio_path}")
        
        # Load target audio
        tgt_audio, tgt_sr = librosa.load(tgt_audio_path, sr=None)
        
        # Create directory for segment audio files
        output_dir = Path("output/audio")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        results = []
        
        # Extract audio segments corresponding to each aligned segment
        for i, segment in enumerate(aligned_segments):
            text = segment['text']
            start_time = segment['start']
            end_time = segment['end']
            duration = end_time - start_time
            
            # If segment has empty text, create silent audio
            if not text.strip():
                # Create silent audio segment
                silent_audio = np.zeros(int(duration * tgt_sr))
                segment_path = output_dir / f"segment_{i}.wav"
                sf.write(segment_path, silent_audio, tgt_sr)
                
                results.append({
                    'text': text,
                    'audio_path': str(segment_path),
                    'duration': duration,
                    'target_duration': duration
                })
                continue
            
            # Calculate sample indices
            start_sample = int(start_time * tgt_sr)
            end_sample = int(end_time * tgt_sr)
            
            # Safety check to avoid out-of-bounds access
            if start_sample >= len(tgt_audio):
                logger.warning(f"Segment {i} start time ({start_time}s) is beyond audio length")
                start_sample = max(0, len(tgt_audio) - 1)
            
            if end_sample > len(tgt_audio):
                logger.warning(f"Segment {i} end time ({end_time}s) is beyond audio length")
                end_sample = len(tgt_audio)
            
            # Extract segment audio
            if start_sample < end_sample:
                segment_audio = tgt_audio[start_sample:end_sample]
            else:
                # Create silence if segment timing is invalid
                segment_audio = np.zeros(int(duration * tgt_sr))
            
            # Save segment audio
            segment_path = output_dir / f"segment_{i}.wav"
            sf.write(segment_path, segment_audio, tgt_sr)
            
            # Create result dictionary
            results.append({
                'text': text,
                'audio_path': str(segment_path),
                'duration': duration,
                'target_duration': duration  # Same as actual duration for pre-synthesized audio
            })
            
            logger.info(f"Extracted segment {i+1}/{len(aligned_segments)}: {text[:30]}...")
        
        return results

    def process_with_text(
        self, 
        src_audio_path: str,
        src_textgrid_path: str,
        tgt_text: List[str],
        output_dir: Path,
        on_screen_segments: Optional[List[bool]] = None,
        use_tts: bool = True
    ) -> Dict:
        """
        Process the dubbing pipeline with just target text (using TTS).
        
        Args:
            src_audio_path: Path to source audio file
            src_textgrid_path: Path to source TextGrid file
            tgt_text: List of target language sentences
            output_dir: Directory to save outputs
            on_screen_segments: List indicating whether each segment is on-screen
            use_tts: Whether to use TTS for synthesis (if False, only alignment is performed)
        
        Returns:
            Dict containing results and evaluation metrics
        """
        from tts import TextToSpeech
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Process source TextGrid to get timing information
        src_segments = self.textgrid_processor.process_textgrid(src_textgrid_path)
        src_text = [seg['text'] for seg in src_segments]
        
        logger.info(f"Source segments: {len(src_segments)}")
        logger.info(f"Target sentences: {len(tgt_text)}")
        
        # If on_screen_segments not provided, use default from config
        if on_screen_segments is None:
            default_on_screen = self.config.get("default_on_screen", False)
            on_screen_segments = [default_on_screen] * len(src_segments)
        
        # Generate embeddings for source and target sentences
        src_embeddings = self.embedder.encode_sentences(src_text, lang=self.src_lang)
        tgt_embeddings = self.embedder.encode_sentences(tgt_text, lang=self.tgt_lang)
        
        # Match sentences based on semantic similarity
        matches = self.matcher.match_sentences(
            src_embeddings=src_embeddings,
            tgt_embeddings=tgt_embeddings,
            src_texts=src_text,
            tgt_texts=tgt_text
        )
        
        # Check if all source segments are matched
        matched_src_indices = set(match['src_idx'] for match in matches)
        if len(matched_src_indices) < len(src_segments):
            logger.warning(f"Not all source segments were matched. "
                        f"Matched {len(matched_src_indices)}/{len(src_segments)}")
        
        # CHANGE: Ensure all target texts are used regardless of matching
        if len(tgt_text) <= len(src_segments):
            # If we have fewer target sentences than source segments, distribute them
            # Create a list of target sentences for each source segment
            tgt_sentences = [""] * len(src_segments)
            
            # First fill in matched sentences
            for match in matches:
                src_idx = match['src_idx']
                tgt_idx = match['tgt_idx']
                
                if src_idx < len(tgt_sentences) and tgt_idx < len(tgt_text):
                    tgt_sentences[src_idx] = tgt_text[tgt_idx]
            
            # Then fill in any remaining target sentences to empty slots
            used_tgt_indices = set(match['tgt_idx'] for match in matches if match['tgt_idx'] < len(tgt_text))
            unused_tgt_indices = [i for i in range(len(tgt_text)) if i not in used_tgt_indices]
            
            # Find empty slots in tgt_sentences
            empty_slots = [i for i, sentence in enumerate(tgt_sentences) if not sentence]
            
            # Fill empty slots with unused target sentences
            for slot_idx, tgt_idx in zip(empty_slots, unused_tgt_indices):
                if tgt_idx < len(tgt_text):
                    tgt_sentences[slot_idx] = tgt_text[tgt_idx]
                    
            # If we still have empty slots but used all target sentences,
            # combine remaining text into the last slot
            if empty_slots and not unused_tgt_indices:
                # Join any missing target text
                all_used_indices = set([match['tgt_idx'] for match in matches])
                missing_indices = [i for i in range(len(tgt_text)) if i not in all_used_indices]
                if missing_indices:
                    missing_text = " ".join([tgt_text[i] for i in missing_indices])
                    if empty_slots:
                        tgt_sentences[empty_slots[0]] = missing_text
        else:
            # If we have more target sentences than source segments, combine them
            tgt_sentences = [""] * len(src_segments)
            
            # First, handle matched sentences
            for match in matches:
                src_idx = match['src_idx']
                tgt_idx = match['tgt_idx']
                
                if src_idx < len(tgt_sentences) and tgt_idx < len(tgt_text):
                    tgt_sentences[src_idx] = tgt_text[tgt_idx]
            
            # For any remaining target sentences, append them to the last segment
            matched_tgt_indices = set(match['tgt_idx'] for match in matches if match['tgt_idx'] < len(tgt_text))
            unmatched_tgt_indices = [i for i in range(len(tgt_text)) if i not in matched_tgt_indices]
            
            if unmatched_tgt_indices:
                unmatched_text = " ".join([tgt_text[i] for i in unmatched_tgt_indices])
                if tgt_sentences[-1]:
                    tgt_sentences[-1] += " " + unmatched_text
                else:
                    tgt_sentences[-1] = unmatched_text
        
        # CRITICAL CHECK: Make sure all target text is included somewhere
        all_tgt_text = " ".join(tgt_text).replace(" ", "").lower()
        all_assigned_text = " ".join(tgt_sentences).replace(" ", "").lower()
        
        if all_tgt_text != all_assigned_text:
            logger.warning(f"Not all target text was assigned to segments!")
            logger.warning(f"Original: {all_tgt_text}")
            logger.warning(f"Assigned: {all_assigned_text}")
            
            # Combine all target sentences if matching failed
            if len(src_segments) >= 1:
                tgt_sentences[0] = " ".join(tgt_text)
                for i in range(1, len(tgt_sentences)):
                    tgt_sentences[i] = ""
        
        # Perform prosodic alignment for all segments
        aligned_segments = []
        for i, (src_segment, tgt_sentence) in enumerate(zip(src_segments, tgt_sentences)):
            is_on_screen = on_screen_segments[i] if i < len(on_screen_segments) else False
            
            if tgt_sentence:
                # Apply alignment for segments with matched text
                aligned_segment = self.aligner.align_from_text_segment(
                    src_segment=src_segment,
                    tgt_sentence=tgt_sentence,
                    is_on_screen=is_on_screen
                )
                aligned_segments.append(aligned_segment)
            else:
                # Create placeholder for segments without matched text
                logger.warning(f"Source segment {i} has no matching target text, creating placeholder")
                aligned_segments.append({
                    "text": "",
                    "start": src_segment["start"],
                    "end": src_segment["end"],
                    "duration": src_segment["end"] - src_segment["start"],
                    "on_screen": is_on_screen
                })
        
        # CHANGE: Ensure we actually have all the text represented in our segments
        combined_aligned_text = " ".join([seg["text"] for seg in aligned_segments]).replace(" ", "").lower()
        if combined_aligned_text != all_tgt_text:
            logger.warning("Text mismatch after alignment. Forcing full text in first segment.")
            if aligned_segments:
                # Put all text in the first segment as a fallback
                aligned_segments[0]["text"] = " ".join(tgt_text)
                for i in range(1, len(aligned_segments)):
                    aligned_segments[i]["text"] = ""
        
        if use_tts:
            # Initialize TTS engine with config parameters
            tts_config = self.config.get("tts", {})
            tts = TextToSpeech(
                lang=self.tgt_lang,
                voice_id=tts_config.get("voice_id"),
                engine=tts_config.get("engine", "neural")
            )
            
            # Generate TTS for aligned segments
            tts_results = tts.synthesize(
                sentences=[seg['text'] for seg in aligned_segments if seg['text'].strip()],  # Only synthesize non-empty segments
                durations=[seg['duration'] for seg in aligned_segments if seg['text'].strip()]
            )
            
            # Map the TTS results back to all segments
            all_tts_results = []
            tts_idx = 0
            for seg in aligned_segments:
                if seg['text'].strip():
                    # Use the generated TTS result for this segment
                    if tts_idx < len(tts_results):
                        all_tts_results.append(tts_results[tts_idx])
                        tts_idx += 1
                    else:
                        # Fallback for unexpected case
                        logger.warning(f"Missing TTS result for segment '{seg['text']}'")
                        all_tts_results.append({
                            'text': seg['text'],
                            'audio_path': '',
                            'duration': seg['duration'],
                            'target_duration': seg['duration']
                        })
                else:
                    # Create silent audio for empty segments
                    all_tts_results.append({
                        'text': '',
                        'audio_path': '',
                        'duration': seg['duration'],
                        'target_duration': seg['duration']
                    })
            
            # Render final audio with background from original
            final_audio_path = output_dir / "dubbed_audio.wav"
            self.renderer.render(
                src_audio_path=src_audio_path,
                tts_audio_paths=[res['audio_path'] for res in all_tts_results if res['audio_path']],
                segment_timings=[(seg['start'], seg['end']) for seg in aligned_segments if seg['text'].strip()],
                output_path=final_audio_path
            )
            
            # Evaluate the dubbed audio
            evaluation = self.evaluator.evaluate(
                src_audio_path=src_audio_path,
                src_segments=src_segments,
                tgt_audio_path=final_audio_path,
                aligned_segments=[seg for seg in aligned_segments if seg['text'].strip()],
                tts_results=[res for res in all_tts_results if res['text'].strip()]
            )
        else:
            # Skip TTS and rendering, just return alignment results
            evaluation = {
                "aligned": {
                    "isochrony": 0.0,
                    "smoothness": 0.0,
                    "fluency": 0.0,
                    "intelligibility": 0.0,
                    "overall": 0.0
                }
            }
        
        # Save alignment and evaluation results
        results = {
            "alignment": [
                {
                    "src_idx": match['src_idx'],
                    "tgt_idx": match['tgt_idx'],
                    "similarity": match['similarity'],
                    "src_text": src_text[match['src_idx']],
                    "tgt_text": tgt_text[match['tgt_idx']] if match['tgt_idx'] < len(tgt_text) else ""
                }
                for match in matches
            ],
            "segments": [
                {
                    "start": seg['start'],
                    "end": seg['end'],
                    "text": seg['text'],
                    "on_screen": seg.get('on_screen', False)
                }
                for seg in aligned_segments
            ],
            "evaluation": evaluation,
            "complete_text": " ".join(tgt_text)  # Add the full intended text for reference
        }
        
        # Save results to JSON
        with open(output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processing completed. Results saved to {output_dir}")
        return results

if __name__ == "__main__":
    # Example usage
    pipeline = AutomaticDubbingPipeline(
        src_lang="ko",
        tgt_lang="en",
        embedding_model="LASER",
        use_relaxation=True
    )
    
    # Example 1: Process with pre-synthesized target audio
    current_directory = os.getcwd()

    # 특정 세그먼트만 on-screen으로 지정
    results = pipeline.process(
    src_audio_path=os.path.join(current_directory, 'input/윤장목소리1.wav'),
    src_textgrid_path=os.path.join(current_directory, 'input/윤장목소리1.TextGrid'),
    tgt_audio_path=os.path.join(current_directory, 'input/mine_en.wav'),
    tgt_textgrid_path=os.path.join(current_directory, 'input/mine_en.Textgrid'),
    output_dir=Path("output"),
    # on_screen_segments=[False, True, False]  # 매개변수 생략 - 모두 off-screen으로 처리됨
    )

    # # Example 2: Process with target text only
    # results = pipeline.process_with_text(
    #     src_audio_path=os.path.join(current_directory, 'input/윤장목소리1.wav'),
    #     src_textgrid_path=os.path.join(current_directory, 'input/윤장목소리1.TextGrid'),
    #     tgt_text=["Hello.", "My name is Jo Yoon-jang.", "Nice to meet you.", "Please take care of me."],
    #     output_dir=Path("output")
    #     # on_screen_segments parameter omitted - default off-screen will be used
    # )
    
    print(f"Evaluation results: {results['evaluation']}")