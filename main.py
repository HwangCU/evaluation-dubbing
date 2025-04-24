# main_improved.py
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import os
# Import modules
from embedder import SentenceEmbedder
from matcher import SentenceMatcher
from aligner import ProsodicAligner
from evaluator import DubbingEvaluator
from renderer import AudioRenderer
from utils import TextGridProcessor
from tts import TextToSpeech
from tts_optimizer import TTSOptimizer
from prosody_analyzer import ProsodyAnalyzer
from tts_feedback_loop import TTSFeedbackLoop
from config import DEFAULT_CONFIG, MATCHER_CONFIG, ALIGNER_CONFIG, EMBEDDER_CONFIG, TTS_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutomaticDubbingPipeline:
    """Main pipeline for automatic dubbing with advanced feedback mechanism."""
    
    def __init__(
        self, 
        src_lang: str = None, 
        tgt_lang: str = None,
        embedding_model: str = None,
        use_relaxation: bool = None,
        min_silence: float = None,
        config: Optional[Dict] = None,
        enable_feedback: bool = True,  # 피드백 루프 활성화 여부
        feedback_iterations: int = 3   # 피드백 루프 최대 반복 횟수
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
            enable_feedback: Whether to enable feedback loop mechanism
            feedback_iterations: Maximum number of feedback iterations
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
        self.enable_feedback = enable_feedback
        
        # Initialize components
        self.embedder = SentenceEmbedder(model_name=embedding_model or self.config.get("embedder", {}).get("model_name"))
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
        
        # Initialize feedback components if enabled
        if self.enable_feedback:
            self.feedback_loop = TTSFeedbackLoop(
                max_iterations=feedback_iterations,
                improvement_threshold=0.05,
                similarity_threshold=0.85
            )
        
        logger.info(f"Automatic Dubbing Pipeline initialized: {self.src_lang} → {self.tgt_lang}")
        logger.info(f"Feedback loop: {'Enabled' if self.enable_feedback else 'Disabled'}")
    
    def process_with_text(
        self, 
        src_audio_path: str,
        src_textgrid_path: str,
        tgt_text: List[str],
        output_dir: Path,
        on_screen_segments: Optional[List[bool]] = None,
        use_tts: bool = True,
        use_feedback: Optional[bool] = None  # 피드백 메커니즘 사용 여부 (None이면 클래스 설정 사용)
    ) -> Dict:
        """
        Process the dubbing pipeline with target text (using TTS).
        
        Args:
            src_audio_path: Path to source audio file
            src_textgrid_path: Path to source TextGrid file
            tgt_text: List of target language sentences
            output_dir: Directory to save outputs
            on_screen_segments: List indicating whether each segment is on-screen
            use_tts: Whether to use TTS for synthesis (if False, only alignment is performed)
            use_feedback: Whether to use feedback mechanism (None to use class default)
        
        Returns:
            Dict containing results and evaluation metrics
        """
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
        
        # Map target sentences to source segments
        tgt_sentences = self._create_target_sentence_mapping(
            src_segments=src_segments,
            tgt_text=tgt_text,
            matches=matches
        )
        
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
        
        # 텍스트 검증
        self._validate_aligned_text(aligned_segments, tgt_text)
        
        if use_tts:
            # Initialize TTS engine with config parameters
            tts_config = self.config.get("tts", {})
            tts = TextToSpeech(
                lang=self.tgt_lang,
                voice_id=tts_config.get("voice_id"),
                engine=tts_config.get("engine", "neural")
            )
            
            # 피드백 루프 사용 여부 결정
            should_use_feedback = use_feedback if use_feedback is not None else self.enable_feedback
            
            # 빈 세그먼트 제외
            non_empty_segments = [seg for seg in aligned_segments if seg['text'].strip()]
            
            if should_use_feedback and non_empty_segments:
                logger.info("Using feedback loop for TTS optimization")
                
                # 피드백 루프 실행
                best_params, tts_results = self.feedback_loop.run(
                    tts=tts,
                    src_audio_path=src_audio_path,
                    src_segments=src_segments,
                    aligned_segments=non_empty_segments,
                    initial_params=tts_config,
                    output_dir=output_dir / "feedback_loop"
                )
                
                # 설정에 최적 파라미터 저장
                self.config["tts"].update(best_params)
                
                # 결과 매핑
                all_tts_results = self._map_tts_results_to_segments(
                    tts_results=tts_results,
                    aligned_segments=aligned_segments
                )
                
                # 최종 오디오 렌더링
                final_audio_path = output_dir / "dubbed_audio.wav"
                valid_indices = [i for i, seg in enumerate(aligned_segments) if seg['text'].strip()]
                
                self.renderer.render(
                    src_audio_path=src_audio_path,
                    tts_audio_paths=[all_tts_results[i]['audio_path'] for i in valid_indices if all_tts_results[i]['audio_path']],
                    segment_timings=[(aligned_segments[i]['start'], aligned_segments[i]['end']) for i in valid_indices],
                    output_path=final_audio_path
                )
                
                # 최종 평가
                evaluation = self.evaluator.evaluate(
                    src_audio_path=src_audio_path,
                    src_segments=src_segments,
                    tgt_audio_path=final_audio_path,
                    aligned_segments=non_empty_segments,
                    tts_results=[all_tts_results[i] for i in valid_indices]
                )
                
                # 프로소디 분석 결과 추가
                prosody_analyzer = ProsodyAnalyzer()
                similarity_scores = prosody_analyzer.analyze(
                    src_audio_path=src_audio_path,
                    tgt_audio_path=str(final_audio_path),
                    src_segments=src_segments,
                    tgt_segments=non_empty_segments,
                    output_dir=output_dir / "prosody_analysis"
                )
                
                # 분석 결과 평가에 추가
                evaluation["prosody_similarity"] = similarity_scores
                
            else:
                # 피드백 없이 TTS 생성
                logger.info("Generating TTS without feedback loop")
                tts_results = tts.synthesize(
                    sentences=[seg['text'] for seg in non_empty_segments],
                    durations=[seg['duration'] for seg in non_empty_segments]
                )
                
                # 결과 매핑
                all_tts_results = self._map_tts_results_to_segments(
                    tts_results=tts_results,
                    aligned_segments=aligned_segments
                )
                
                # 최종 오디오 렌더링
                final_audio_path = output_dir / "dubbed_audio.wav"
                valid_indices = [i for i, seg in enumerate(aligned_segments) if seg['text'].strip()]
                
                self.renderer.render(
                    src_audio_path=src_audio_path,
                    tts_audio_paths=[all_tts_results[i]['audio_path'] for i in valid_indices if all_tts_results[i]['audio_path']],
                    segment_timings=[(aligned_segments[i]['start'], aligned_segments[i]['end']) for i in valid_indices],
                    output_path=final_audio_path
                )
                
                # 평가
                evaluation = self.evaluator.evaluate(
                    src_audio_path=src_audio_path,
                    src_segments=src_segments,
                    tgt_audio_path=final_audio_path,
                    aligned_segments=non_empty_segments,
                    tts_results=[all_tts_results[i] for i in valid_indices]
                )
        else:
            # TTS 없이 정렬 결과만 반환
            evaluation = {
                "aligned": {
                    "isochrony": 0.0,
                    "smoothness": 0.0,
                    "fluency": 0.0,
                    "intelligibility": 0.0,
                    "overall": 0.0
                }
            }
        
        # 결과 저장
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
            "complete_text": " ".join(tgt_text)
        }
        
        # 피드백 루프 결과 추가
        if use_tts and 'should_use_feedback' in locals() and should_use_feedback:
            feedback_history = self.feedback_loop.history
            
            if feedback_history:
                results["feedback_loop"] = {
                    "iterations": len(feedback_history),
                    "initial_score": feedback_history[0]["overall_score"] if feedback_history else 0.0,
                    "final_score": feedback_history[-1]["overall_score"] if feedback_history else 0.0,
                    "best_params": best_params,
                    "improvement": feedback_history[-1]["overall_score"] - feedback_history[0]["overall_score"] 
                        if len(feedback_history) > 1 else 0.0
                }
        
        # JSON 파일로 저장
        with open(output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processing completed. Results saved to {output_dir}")
        return results
    
    def _create_target_sentence_mapping(
        self,
        src_segments: List[Dict[str, Any]],
        tgt_text: List[str],
        matches: List[Dict[str, Any]]
    ) -> List[str]:
        """
        매칭 결과를 기반으로 원본 세그먼트에 대응하는 타겟 문장 매핑을 생성합니다.
        
        Args:
            src_segments: 원본 세그먼트 목록
            tgt_text: 타겟 문장 목록
            matches: 매칭 결과 목록
            
        Returns:
            원본 세그먼트에 대응하는 타겟 문장 목록
        """
        # 빈 매핑 초기화
        tgt_sentences = [""] * len(src_segments)
        
        # 매칭된 문장 매핑
        for match in matches:
            src_idx = match['src_idx']
            tgt_idx = match['tgt_idx']
            
            if src_idx < len(tgt_sentences) and tgt_idx < len(tgt_text):
                tgt_sentences[src_idx] = tgt_text[tgt_idx]
        
        # 사용된 타겟 인덱스 확인
        used_tgt_indices = set(match['tgt_idx'] for match in matches if match['tgt_idx'] < len(tgt_text))
        
        # 미사용 타겟 문장 처리
        unused_tgt_indices = [i for i in range(len(tgt_text)) if i not in used_tgt_indices]
        
        if unused_tgt_indices:
            # 빈 슬롯 찾기
            empty_slots = [i for i, sentence in enumerate(tgt_sentences) if not sentence]
            
            # 빈 슬롯에 미사용 문장 배치
            for slot_idx, tgt_idx in zip(empty_slots, unused_tgt_indices):
                if tgt_idx < len(tgt_text):
                    tgt_sentences[slot_idx] = tgt_text[tgt_idx]
            
            # 여전히 미사용 문장이 있으면 마지막 세그먼트에 결합
            remaining_unused = [i for i in unused_tgt_indices if i >= len(empty_slots)]
            if remaining_unused:
                remaining_text = " ".join([tgt_text[i] for i in remaining_unused])
                if tgt_sentences[-1]:
                    tgt_sentences[-1] += " " + remaining_text
                else:
                    tgt_sentences[-1] = remaining_text
        
        # 타겟 텍스트가 모두 포함되었는지 확인
        used_text = set(" ".join(tgt_sentences).replace(" ", "").lower())
        expected_text = set(" ".join(tgt_text).replace(" ", "").lower())
        
        # 누락된 텍스트가 있으면 첫 번째 세그먼트에 모든 텍스트 배치
        if used_text != expected_text and src_segments:
            logger.warning("Not all target text was mapped. Placing all text in first segment.")
            tgt_sentences[0] = " ".join(tgt_text)
            for i in range(1, len(tgt_sentences)):
                tgt_sentences[i] = ""
        
        return tgt_sentences
    
    def _validate_aligned_text(self, aligned_segments: List[Dict[str, Any]], tgt_text: List[str]) -> None:
        """
        정렬된 세그먼트의 텍스트가 원본 타겟 텍스트를 모두 포함하는지 확인합니다.
        
        Args:
            aligned_segments: 정렬된 세그먼트 목록
            tgt_text: 원본 타겟 텍스트 목록
        """
        if not aligned_segments:
            return
            
        # 정렬된 텍스트와 원본 텍스트 비교
        aligned_text = " ".join([seg["text"] for seg in aligned_segments]).replace(" ", "").lower()
        original_text = " ".join(tgt_text).replace(" ", "").lower()
        
        if aligned_text != original_text:
            logger.warning("Text mismatch after alignment. Forcing full text in first segment.")
            
            # 첫 번째 세그먼트에 모든 텍스트 배치
            aligned_segments[0]["text"] = " ".join(tgt_text)
            for i in range(1, len(aligned_segments)):
                aligned_segments[i]["text"] = ""
    
    def _map_tts_results_to_segments(
        self,
        tts_results: List[Dict[str, Any]],
        aligned_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        TTS 결과를 모든 세그먼트에 매핑합니다.
        
        Args:
            tts_results: TTS 결과 목록
            aligned_segments: 정렬된 세그먼트 목록
            
        Returns:
            모든 세그먼트에 대응하는 TTS 결과 목록
        """
        all_tts_results = []
        tts_idx = 0
        
        for seg in aligned_segments:
            if seg['text'].strip():
                # 해당 세그먼트에 TTS 결과 매핑
                if tts_idx < len(tts_results):
                    all_tts_results.append(tts_results[tts_idx])
                    tts_idx += 1
                else:
                    # 예상치 못한 경우에 대한 예비 처리
                    logger.warning(f"Missing TTS result for segment '{seg['text']}'")
                    all_tts_results.append({
                        'text': seg['text'],
                        'audio_path': '',
                        'duration': seg['duration'],
                        'target_duration': seg['duration']
                    })
            else:
                # 빈 세그먼트를 위한 무음 처리
                all_tts_results.append({
                    'text': '',
                    'audio_path': '',
                    'duration': seg['duration'],
                    'target_duration': seg['duration']
                })
        
        return all_tts_results

if __name__ == "__main__":
    # 예제 실행
    pipeline = AutomaticDubbingPipeline(
        src_lang="ko",
        tgt_lang="en",
        embedding_model="LASER", # LASER, SBERT
        use_relaxation=True,
        enable_feedback=True,   # 피드백 루프 활성화
        feedback_iterations=2    # 최대 2회 반복
    )
    
    # 현재 디렉토리 가져오기
    current_directory = os.getcwd()
    
    # 피드백 루프를 활용한 자동 더빙 실행
    results = pipeline.process_with_text(
        src_audio_path=os.path.join(current_directory, 'input/윤장목소리1.wav'),
        src_textgrid_path=os.path.join(current_directory, 'input/윤장목소리1.TextGrid'),
        tgt_text=["Hello.", "My name is Jo Yoon-jang.", "Nice to meet you.", "Please take care of me."],
        output_dir=Path("output/improved_dubbing"),
        use_feedback=True  # 명시적으로 피드백 사용 설정
    )
    
    # 결과 출력
    if "feedback_loop" in results:
        print(f"=== 피드백 루프 결과 ===")
        print(f"반복 횟수: {results['feedback_loop']['iterations']}")
        print(f"초기 점수: {results['feedback_loop']['initial_score']:.4f}")
        print(f"최종 점수: {results['feedback_loop']['final_score']:.4f}")
        print(f"개선도: {results['feedback_loop']['improvement']:.4f}")
        print(f"최적 파라미터: {results['feedback_loop']['best_params']}")
    
    print(f"\n=== 평가 결과 ===")
    print(f"전체 점수: {results['evaluation']['aligned']['overall']}")
    print(f"등시성: {results['evaluation']['aligned']['isochrony']}")
    print(f"매끄러움: {results['evaluation']['aligned']['smoothness']}")
    print(f"유창성: {results['evaluation']['aligned']['fluency']}")
    print(f"명료성: {results['evaluation']['aligned']['intelligibility']}")
    
    if "prosody_similarity" in results['evaluation']:
        print(f"\n=== 프로소디 유사도 ===")
        for key, value in results['evaluation']['prosody_similarity'].items():
            print(f"{key}: {value:.4f}")
    
    print(f"\n최종 결과는 output/improved_dubbing 폴더에 저장되었습니다.")