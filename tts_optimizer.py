# tts_optimizer.py
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class TTSOptimizer:
    """
    TTS 최적화 클래스: 평가 메트릭에 기반하여 TTS 매개변수를 조정합니다.
    이 클래스는 피드백 루프를 구현하여 합성 음성의 품질을 점진적으로 향상시킵니다.
    """
    
    def __init__(self, max_iterations: int = 3, improvement_threshold: float = 0.05):
        """
        TTS 최적화기를 초기화합니다.
        
        Args:
            max_iterations: 최적화 반복 횟수 최대값
            improvement_threshold: 최적화를 계속하기 위한 최소 개선 점수
        """
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.history = []
        logger.info(f"TTS Optimizer initialized with max_iterations={max_iterations}, "
                   f"improvement_threshold={improvement_threshold}")
    
    def optimize(
        self, 
        tts: Any,  # TTS 엔진
        aligner: Any,  # Prosodic Aligner
        evaluator: Any,  # Dubbing Evaluator
        src_audio_path: str,
        src_segments: List[Dict[str, Any]],
        aligned_segments: List[Dict[str, Any]],
        on_screen_segments: List[bool],
        initial_params: Dict[str, Any] = None,
        output_dir: Optional[Path] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        TTS 파라미터를 최적화하여 더 나은 품질의 합성 음성을 생성합니다.
        
        Args:
            tts: TTS 엔진 인스턴스
            aligner: Prosodic Aligner 인스턴스
            evaluator: Dubbing Evaluator 인스턴스
            src_audio_path: 원본 오디오 경로
            src_segments: 원본 세그먼트 정보
            aligned_segments: 정렬된 세그먼트 정보
            on_screen_segments: 각 세그먼트가 화면에 보이는지 여부
            initial_params: 초기 TTS 파라미터
            output_dir: 출력 디렉토리
            
        Returns:
            최적화된 TTS 결과 및 최적 파라미터
        """
        if output_dir is None:
            output_dir = Path("output")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 초기 파라미터 설정
        current_params = initial_params or {
            "speaking_rate": 1.0,
            "pitch": 0.0,
            "volume": 1.0,
            "voice_style": "neutral"
        }
        
        best_params = current_params.copy()
        best_score = 0.0
        best_tts_results = None
        
        # 최적화 반복
        for iteration in range(self.max_iterations):
            logger.info(f"Starting optimization iteration {iteration+1}/{self.max_iterations}")
            logger.info(f"Current parameters: {current_params}")
            
            # 현재 파라미터로 TTS 합성
            tts_engine = self._create_tts_with_params(tts, current_params)
            
            # TTS 합성 수행
            tts_results = tts_engine.synthesize(
                sentences=[seg['text'] for seg in aligned_segments if seg['text'].strip()],
                durations=[seg['duration'] for seg in aligned_segments if seg['text'].strip()]
            )
            
            # 합성 결과 평가
            eval_result = self._evaluate_synthesis(
                evaluator=evaluator,
                src_audio_path=src_audio_path,
                src_segments=src_segments,
                tts_results=tts_results,
                aligned_segments=[seg for seg in aligned_segments if seg['text'].strip()],
                output_dir=output_dir,
                iteration=iteration
            )
            
            # 전체 점수 계산
            overall_score = eval_result['aligned']['overall']
            
            # 히스토리에 결과 저장
            self.history.append({
                'iteration': iteration,
                'params': current_params.copy(),
                'evaluation': eval_result,
                'overall_score': overall_score
            })
            
            # 최고 결과 업데이트
            if overall_score > best_score:
                best_score = overall_score
                best_params = current_params.copy()
                best_tts_results = tts_results
                logger.info(f"New best score: {best_score:.4f} with parameters: {best_params}")
            
            # 다음 파라미터 결정
            if iteration < self.max_iterations - 1:
                prev_score = 0.0 if iteration == 0 else self.history[iteration-1]['overall_score']
                improvement = overall_score - prev_score
                
                # 개선이 threshold보다 작으면 최적화 중단
                if iteration > 0 and improvement < self.improvement_threshold:
                    logger.info(f"Stopping early: improvement {improvement:.4f} < threshold {self.improvement_threshold}")
                    break
                
                # 다음 파라미터 계산
                current_params = self._compute_next_parameters(
                    current_params=current_params,
                    eval_result=eval_result,
                    improvement=improvement
                )
        
        # 최적화 결과 저장
        self._save_optimization_results(output_dir)
        
        logger.info(f"Optimization completed. Best score: {best_score:.4f} with parameters: {best_params}")
        return best_tts_results, best_params
    
    def _create_tts_with_params(self, tts_base, params: Dict[str, Any]) -> Any:
        """
        주어진 파라미터로 새 TTS 인스턴스를 생성합니다.
        
        Args:
            tts_base: 기본 TTS 클래스 또는 인스턴스
            params: TTS 파라미터
            
        Returns:
            설정된 TTS 인스턴스
        """
        # TTS 인스턴스 복제 및 파라미터 적용
        from tts import TextToSpeech
        
        return TextToSpeech(
            lang=tts_base.lang,
            voice_id=tts_base.voice_id,
            engine=tts_base.engine,
            speaking_rate=params.get("speaking_rate", 1.0),
            pitch=params.get("pitch", 0.0),
            volume=params.get("volume", 1.0),
            voice_style=params.get("voice_style", "neutral")
        )
    
    def _evaluate_synthesis(
        self,
        evaluator: Any,
        src_audio_path: str,
        src_segments: List[Dict[str, Any]],
        tts_results: List[Dict[str, Any]],
        aligned_segments: List[Dict[str, Any]],
        output_dir: Path,
        iteration: int
    ) -> Dict[str, Any]:
        """
        합성 결과를 평가합니다.
        
        Args:
            evaluator: 평가기 인스턴스
            src_audio_path: 원본 오디오 경로
            src_segments: 원본 세그먼트 정보
            tts_results: TTS 합성 결과
            aligned_segments: 정렬된 세그먼트 정보
            output_dir: 출력 디렉토리
            iteration: 현재 반복 횟수
            
        Returns:
            평가 결과
        """
        # 오디오 렌더링
        from renderer import AudioRenderer
        renderer = AudioRenderer()
        
        final_audio_path = output_dir / f"dubbed_audio_iteration_{iteration}.wav"
        renderer.render(
            src_audio_path=src_audio_path,
            tts_audio_paths=[res['audio_path'] for res in tts_results if res['audio_path']],
            segment_timings=[(seg['start'], seg['end']) for seg in aligned_segments],
            output_path=final_audio_path
        )
        
        # 평가 수행
        eval_result = evaluator.evaluate(
            src_audio_path=src_audio_path,
            src_segments=src_segments,
            tgt_audio_path=final_audio_path,
            aligned_segments=aligned_segments,
            tts_results=tts_results
        )
        
        return eval_result
    
    def _compute_next_parameters(
        self,
        current_params: Dict[str, Any],
        eval_result: Dict[str, Dict[str, float]],
        improvement: float
    ) -> Dict[str, Any]:
        """
        평가 결과에 기반하여 다음 TTS 파라미터를 계산합니다.
        
        Args:
            current_params: 현재 파라미터
            eval_result: 평가 결과
            improvement: 이전 반복과 비교한 개선도
            
        Returns:
            다음 반복에 사용할 새 파라미터
        """
        # 현재 파라미터 복사
        next_params = current_params.copy()
        
        # 메트릭 추출
        metrics = eval_result['aligned']
        isochrony = metrics['isochrony']
        smoothness = metrics['smoothness']
        fluency = metrics['fluency']
        intelligibility = metrics['intelligibility']
        
        # 개선이 있으면 같은 방향으로 계속 조정
        direction = 1.0 if improvement > 0 else -1.0
        
        # speaking_rate 조정: 낮은 isochrony는 속도 조정 필요성을 의미
        if isochrony < 0.7:
            # 현재 speaking_rate 조정 (0.8~1.2 범위 내에서)
            rate_change = 0.05 * direction if fluency < 0.7 else -0.05 * direction
            next_params["speaking_rate"] = max(0.8, min(1.2, current_params["speaking_rate"] + rate_change))
        
        # pitch 조정: 낮은 intelligibility는 피치 조정 필요성을 의미할 수 있음
        if intelligibility < 0.7:
            # 현재 pitch 조정 (-3.0~3.0 범위 내에서)
            pitch_change = 0.5 * direction
            next_params["pitch"] = max(-3.0, min(3.0, current_params["pitch"] + pitch_change))
        
        # 음성 스타일 변경 시도
        if fluency < 0.6 and current_params["voice_style"] == "neutral":
            next_params["voice_style"] = "conversational"
        elif fluency < 0.6 and current_params["voice_style"] == "conversational":
            next_params["voice_style"] = "formal"
        
        logger.info(f"Computed next parameters: {next_params}")
        return next_params
    
    def _save_optimization_results(self, output_dir: Path) -> None:
        """
        최적화 결과를 파일로 저장합니다.
        
        Args:
            output_dir: 출력 디렉토리
        """
        results_path = output_dir / "tts_optimization_history.json"
        
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved optimization history to {results_path}")