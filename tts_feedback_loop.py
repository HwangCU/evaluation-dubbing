# tts_feedback_loop.py
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import numpy as np
import os

# 필요한 모듈 가져오기
from tts_optimizer import TTSOptimizer
from prosody_analyzer import ProsodyAnalyzer
from tts import TextToSpeech
from renderer import AudioRenderer

logger = logging.getLogger(__name__)

class TTSFeedbackLoop:
    """
    음성 유사도 기반 TTS 피드백 루프를 구현한 클래스.
    원본 음성과 합성 음성 간의 유사도를 분석하고, 
    유사도 결과를 TTS 파라미터에 반영하여 더 나은 더빙 결과를 생성합니다.
    """
    
    def __init__(
        self, 
        max_iterations: int = 3,
        improvement_threshold: float = 0.05,
        similarity_threshold: float = 0.85
    ):
        """
        TTS 피드백 루프를 초기화합니다.
        
        Args:
            max_iterations: 최대 반복 횟수
            improvement_threshold: 계속 진행하기 위한 최소 개선 임계값
            similarity_threshold: 만족할만한 유사도 임계값
        """
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.similarity_threshold = similarity_threshold
        
        # 필요한 컴포넌트 초기화
        self.optimizer = TTSOptimizer(max_iterations, improvement_threshold)
        self.analyzer = ProsodyAnalyzer()
        self.renderer = AudioRenderer()
        
        self.history = []
        
        logger.info(f"TTS Feedback Loop initialized with max_iterations={max_iterations}, "
                   f"improvement_threshold={improvement_threshold}, "
                   f"similarity_threshold={similarity_threshold}")
    
    def run(
        self,
        tts: TextToSpeech,
        src_audio_path: str,
        src_segments: List[Dict[str, Any]],
        aligned_segments: List[Dict[str, Any]],
        initial_params: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        TTS 피드백 루프를 실행합니다.
        
        Args:
            tts: TTS 엔진 인스턴스
            src_audio_path: 원본 오디오 경로
            src_segments: 원본 세그먼트 정보
            aligned_segments: 정렬된 세그먼트 정보
            initial_params: 초기 TTS 파라미터
            output_dir: 출력 디렉토리
            
        Returns:
            (최적 파라미터, TTS 합성 결과) 쌍
        """
        if output_dir is None:
            output_dir = Path("output/feedback_loop")
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)
            
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
        
        # 필터링: 빈 세그먼트 제거
        non_empty_segments = [seg for seg in aligned_segments if seg['text'].strip()]
        
        logger.info(f"Starting TTS feedback loop with {len(non_empty_segments)} segments")
        
        # 피드백 루프 시작
        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration+1}/{self.max_iterations} - Current parameters: {current_params}")
            
            # 1. 현재 파라미터로 TTS 합성
            tts_engine = self._create_tts_with_params(tts, current_params)
            
            tts_results = tts_engine.synthesize(
                sentences=[seg['text'] for seg in non_empty_segments],
                durations=[seg['duration'] for seg in non_empty_segments]
            )
            
            # 2. 더빙 오디오 렌더링
            iteration_audio_path = output_dir / f"dubbed_audio_iteration_{iteration+1}.wav"
            self.renderer.render(
                src_audio_path=src_audio_path,
                tts_audio_paths=[res['audio_path'] for res in tts_results if res['audio_path']],
                segment_timings=[(seg['start'], seg['end']) for seg in non_empty_segments],
                output_path=iteration_audio_path
            )
            
            # 3. 프로소디 분석 및 유사도 점수 계산
            similarity_scores = self.analyzer.analyze(
                src_audio_path=src_audio_path,
                tgt_audio_path=str(iteration_audio_path),
                src_segments=src_segments,
                tgt_segments=non_empty_segments,
                output_dir=output_dir / f"analysis_iteration_{iteration+1}"
            )
            
            overall_score = similarity_scores['overall']
            
            # 4. 결과 기록
            self.history.append({
                'iteration': iteration + 1,
                'params': current_params.copy(),
                'similarity_scores': similarity_scores,
                'overall_score': overall_score
            })
            
            # 5. 최고 결과 업데이트
            if overall_score > best_score:
                best_score = overall_score
                best_params = current_params.copy()
                best_tts_results = tts_results
                logger.info(f"New best score: {best_score:.4f} with parameters: {best_params}")
            
            # 6. 결과 시각화 및 저장
            self._visualize_iteration_results(
                iteration=iteration + 1,
                current_score=overall_score,
                best_score=best_score,
                output_dir=output_dir
            )
            
            # 7. 성공 조건 확인: 유사도가 임계값보다 높으면 종료
            if overall_score >= self.similarity_threshold:
                logger.info(f"Reached similarity threshold ({overall_score:.4f} >= {self.similarity_threshold})")
                break
            
            # 마지막 반복이 아니면 다음 파라미터 계산
            if iteration < self.max_iterations - 1:
                # 이전 반복의 점수와 비교하여 개선도 계산
                prev_score = 0.0 if iteration == 0 else self.history[iteration-1]['overall_score']
                improvement = overall_score - prev_score
                
                # 개선이 충분하지 않으면 종료
                if iteration > 0 and improvement < self.improvement_threshold:
                    logger.info(f"Stopping early: improvement {improvement:.4f} < threshold {self.improvement_threshold}")
                    break
                
                # 다음 파라미터 계산
                recommendations = self.analyzer.generate_recommendations(
                    scores=similarity_scores,
                    current_params=current_params
                )
                
                # 추천된 파라미터 적용
                current_params = recommendations['parameters']
                
                logger.info(f"Parameter changes for next iteration: {recommendations['changes']}")
                logger.info(f"Expected improvement: {recommendations['expected_improvement']:.4f}")
        
        # 피드백 루프 결과 저장
        self._save_feedback_results(output_dir)
        
        # 최종 결과 생성 (최적 파라미터 사용)
        if best_score < self.similarity_threshold and self.history:
            logger.info(f"Best similarity score ({best_score:.4f}) below threshold. "
                      f"Generating final output with best parameters: {best_params}")
            
            # 최적 파라미터로 최종 TTS 생성
            final_tts = self._create_tts_with_params(tts, best_params)
            best_tts_results = final_tts.synthesize(
                sentences=[seg['text'] for seg in non_empty_segments],
                durations=[seg['duration'] for seg in non_empty_segments]
            )
            
            # 최종 오디오 렌더링
            final_audio_path = output_dir / "dubbed_audio_final.wav"
            self.renderer.render(
                src_audio_path=src_audio_path,
                tts_audio_paths=[res['audio_path'] for res in best_tts_results if res['audio_path']],
                segment_timings=[(seg['start'], seg['end']) for seg in non_empty_segments],
                output_path=final_audio_path
            )
        
        # 최적 파라미터 및 TTS 결과 반환
        return best_params, best_tts_results
    
    def _create_tts_with_params(self, tts_base: TextToSpeech, params: Dict[str, Any]) -> TextToSpeech:
        """
        주어진 파라미터로 새 TTS 인스턴스를 생성합니다.
        
        Args:
            tts_base: 기본 TTS 인스턴스
            params: TTS 파라미터
            
        Returns:
            설정된 TTS 인스턴스
        """
        return TextToSpeech(
            lang=tts_base.lang,
            voice_id=tts_base.voice_id,
            engine=params.get("engine", tts_base.engine),
            speaking_rate=params.get("speaking_rate", 1.0),
            pitch=params.get("pitch", 0.0),
            volume=params.get("volume", 1.0),
            voice_style=params.get("voice_style", "neutral")
        )
    
    def _visualize_iteration_results(
        self,
        iteration: int,
        current_score: float,
        best_score: float,
        output_dir: Path
    ) -> None:
        """
        반복 결과를 시각화합니다.
        
        Args:
            iteration: 현재 반복 횟수
            current_score: 현재 점수
            best_score: 최고 점수
            output_dir: 출력 디렉토리
        """
        import matplotlib.pyplot as plt
        
        # 점수 추이 그래프
        plt.figure(figsize=(10, 6))
        
        iterations = [h['iteration'] for h in self.history]
        scores = [h['overall_score'] for h in self.history]
        
        plt.plot(iterations, scores, 'bo-', linewidth=2, markersize=8)
        plt.axhline(y=self.similarity_threshold, color='r', linestyle='--', 
                   label=f'Threshold: {self.similarity_threshold}')
        
        # 현재 및 최고 점수 표시
        plt.scatter([iteration], [current_score], color='green', s=100, 
                   label=f'Current: {current_score:.4f}')
        plt.scatter([iterations[scores.index(best_score)]], [best_score], color='red', s=100, 
                   label=f'Best: {best_score:.4f}')
        
        plt.title('TTS Feedback Loop Progress')
        plt.xlabel('Iteration')
        plt.ylabel('Similarity Score')
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 1.0)
        
        # x축을 정수로 표시
        plt.xticks(range(1, self.max_iterations + 1))
        
        plt.tight_layout()
        plt.savefig(output_dir / "feedback_progress.png")
        plt.close()
    
    def _save_feedback_results(self, output_dir: Path) -> None:
        """
        피드백 루프 결과를 저장합니다.
        
        Args:
            output_dir: 출력 디렉토리
        """
        results_path = output_dir / "feedback_loop_history.json"
        
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved feedback loop history to {results_path}")
        
        # 요약 보고서 생성
        summary = {
            "iterations_completed": len(self.history),
            "max_iterations": self.max_iterations,
            "initial_params": self.history[0]["params"] if self.history else {},
            "best_iteration": 0,
            "best_score": 0.0,
            "best_params": {},
            "improvement": 0.0
        }
        
        if self.history:
            # 최고 점수 찾기
            best_idx = max(range(len(self.history)), key=lambda i: self.history[i]['overall_score'])
            summary["best_iteration"] = self.history[best_idx]["iteration"]
            summary["best_score"] = self.history[best_idx]["overall_score"]
            summary["best_params"] = self.history[best_idx]["params"]
            
            # 개선 계산
            initial_score = self.history[0]["overall_score"]
            best_score = self.history[best_idx]["overall_score"]
            summary["improvement"] = best_score - initial_score
            summary["relative_improvement"] = f"{(best_score / max(0.0001, initial_score) - 1) * 100:.2f}%"
        
        # 요약 저장
        summary_path = output_dir / "feedback_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved feedback summary to {summary_path}")
        
        # 텍스트 보고서 생성
        report_path = output_dir / "feedback_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== TTS 피드백 루프 보고서 ===\n\n")
            f.write(f"완료된 반복 횟수: {summary['iterations_completed']}/{summary['max_iterations']}\n")
            f.write(f"최고 점수: {summary['best_score']:.4f} (반복 {summary['best_iteration']})\n")
            f.write(f"개선도: {summary['improvement']:.4f} ({summary['relative_improvement']})\n\n")
            
            f.write("초기 파라미터:\n")
            for param, value in summary["initial_params"].items():
                f.write(f"  - {param}: {value}\n")
            
            f.write("\n최적 파라미터:\n")
            for param, value in summary["best_params"].items():
                f.write(f"  - {param}: {value}\n")
            
            f.write("\n반복별 점수:\n")
            for entry in self.history:
                f.write(f"  - 반복 {entry['iteration']}: {entry['overall_score']:.4f}\n")
        
        logger.info(f"Saved feedback report to {report_path}")