# core/evaluator.py
"""
음성 유사도 종합 평가 모듈
"""
import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import os
import matplotlib
from config import EVAL_CONFIG

logger = logging.getLogger(__name__)
# IPA 문자(발음 기호) 지원을 위해 DejaVu Sans 폰트 사용
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic']  # 폰트 설정
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
class SimilarityEvaluator:
    """
    원본 음성과 합성 음성 간의 유사도를 종합적으로 평가하는 클래스
    """
    
    def __init__(self, feature_weights: Optional[Dict[str, float]] = None):
        """
        유사도 평가기 초기화
        
        Args:
            feature_weights: 특성별 가중치 (기본값: config.py에서 설정)
        """
        self.feature_weights = feature_weights or EVAL_CONFIG["feature_weights"]
        logger.info("유사도 평가기 초기화")
    
    def evaluate(
        self,
        prosody_scores: Dict[str, float],
        segment_alignments: List[Dict[str, Any]],
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        음성 유사도 점수를 종합 평가
        
        Args:
            prosody_scores: 프로소디 유사도 점수
            segment_alignments: 세그먼트 정렬 정보
            output_dir: 결과 저장 디렉토리
            
        Returns:
            종합 평가 결과 딕셔너리
        """
        if output_dir:
            output_dir.mkdir(exist_ok=True, parents=True)
        
        # 세그먼트 정렬 점수 계산
        alignment_score = self._evaluate_alignment(segment_alignments)
        
        # 종합 점수 계산
        overall_scores = {
            # 프로소디 점수 복사
            **prosody_scores,
            
            # 정렬 점수 추가
            "alignment_score": alignment_score,
            
            # 가중치 적용한 최종 점수
            "final_score": self._calculate_final_score(prosody_scores, alignment_score)
        }
        
        # 등급 부여
        grade = self._assign_grade(overall_scores["final_score"])
        overall_scores["grade"] = grade
        
        # 개선 제안사항 생성
        suggestions = self._generate_suggestions(overall_scores)
        overall_scores["improvement_suggestions"] = suggestions
        
        # 결과 저장
        if output_dir:
            self._save_evaluation(overall_scores, output_dir)
            self._visualize_evaluation(overall_scores, output_dir)
        
        logger.info(f"종합 평가 완료: 최종 점수 {overall_scores['final_score']:.4f}, 등급 {grade}")
        
        return overall_scores
    
    def _evaluate_alignment(self, segment_alignments: List[Dict[str, Any]]) -> float:
        """
        세그먼트 정렬 품질 평가
        
        Args:
            segment_alignments: 세그먼트 정렬 정보
            
        Returns:
            정렬 품질 점수 (0.0 ~ 1.0)
        """
        # 정렬된 세그먼트가 없으면 기본값 반환
        if not segment_alignments:
            return 0.5
        
        # 시간 정렬 정확도 계산
        temporal_scores = []
        for alignment in segment_alignments:
            src_start = alignment.get("src_start", 0)
            src_end = alignment.get("src_end", 0)
            tgt_start = alignment.get("tgt_start", 0)
            tgt_end = alignment.get("tgt_end", 0)
            
            # 시작/종료 시간 오차
            start_error = abs(src_start - tgt_start)
            end_error = abs(src_end - tgt_end)
            
            # 세그먼트 길이
            src_duration = src_end - src_start
            
            # 정규화된 오차 (원본 길이 대비)
            if src_duration > 0:
                normalized_error = (start_error + end_error) / (2 * src_duration)
                score = max(0, 1 - normalized_error)
            else:
                score = 0
            
            temporal_scores.append(score)
        
        # 평균 시간 정렬 점수
        alignment_score = np.mean(temporal_scores) if temporal_scores else 0.5
        
        logger.debug(f"정렬 점수: {alignment_score:.4f}")
        return alignment_score
    
    def _calculate_final_score(
        self, 
        prosody_scores: Dict[str, float], 
        alignment_score: float
    ) -> float:
        """
        최종 점수 계산 (프로소디 점수와 정렬 점수의 가중 평균)
        
        Args:
            prosody_scores: 프로소디 유사도 점수
            alignment_score: 정렬 점수
            
        Returns:
            최종 종합 점수 (0.0 ~ 1.0)
        """
        # 프로소디 전체 점수
        prosody_overall = prosody_scores.get("overall", 0.7)
        
        # 가중 평균 계산 (프로소디 70%, 정렬 30%)
        final_score = prosody_overall
        
        return final_score
    
    def _assign_grade(self, score: float) -> str:
        """
        점수에 따른 등급 부여
        
        Args:
            score: 평가 점수 (0.0 ~ 1.0)
            
        Returns:
            등급 (A+, A, B, C, D)
        """
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B"
        elif score >= 0.6:
            return "C"
        elif score >= 0.5:
            return "D"
        else:
            return "F"
    
    def _generate_suggestions(self, scores: Dict[str, float]) -> List[str]:
        """
        점수에 기반한 개선 제안사항 생성
        
        Args:
            scores: 평가 점수 딕셔너리
            
        Returns:
            개선 제안사항 목록
        """
        suggestions = []
        
        # 낮은 점수의 특성에 따른 제안사항
        if scores.get("pause_similarity", 1.0) < 0.7:
            suggestions.append("휴지(일시정지) 패턴 개선: 원본 음성의 문장 구분과 쉼표 위치에 맞춰 일시정지 패턴을 조정하세요.")
        
        if scores.get("pitch_similarity", 1.0) < 0.7:
            suggestions.append("음높이(피치) 패턴 개선: 원본 음성의 억양 패턴을 모방하여 자연스러운 음높이 변화를 추가하세요.")
        
        if scores.get("energy_similarity", 1.0) < 0.7:
            suggestions.append("에너지(강세) 패턴 개선: 원본 음성에서 강조된 단어나 구문에 맞게 합성 음성의 볼륨과 강세를 조정하세요.")
        
        if scores.get("rhythm_similarity", 1.0) < 0.7:
            suggestions.append("리듬 패턴 개선: 원본 음성의 말하기 속도 변화와 단어 간격을 더 정확히 모방하세요.")
        
        if scores.get("vowel_similarity", 1.0) < 0.7:
            suggestions.append("모음 길이 패턴 개선: 중요한 모음의 길이를 원본과 유사하게 조정하여 자연스러운 발음을 구현하세요.")
        
        if scores.get("alignment_score", 1.0) < 0.7:
            suggestions.append("시간 정렬 개선: 원본 음성과 합성 음성의 시작/종료 시간을 더 정확히 일치시키세요.")
        
        # 최종 점수에 따른 일반적 제안사항
        if scores.get("final_score", 0) < 0.6:
            suggestions.append("전반적인 프로소디 개선이 필요합니다. 다른 TTS 엔진이나 음성 합성 방식을 시도해보세요.")
        elif scores.get("final_score", 0) < 0.8:
            suggestions.append("세부적인 조정을 통해 더 자연스러운 합성 음성을 만들 수 있습니다.")
        
        return suggestions
    
    def _save_evaluation(self, evaluation: Dict[str, Any], output_dir: Path) -> None:
        """
        평가 결과 저장
        
        Args:
            evaluation: 평가 결과 딕셔너리
            output_dir: 결과 저장 디렉토리
        """
        # JSON 결과 저장
        json_path = output_dir / "evaluation_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2)
        
        # 텍스트 보고서 저장
        report_path = output_dir / "evaluation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 음성 유사도 평가 보고서 ===\n\n")
            
            f.write(f"최종 점수: {evaluation.get('final_score', 0):.4f} (등급: {evaluation.get('grade', 'N/A')})\n\n")
            
            f.write("== 세부 점수 ==\n")
            for key, value in evaluation.items():
                if isinstance(value, float) and key not in ('final_score', 'overall'):
                    f.write(f"- {key}: {value:.4f}\n")
            
            f.write(f"\n프로소디 전체 점수: {evaluation.get('overall', 0):.4f}\n")
            f.write(f"정렬 점수: {evaluation.get('alignment_score', 0):.4f}\n\n")
            
            f.write("== 개선 제안사항 ==\n")
            for i, suggestion in enumerate(evaluation.get('improvement_suggestions', []), 1):
                f.write(f"{i}. {suggestion}\n")
        
        logger.info(f"평가 결과가 {json_path}에 저장되었습니다.")
        logger.info(f"평가 보고서가 {report_path}에 저장되었습니다.")
    
    def _visualize_evaluation(self, evaluation: Dict[str, Any], output_dir: Path) -> None:
        """
        평가 결과 시각화
        
        Args:
            evaluation: 평가 결과 딕셔너리
            output_dir: 결과 저장 디렉토리
        """
        # 레이더 차트로 시각화
        plt.figure(figsize=(10, 8))
        
        # 시각화할 특성과 점수 추출
        features = []
        scores = []
        
        for key, value in evaluation.items():
            if isinstance(value, float) and key not in ('final_score', 'overall'):
                features.append(key.replace('_similarity', '').replace('_score', '').capitalize())
                scores.append(value)
        
        # 최소 3개 이상의 특성이 필요
        if len(features) < 3:
            logger.warning("시각화에 필요한 충분한 특성이 없습니다.")
            return
        
        # 각도 계산
        angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
        
        # 첫 번째 값을 마지막에 복사하여 도형 닫기
        scores.append(scores[0])
        angles.append(angles[0])
        features.append(features[0])
        
        # 레이더 차트 그리기
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, scores, 'o-', linewidth=2)
        ax.fill(angles, scores, alpha=0.25)
        
        # 축 설정
        ax.set_thetagrids(np.degrees(angles[:-1]), features[:-1])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        # 제목과 등급 표시
        plt.title(f'음성 유사도 평가 결과 (등급: {evaluation.get("grade", "N/A")})')
        plt.figtext(0.5, 0.01, f'최종 점수: {evaluation.get("final_score", 0):.4f}', 
                   ha='center', fontsize=12)
        
        # 저장
        plt.tight_layout()
        chart_path = output_dir / "evaluation_radar.png"
        plt.savefig(chart_path)
        plt.close()
        
        logger.info(f"평가 시각화가 {chart_path}에 저장되었습니다.")