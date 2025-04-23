"""
자동 더빙 평가 지표 계산을 위한 모듈

이 모듈은 자동 더빙 시스템의 품질을 평가하기 위한 다양한 지표를 계산합니다.
논문에서 제시된 Isochrony, Smoothness, Fluency, Intelligibility 등의 지표를 구현합니다.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

from ..models.sentence import Sentence, SentenceMapping
from ..config import SMOOTHNESS_THRESHOLD, FLUENCY_THRESHOLD

logger = logging.getLogger(__name__)


class DubbingEvaluator:
    """자동 더빙 품질 평가를 위한 클래스"""
    
    @staticmethod
    def calculate_isochrony(mappings: List[SentenceMapping]) -> float:
        """
        이소크로니(Isochrony) 점수 계산
        
        이소크로니는 원본과 더빙된 비디오 간의 시간적 일치도를 측정합니다.
        
        Args:
            mappings: SentenceMapping 객체 리스트
            
        Returns:
            이소크로니 점수 (0.0 ~ 1.0)
        """
        if not mappings:
            return 0.0
        
        # 각 매핑의 이소크로니 점수 계산
        valid_mappings = [m for m in mappings 
                          if m.source.has_timing_info() and m.target.has_timing_info()]
        
        if not valid_mappings:
            logger.warning("시간 정보가 있는 매핑이 없어 이소크로니 점수를 계산할 수 없습니다.")
            return 0.0
        
        # 각 매핑의 이소크로니 점수 계산
        iso_scores = [m.get_isochrony_score() for m in valid_mappings]
        
        # 평균 이소크로니 점수
        avg_iso_score = sum(iso_scores) / len(iso_scores)
        
        return avg_iso_score
    
    @staticmethod
    def calculate_smoothness(mappings: List[SentenceMapping]) -> float:
        """
        스무스니스(Smoothness) 점수 계산
        
        스무스니스는 연속된 세그먼트 간의 발화 속도 변화가 얼마나 부드러운지 측정합니다.
        
        Args:
            mappings: SentenceMapping 객체 리스트
            
        Returns:
            스무스니스 점수 (0.0 ~ 1.0)
        """
        if len(mappings) < 2:
            logger.warning("스무스니스를 계산하려면 최소 2개의 매핑이 필요합니다.")
            return 1.0  # 1개 문장은 완벽하게 스무스하다고 가정
        
        # 시간 정보가 있는 매핑만 필터링
        valid_mappings = [m for m in mappings 
                          if m.source.has_timing_info() and m.target.has_timing_info()]
        
        if len(valid_mappings) < 2:
            logger.warning("시간 정보가 있는 매핑이 충분하지 않아 스무스니스 점수를 계산할 수 없습니다.")
            return 0.0
        
        # 각 매핑의 발화 속도 계산
        speaking_rates = []
        for mapping in valid_mappings:
            source_rate, target_rate = mapping.compute_speaking_rate()
            
            # 이완이 적용된 유효 타겟 지속 시간 고려
            effective_duration = mapping.get_effective_target_duration()
            if effective_duration > 0:
                effective_rate = len(mapping.target.words) / effective_duration
                speaking_rates.append(effective_rate)
            elif target_rate > 0:
                speaking_rates.append(target_rate)
        
        if not speaking_rates:
            return 0.0
        
        # 연속된 발화 속도 간의 변화율 계산
        variations = []
        for i in range(1, len(speaking_rates)):
            prev_rate = speaking_rates[i-1]
            curr_rate = speaking_rates[i]
            
            if prev_rate > 0:
                variation = abs(curr_rate - prev_rate) / prev_rate
                variations.append(variation)
        
        if not variations:
            return 1.0
        
        # 평균 변화율을 스무스니스 점수로 변환
        avg_variation = sum(variations) / len(variations)
        
        # 임계값을 초과하는 변화는 더 심각한 페널티 적용
        smoothness_score = 0.0
        if avg_variation <= SMOOTHNESS_THRESHOLD:
            # 변화가 임계값 이하인 경우 선형 점수
            smoothness_score = 1.0 - (avg_variation / SMOOTHNESS_THRESHOLD)
        else:
            # 변화가 임계값을 초과하는 경우 비선형 점수
            excess = avg_variation - SMOOTHNESS_THRESHOLD
            penalty = min(1.0, excess / (1.0 - SMOOTHNESS_THRESHOLD))
            smoothness_score = (1.0 - penalty) * (1.0 - SMOOTHNESS_THRESHOLD)
        
        return max(0.0, min(1.0, smoothness_score))
    
    @staticmethod
    def calculate_fluency(mappings: List[SentenceMapping]) -> float:
        """
        플루언시(Fluency) 점수 계산
        
        플루언시는 번역된 텍스트가 얼마나 자연스럽게 발화될 수 있는지 측정합니다.
        
        Args:
            mappings: SentenceMapping 객체 리스트
            
        Returns:
            플루언시 점수 (0.0 ~ 1.0)
        """
        if not mappings:
            return 0.0
        
        # 시간 정보가 있는 매핑만 필터링
        valid_mappings = [m for m in mappings 
                          if m.source.has_timing_info() and m.target.has_timing_info()]
        
        if not valid_mappings:
            logger.warning("시간 정보가 있는 매핑이 없어 플루언시 점수를 계산할 수 없습니다.")
            return 0.0
        
        # 각 매핑의 발화 속도 비율 계산
        rate_ratios = []
        for mapping in valid_mappings:
            source_rate, target_rate = mapping.compute_speaking_rate()
            
            if source_rate > 0 and target_rate > 0:
                # 이완 적용 후 유효 타겟 지속 시간 고려
                effective_duration = mapping.get_effective_target_duration()
                if effective_duration > 0:
                    effective_rate = len(mapping.target.words) / effective_duration
                    ratio = effective_rate / source_rate
                else:
                    ratio = target_rate / source_rate
                
                rate_ratios.append(ratio)
        
        if not rate_ratios:
            return 0.0
        
        # 플루언시 점수 계산
        # 발화 속도 비율이 1에 가까울수록 더 자연스러움
        fluency_scores = []
        for ratio in rate_ratios:
            if ratio <= 1.0:
                # 발화 속도가 원본보다 느리거나 같은 경우
                score = ratio
            else:
                # 발화 속도가 원본보다 빠른 경우 (더 빠르면 더 낮은 점수)
                excess = ratio - 1.0
                penalty = min(1.0, excess / FLUENCY_THRESHOLD)
                score = 1.0 - penalty
            
            fluency_scores.append(score)
        
        # 평균 플루언시 점수
        avg_fluency = sum(fluency_scores) / len(fluency_scores)
        
        return max(0.0, min(1.0, avg_fluency))
    
    @staticmethod
    def calculate_intelligibility(
        mappings: List[SentenceMapping],
        asr_word_error_rates: Optional[Dict[int, Tuple[float, float]]] = None
    ) -> float:
        """
        인텔리지빌리티(Intelligibility) 점수 계산
        
        인텔리지빌리티는 더빙된 음성이 얼마나 명확하게 인식될 수 있는지 측정합니다.
        
        Args:
            mappings: SentenceMapping 객체 리스트
            asr_word_error_rates: {매핑 인덱스: (원본 WER, 더빙 WER)} 형태의 ASR 단어 오류율 사전
            
        Returns:
            인텔리지빌리티 점수 (0.0 ~ 1.0)
        """
        if not asr_word_error_rates:
            logger.warning("ASR 단어 오류율이 제공되지 않아 인텔리지빌리티 점수를 계산할 수 없습니다.")
            
            # 대체 지표로 발화 속도 기반 추정치 사용
            return DubbingEvaluator._estimate_intelligibility_from_speaking_rate(mappings)
        
        # 유효한 매핑에 대한 인텔리지빌리티 계산
        intelligibility_scores = []
        
        for idx, (original_wer, dubbed_wer) in asr_word_error_rates.items():
            if idx < len(mappings):
                # 공식: I(F) = (1 - WER(dubbed)) / (1 - WER(original))
                if original_wer < 1.0:  # 완전히 인식 불가능한 경우 제외
                    original_recognition = 1.0 - original_wer
                    dubbed_recognition = 1.0 - dubbed_wer
                    
                    # 원본 대비 더빙된 음성의 인식률 비율
                    intel_score = dubbed_recognition / original_recognition
                    intelligibility_scores.append(min(1.0, intel_score))  # 최대 1.0으로 제한
        
        if not intelligibility_scores:
            logger.warning("유효한 인텔리지빌리티 점수가 없습니다.")
            return DubbingEvaluator._estimate_intelligibility_from_speaking_rate(mappings)
        
        # 평균 인텔리지빌리티 점수
        avg_intel = sum(intelligibility_scores) / len(intelligibility_scores)
        
        return max(0.0, min(1.0, avg_intel))
    
    @staticmethod
    def _estimate_intelligibility_from_speaking_rate(mappings: List[SentenceMapping]) -> float:
        """
        발화 속도를 기반으로 인텔리지빌리티 추정
        
        ASR 결과가 없을 때 발화 속도를 기반으로 인텔리지빌리티 점수 추정
        
        Args:
            mappings: SentenceMapping 객체 리스트
            
        Returns:
            추정된 인텔리지빌리티 점수 (0.0 ~ 1.0)
        """
        if not mappings:
            return 0.0
        
        # 시간 정보가 있는 매핑만 필터링
        valid_mappings = [m for m in mappings 
                          if m.source.has_timing_info() and m.target.has_timing_info()]
        
        if not valid_mappings:
            logger.warning("시간 정보가 있는 매핑이 없어 인텔리지빌리티를 추정할 수 없습니다.")
            return 0.0
        
        # 각 매핑의 발화 속도 비율 계산
        intel_scores = []
        for mapping in valid_mappings:
            # 발화 속도 계산
            source_rate, target_rate = mapping.compute_speaking_rate()
            
            # 이완 적용 후 유효 타겟 지속 시간 고려
            effective_duration = mapping.get_effective_target_duration()
            if effective_duration > 0:
                effective_rate = len(mapping.target.words) / effective_duration
            else:
                effective_rate = target_rate
            
            # 발화 속도가 너무 빠르거나 느리면 이해도가 떨어짐
            # 최적 발화 속도 비율을 1.0으로 가정
            if effective_rate <= 0:
                intel_score = 0.0
            elif source_rate <= 0:
                intel_score = 0.8  # 기본값
            else:
                ratio = effective_rate / source_rate
                
                if ratio <= 1.0:
                    # 원본보다 느리거나 같은 경우 (점진적으로 감소)
                    intel_score = 0.5 + 0.5 * ratio
                elif ratio <= 1.5:
                    # 약간 빠른 경우 (1.0 ~ 1.5배)
                    intel_score = 1.0 - 0.4 * (ratio - 1.0) / 0.5
                else:
                    # 너무 빠른 경우 (1.5배 이상)
                    intel_score = 0.6 - 0.6 * min(1.0, (ratio - 1.5) / 0.5)
            
            intel_scores.append(max(0.0, min(1.0, intel_score)))
        
        # 평균 추정 인텔리지빌리티
        avg_intel = sum(intel_scores) / len(intel_scores)
        
        return avg_intel
    
    @staticmethod
    def evaluate_all_metrics(
        mappings: List[SentenceMapping],
        asr_word_error_rates: Optional[Dict[int, Tuple[float, float]]] = None
    ) -> Dict[str, float]:
        """
        모든 평가 지표 계산
        
        Args:
            mappings: SentenceMapping 객체 리스트
            asr_word_error_rates: ASR 단어 오류율 (인텔리지빌리티 계산용)
            
        Returns:
            평가 지표 사전 {'isochrony': 점수, 'smoothness': 점수, ...}
        """
        # 각 지표 계산
        isochrony = DubbingEvaluator.calculate_isochrony(mappings)
        smoothness = DubbingEvaluator.calculate_smoothness(mappings)
        fluency = DubbingEvaluator.calculate_fluency(mappings)
        intelligibility = DubbingEvaluator.calculate_intelligibility(
            mappings, asr_word_error_rates)
        
        # 종합 점수 (가중 평균)
        overall_score = (
            0.4 * isochrony +
            0.2 * smoothness +
            0.2 * fluency +
            0.2 * intelligibility
        )
        
        return {
            'isochrony': isochrony,
            'smoothness': smoothness,
            'fluency': fluency,
            'intelligibility': intelligibility,
            'overall': overall_score
        }