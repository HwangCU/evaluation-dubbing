"""
프로소딕 얼라인먼트를 위한 모듈

이 모듈은 매핑된 문장 쌍의 시간 경계를 조정하여 자연스러운 더빙을 위한
프로소딕 얼라인먼트(운율적 정렬)를 수행합니다.
시간 경계 이완(relaxation)과 발화 속도 제어 기능을 제공합니다.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Union

from ..models.sentence import Sentence, SentenceMapping
from ..config import ALIGNMENT_MAX_RELAXATION, SPEAKING_RATE_MAX

logger = logging.getLogger(__name__)


class ProsodicAligner:
    """프로소딕 얼라인먼트를 처리하는 클래스"""
    
    def __init__(
        self, 
        max_relaxation: float = ALIGNMENT_MAX_RELAXATION,
        max_speaking_rate: float = SPEAKING_RATE_MAX
    ):
        """
        프로소딕 얼라이너 초기화
        
        Args:
            max_relaxation: 시간 경계 최대 이완 비율 (원본 시간의 %)
            max_speaking_rate: 최대 허용 발화 속도 비율
        """
        self.max_relaxation = max_relaxation
        self.max_speaking_rate = max_speaking_rate
    
    def align_mappings(self, mappings: List[SentenceMapping]) -> List[SentenceMapping]:
        """
        매핑된 문장 쌍의 프로소딕 얼라인먼트 수행
        
        Args:
            mappings: SentenceMapping 객체 리스트
            
        Returns:
            프로소딕 얼라인먼트가 적용된 SentenceMapping 객체 리스트
        """
        # 시간 정보가 없는 매핑은 처리할 수 없음
        valid_mappings = [m for m in mappings 
                         if m.source.has_timing_info() and m.target.has_timing_info()]
        
        if len(valid_mappings) != len(mappings):
            logger.warning(f"{len(mappings) - len(valid_mappings)}개 매핑에 시간 정보가 없어 얼라인먼트를 건너뜁니다.")
        
        if not valid_mappings:
            return mappings
        
        # 각 매핑에 대해 얼라인먼트 수행
        result = []
        for i, mapping in enumerate(valid_mappings):
            # 이전/다음 매핑 (있는 경우)
            prev_mapping = valid_mappings[i-1] if i > 0 else None
            next_mapping = valid_mappings[i+1] if i < len(valid_mappings) - 1 else None
            
            # 얼라인먼트 계산
            aligned_mapping = self._align_single_mapping(mapping, prev_mapping, next_mapping)
            result.append(aligned_mapping)
            
            logger.info(f"얼라인먼트 결과: {aligned_mapping}")
        
        # 원래 리스트에서 시간 정보가 없는 매핑을 유지
        for mapping in mappings:
            if mapping not in valid_mappings:
                result.append(mapping)
        
        return result
    
    def _align_single_mapping(
        self, 
        mapping: SentenceMapping,
        prev_mapping: Optional[SentenceMapping] = None,
        next_mapping: Optional[SentenceMapping] = None
    ) -> SentenceMapping:
        """
        단일 매핑의 프로소딕 얼라인먼트 수행
        
        Args:
            mapping: 얼라인먼트를 적용할 SentenceMapping 객체
            prev_mapping: 이전 매핑 (있는 경우)
            next_mapping: 다음 매핑 (있는 경우)
            
        Returns:
            얼라인먼트가 적용된 SentenceMapping 객체
        """
        source = mapping.source
        target = mapping.target
        
        # 발화 속도 계산
        source_rate, target_rate = mapping.compute_speaking_rate()
        
        # 발화 속도 비율 (target/source)
        rate_ratio = target_rate / source_rate if source_rate > 0 else 1.0
        
        # 시간 이완 계산
        start_relaxation, end_relaxation = self._calculate_relaxations(
            mapping, prev_mapping, next_mapping, rate_ratio)
        
        # 결과 매핑 생성
        result = SentenceMapping(
            source=source,
            target=target,
            similarity=mapping.similarity,
            time_dilation=1.0 / max(rate_ratio, 0.1),  # 발화 속도 조절을 위한 시간 팽창 계수
            start_relaxation=start_relaxation,
            end_relaxation=end_relaxation
        )
        
        return result
    
    def _calculate_relaxations(
        self, 
        mapping: SentenceMapping,
        prev_mapping: Optional[SentenceMapping],
        next_mapping: Optional[SentenceMapping],
        rate_ratio: float
    ) -> Tuple[float, float]:
        """
        최적의 시간 이완(relaxation) 값 계산
        
        Args:
            mapping: 현재 매핑
            prev_mapping: 이전 매핑 (또는 None)
            next_mapping: 다음 매핑 (또는 None)
            rate_ratio: 발화 속도 비율 (target/source)
            
        Returns:
            (시작 이완 값, 종료 이완 값) 튜플 (초 단위)
        """
        source = mapping.source
        target = mapping.target
        
        # 기본값: 이완 없음
        start_relaxation = 0.0
        end_relaxation = 0.0
        
        # 소스와 타겟 지속 시간
        source_duration = source.duration
        target_duration = target.duration
        
        # 시간 차이 및 필요 이완 계산
        time_diff = target_duration - source_duration
        
        # 이전/다음 매핑 간의 간격 계산
        pre_gap = 0.0
        post_gap = 0.0
        
        if prev_mapping:
            pre_gap = mapping.source.start_time - prev_mapping.source.end_time
        else:
            pre_gap = mapping.source.start_time  # 첫 문장이면 시작 시간이 갭
            
        if next_mapping:
            post_gap = next_mapping.source.start_time - mapping.source.end_time
        else:
            post_gap = 2.0  # 마지막 문장이면 충분한 갭 가정
        
        # 최대 허용 이완 계산
        max_start_relaxation = min(pre_gap * 0.8, source_duration * self.max_relaxation)
        max_end_relaxation = min(post_gap * 0.8, source_duration * self.max_relaxation)
        
        # 발화 속도 제어 로직
        if rate_ratio > self.max_speaking_rate:
            # 발화 속도가 너무 빠른 경우: 타겟 발화 시간을 늘려야 함
            needed_extension = target_duration * (rate_ratio / self.max_speaking_rate - 1)
            
            # 앞뒤로 이완 분배
            start_fraction = min(max_start_relaxation / (max_start_relaxation + max_end_relaxation), 0.5)
            start_relaxation = min(needed_extension * start_fraction, max_start_relaxation)
            end_relaxation = min(needed_extension * (1 - start_fraction), max_end_relaxation)
            
        elif time_diff < 0:
            # 타겟 지속 시간이 소스보다 짧은 경우
            needed_extension = abs(time_diff)
            
            # 앞뒤로 이완 분배 (필요한 만큼만)
            start_fraction = min(max_start_relaxation / (max_start_relaxation + max_end_relaxation), 0.5)
            start_relaxation = min(needed_extension * start_fraction, max_start_relaxation)
            end_relaxation = min(needed_extension * (1 - start_fraction), max_end_relaxation)
        
        logger.debug(f"발화 속도 비율: {rate_ratio:.2f}, 이완: 시작={start_relaxation:.3f}초, 종료={end_relaxation:.3f}초")
        return start_relaxation, end_relaxation
    
    def align_for_dubbing(self, mappings: List[SentenceMapping]) -> List[SentenceMapping]:
        """
        자동 더빙을 위한 프로소딕 얼라인먼트 수행
        
        Args:
            mappings: SentenceMapping 객체 리스트
            
        Returns:
            더빙용 얼라인먼트가 적용된 SentenceMapping 객체 리스트
        """
        # 기본 얼라인먼트 수행
        aligned_mappings = self.align_mappings(mappings)
        
        # 발화 속도 제어를 위한 추가 처리
        result = []
        for mapping in aligned_mappings:
            if not mapping.source.has_timing_info() or not mapping.target.has_timing_info():
                result.append(mapping)
                continue
            
            # 발화 속도 계산
            source_rate, target_rate = mapping.compute_speaking_rate()
            effective_target_duration = mapping.get_effective_target_duration()
            
            # 필요한 시간 팽창 계수 계산
            if target_rate > self.max_speaking_rate * source_rate:
                # 타겟 발화 속도가 너무 빠른 경우
                time_dilation = target_rate / (self.max_speaking_rate * source_rate)
            else:
                # 정상적인 경우
                time_dilation = 1.0
            
            # 새 매핑 생성 (기존 매핑 복사 후 time_dilation 업데이트)
            updated_mapping = SentenceMapping(
                source=mapping.source,
                target=mapping.target,
                similarity=mapping.similarity,
                time_dilation=time_dilation,
                start_relaxation=mapping.start_relaxation,
                end_relaxation=mapping.end_relaxation
            )
            
            result.append(updated_mapping)
            
            # 로깅
            iso_score = updated_mapping.get_isochrony_score()
            logger.info(f"더빙 얼라인먼트: 시간 팽창={time_dilation:.2f}, Isochrony={iso_score:.2f}")
        
        return result


class DynamicProgrammingAligner:
    """
    동적 프로그래밍을 사용한 프로소딕 얼라인먼트 클래스
    
    이 클래스는 논문에서 제시된 동적 프로그래밍 기반 얼라인먼트 알고리즘 구현
    """
    
    def __init__(
        self,
        max_relaxation: float = ALIGNMENT_MAX_RELAXATION,
        max_speaking_rate: float = SPEAKING_RATE_MAX,
        relaxation_steps: int = 8  # 이완 단계 수 (양수/음수 각각)
    ):
        """
        동적 프로그래밍 얼라이너 초기화
        
        Args:
            max_relaxation: 시간 경계 최대 이완 비율
            max_speaking_rate: 최대 허용 발화 속도 비율
            relaxation_steps: 이완 단계 수
        """
        self.max_relaxation = max_relaxation
        self.max_speaking_rate = max_speaking_rate
        self.relaxation_steps = relaxation_steps
        
        # 이완 값 배열 생성 (-max_relaxation에서 +max_relaxation까지)
        self.relaxation_values = np.linspace(
            -max_relaxation, max_relaxation, 2 * relaxation_steps + 1)
    
    def align_mappings(self, mappings: List[SentenceMapping]) -> List[SentenceMapping]:
        """
        매핑된 문장 쌍의 프로소딕 얼라인먼트 수행 (동적 프로그래밍)
        
        Args:
            mappings: SentenceMapping 객체 리스트
            
        Returns:
            프로소딕 얼라인먼트가 적용된 SentenceMapping 객체 리스트
        """
        # 시간 정보가 없는 매핑은 처리할 수 없음
        valid_mappings = [m for m in mappings 
                         if m.source.has_timing_info() and m.target.has_timing_info()]
        
        if not valid_mappings:
            return mappings
        
        # 첫 번째 단계: 기본 세그먼트 결정 (이완 없이)
        segments = [(m.source, m.target, m.similarity) for m in valid_mappings]
        
        # 두 번째 단계: 최적 이완 값 탐색
        optimal_relaxations = self._find_optimal_relaxations(segments)
        
        # 결과 생성
        result = []
        for i, mapping in enumerate(valid_mappings):
            if i < len(optimal_relaxations):
                start_rel, end_rel = optimal_relaxations[i]
                
                # 새 매핑 생성
                updated_mapping = SentenceMapping(
                    source=mapping.source,
                    target=mapping.target,
                    similarity=mapping.similarity,
                    start_relaxation=start_rel,
                    end_relaxation=end_rel
                )
                
                result.append(updated_mapping)
            else:
                result.append(mapping)
        
        # 원래 리스트에서 시간 정보가 없는 매핑을 유지
        for mapping in mappings:
            if mapping not in valid_mappings:
                result.append(mapping)
        
        return result
    
    def _find_optimal_relaxations(
        self, 
        segments: List[Tuple[Sentence, Sentence, float]]
    ) -> List[Tuple[float, float]]:
        """
        동적 프로그래밍으로 최적의 이완 값 찾기
        
        Args:
            segments: (소스 문장, 타겟 문장, 유사도) 튜플 리스트
            
        Returns:
            (시작 이완, 종료 이완) 튜플 리스트
        """
        if not segments:
            return []
        
        # 각 세그먼트와 이완 값 조합에 대한 점수 계산
        n_segments = len(segments)
        n_relaxations = len(self.relaxation_values)
        
        # DP 테이블: [세그먼트 인덱스][왼쪽 이완 인덱스][오른쪽 이완 인덱스] -> 최대 점수
        dp = np.zeros((n_segments, n_relaxations, n_relaxations)) - float('inf')
        
        # 이전 선택 저장 (역추적용)
        prev = np.zeros((n_segments, n_relaxations, n_relaxations, 3), dtype=int)
        
        # 첫 번째 세그먼트 초기화
        for left_idx in range(n_relaxations):
            for right_idx in range(n_relaxations):
                left_rel = self.relaxation_values[left_idx]
                right_rel = self.relaxation_values[right_idx]
                
                # 첫 번째 세그먼트의 점수 계산
                source, target, _ = segments[0]
                score = self._calculate_segment_score(source, target, left_rel, right_rel)
                
                dp[0, left_idx, right_idx] = score
                prev[0, left_idx, right_idx] = [-1, -1, -1]  # 첫 번째 세그먼트는 이전이 없음
        
        # 나머지 세그먼트 처리
        for i in range(1, n_segments):
            source, target, _ = segments[i]
            
            # 현재 세그먼트의 왼쪽 이완 값에 대해
            for curr_left_idx in range(n_relaxations):
                # 현재 세그먼트의 오른쪽 이완 값에 대해
                for curr_right_idx in range(n_relaxations):
                    curr_left_rel = self.relaxation_values[curr_left_idx]
                    curr_right_rel = self.relaxation_values[curr_right_idx]
                    
                    # 이전 세그먼트의 오른쪽 이완 값에 대해 (현재 왼쪽과 호환되어야 함)
                    for prev_right_idx in range(n_relaxations):
                        # 이전/현재 세그먼트 간 호환성 검사
                        if not self._is_compatible(
                            self.relaxation_values[prev_right_idx], curr_left_rel,
                            segments[i-1][0], source):
                            continue
                        
                        # 현재 세그먼트의 점수 계산
                        segment_score = self._calculate_segment_score(
                            source, target, curr_left_rel, curr_right_rel)
                        
                        # 이전 세그먼트의 최적 왼쪽 이완 찾기
                        max_score = -float('inf')
                        best_prev_left = -1
                        
                        for prev_left_idx in range(n_relaxations):
                            prev_score = dp[i-1, prev_left_idx, prev_right_idx]
                            if prev_score > max_score:
                                max_score = prev_score
                                best_prev_left = prev_left_idx
                        
                        if max_score > -float('inf'):
                            total_score = max_score + segment_score
                            
                            # 더 좋은 점수를 찾았다면 업데이트
                            if total_score > dp[i, curr_left_idx, curr_right_idx]:
                                dp[i, curr_left_idx, curr_right_idx] = total_score
                                prev[i, curr_left_idx, curr_right_idx] = [i-1, best_prev_left, prev_right_idx]
        
        # 최적 이완 값 역추적
        relaxations = []
        
        # 마지막 세그먼트의 최적 이완 찾기
        max_score = -float('inf')
        best_left_idx, best_right_idx = -1, -1
        
        for left_idx in range(n_relaxations):
            for right_idx in range(n_relaxations):
                if dp[n_segments-1, left_idx, right_idx] > max_score:
                    max_score = dp[n_segments-1, left_idx, right_idx]
                    best_left_idx, best_right_idx = left_idx, right_idx
        
        if max_score == -float('inf'):
            logger.warning("유효한 이완 조합을 찾을 수 없습니다.")
            return [(0.0, 0.0)] * n_segments
        
        # 역추적
        i, left_idx, right_idx = n_segments-1, best_left_idx, best_right_idx
        while i >= 0:
            # 현재 세그먼트의 이완 값 저장
            relaxations.insert(0, (
                self.relaxation_values[left_idx],
                self.relaxation_values[right_idx]
            ))
            
            # 이전 세그먼트로 이동
            prev_i, prev_left_idx, prev_right_idx = prev[i, left_idx, right_idx]
            if prev_i < 0:
                break
                
            i, left_idx, right_idx = prev_i, prev_left_idx, prev_right_idx
        
        return relaxations
    
    def _calculate_segment_score(
        self, 
        source: Sentence, 
        target: Sentence, 
        left_relaxation: float, 
        right_relaxation: float
    ) -> float:
        """
        세그먼트와 이완 값 조합에 대한 점수 계산
        
        Args:
            source: 소스 문장
            target: 타겟 문장
            left_relaxation: 왼쪽 이완 값 (초)
            right_relaxation: 오른쪽 이완 값 (초)
            
        Returns:
            점수 (높을수록 좋음)
        """
        # 기본 점수: 이완이 작을수록 높음
        relaxation_penalty = abs(left_relaxation) + abs(right_relaxation)
        
        # 소스/타겟 지속 시간
        source_duration = source.duration
        target_duration = target.duration
        
        # 이완 적용 후 타겟 지속 시간
        effective_target_duration = target_duration + left_relaxation + right_relaxation
        
        # 발화 속도 점수
        source_rate = len(source.words) / source_duration if source_duration > 0 else 0
        target_rate = len(target.words) / effective_target_duration if effective_target_duration > 0 else 0
        
        # 발화 속도 비율이 1에 가까울수록 좋음 (로그 스케일)
        if source_rate > 0 and target_rate > 0:
            rate_ratio = target_rate / source_rate
            rate_score = -abs(np.log(rate_ratio))  # 로그 비율의 절대값 (0에 가까울수록 좋음)
        else:
            rate_score = -10  # 발화 속도를 계산할 수 없는 경우 큰 페널티
        
        # 이소크로니 점수 (타겟과 소스 지속 시간이 비슷할수록 좋음)
        time_diff = abs(source_duration - effective_target_duration)
        isochrony_score = 1.0 - min(1.0, time_diff / max(source_duration, effective_target_duration))
        
        # 발화 속도 한계 초과 페널티
        if target_rate > self.max_speaking_rate * source_rate:
            speaking_rate_penalty = -5.0 * (target_rate / (self.max_speaking_rate * source_rate) - 1.0)
        else:
            speaking_rate_penalty = 0.0
        
        # 최종 점수
        total_score = (
            -0.5 * relaxation_penalty +  # 이완 페널티
            3.0 * rate_score +           # 발화 속도 점수
            5.0 * isochrony_score +      # 이소크로니 점수
            speaking_rate_penalty        # 발화 속도 한계 초과 페널티
        )
        
        return total_score
    
    def _is_compatible(
        self,
        prev_right_relaxation: float,
        curr_left_relaxation: float,
        prev_source: Sentence,
        curr_source: Sentence
    ) -> bool:
        """
        이전/현재 세그먼트의 이완 값이 호환되는지 확인
        
        Args:
            prev_right_relaxation: 이전 세그먼트의 오른쪽 이완 값
            curr_left_relaxation: 현재 세그먼트의 왼쪽 이완 값
            prev_source: 이전 소스 문장
            curr_source: 현재 소스 문장
            
        Returns:
            호환되면 True, 아니면 False
        """
        # 세그먼트 간 간격
        gap = curr_source.start_time - prev_source.end_time
        
        # 최소 필요 간격 (이전 오른쪽 이완과 현재 왼쪽 이완의 합)
        min_gap_needed = prev_right_relaxation + curr_left_relaxation
        
        # 간격이 충분한지 확인
        return gap >= min_gap_needed