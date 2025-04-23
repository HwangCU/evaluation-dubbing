"""
자동 더빙 시스템에서 사용하는 문장 및 매핑 관련 데이터 클래스

이 모듈은 문장 정보와 관련 메타데이터를 저장하고 관리하는 데이터 클래스를 정의합니다.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class Sentence:
    """문장 정보를 저장하는 데이터 클래스"""
    
    # 기본 속성
    text: str                     # 문장 텍스트 (전체)
    words: List[str]              # 단어 리스트
    lang: str                     # 언어 코드 (ko, en 등)
    
    # 시간 정보
    start_time: float = -1.0      # 시작 시간 (초)
    end_time: float = -1.0        # 종료 시간 (초)
    
    # 메타데이터
    index: int = -1               # 원본 스크립트에서의 인덱스
    
    # 내부 작업용 데이터
    embedding: Optional[np.ndarray] = None  # 문장 임베딩 벡터
    
    def __post_init__(self):
        """초기화 후 처리: 단어가 제공되지 않은 경우 텍스트에서 추출"""
        if not self.words and self.text:
            self.words = self.text.split()
    
    @property
    def duration(self) -> float:
        """문장 지속 시간 (초)"""
        if self.start_time >= 0 and self.end_time >= 0:
            return self.end_time - self.start_time
        return 0.0
    
    def has_timing_info(self) -> bool:
        """시간 정보가 있는지 확인"""
        return self.start_time >= 0 and self.end_time >= 0
    
    def merge_with(self, other: 'Sentence') -> 'Sentence':
        """다른 문장과 병합"""
        if self.lang != other.lang:
            raise ValueError("서로 다른 언어의 문장은 병합할 수 없습니다.")
        
        # 시간 정보 계산
        start_time = min(self.start_time, other.start_time) if self.has_timing_info() and other.has_timing_info() else -1.0
        end_time = max(self.end_time, other.end_time) if self.has_timing_info() and other.has_timing_info() else -1.0
        
        # 새 문장 생성
        return Sentence(
            text=f"{self.text} {other.text}".strip(),
            words=self.words + other.words,
            lang=self.lang,
            start_time=start_time,
            end_time=end_time,
            index=self.index  # 첫 번째 문장의 인덱스를 유지
        )
    
    def __str__(self) -> str:
        """문자열 표현: 디버깅 및 로깅용"""
        timing_info = f" ({self.start_time:.2f}s ~ {self.end_time:.2f}s)" if self.has_timing_info() else ""
        return f"Sentence[{self.lang}]({self.index}): {self.text}{timing_info}"


@dataclass
class SentenceMapping:
    """두 문장 간의 매핑 정보를 저장하는 데이터 클래스"""
    
    source: Sentence              # 소스 문장 (원본)
    target: Sentence              # 타겟 문장 (번역)
    similarity: float             # 문장 간 유사도 (0.0 ~ 1.0)
    
    # 얼라인먼트 정보
    time_dilation: float = 1.0    # 시간 확장/축소 비율 (1.0 = 변화 없음)
    
    # 프로소딕 얼라인먼트 세부 정보
    start_relaxation: float = 0.0  # 시작 시간 이완(relaxation) 값 (초)
    end_relaxation: float = 0.0    # 종료 시간 이완(relaxation) 값 (초)
    
    def compute_speaking_rate(self) -> Tuple[float, float]:
        """
        소스 및 타겟 문장의 발화 속도 계산
        
        Returns:
            Tuple[float, float]: (소스 발화 속도, 타겟 발화 속도)
                발화 속도는 초당 음절 수 또는 단어 수로 근사됨
        """
        if not self.source.has_timing_info() or not self.target.has_timing_info():
            return 0.0, 0.0
        
        # 간단한 근사: 단어 수를 지속 시간으로 나눔
        source_rate = len(self.source.words) / self.source.duration if self.source.duration > 0 else 0.0
        target_rate = len(self.target.words) / self.target.duration if self.target.duration > 0 else 0.0
        
        return source_rate, target_rate
    
    def get_effective_target_duration(self) -> float:
        """이완이 적용된 유효 타겟 지속 시간"""
        if not self.target.has_timing_info():
            return 0.0
        
        return self.target.duration + self.start_relaxation + self.end_relaxation
    
    def get_isochrony_score(self) -> float:
        """이 매핑의 isochrony 점수 계산 (0.0 ~ 1.0)"""
        if not self.source.has_timing_info() or not self.target.has_timing_info():
            return 0.0
        
        # 효과적인 타겟 지속 시간
        effective_target_duration = self.get_effective_target_duration()
        
        # 시간 차이의 절대값
        time_diff = abs(self.source.duration - effective_target_duration)
        
        # 더 긴 지속 시간으로 정규화
        max_duration = max(self.source.duration, effective_target_duration)
        if max_duration > 0:
            return 1.0 - (time_diff / max_duration)
        
        return 0.0
    
    def __str__(self) -> str:
        """문자열 표현: 디버깅 및 로깅용"""
        iso_score = self.get_isochrony_score()
        return (f"SentenceMapping (sim={self.similarity:.2f}, iso={iso_score:.2f}):\n"
                f"  Source: {self.source}\n"
                f"  Target: {self.target}")