"""
의미 기반 문장 매칭을 위한 모듈

이 모듈은 임베딩 기반으로 두 언어 간 문장의 의미적 매칭을 수행합니다.
헝가리안 알고리즘을 사용하여 최적의 문장 매핑을 찾습니다.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Set, Optional

from ..models.sentence import Sentence, SentenceMapping
from ..config import SIMILARITY_THRESHOLD, ENFORCE_SEQUENTIAL
from .embedder import SentenceEmbedder

logger = logging.getLogger(__name__)


class SemanticMatcher:
    """의미 기반 문장 매칭을 처리하는 클래스"""
    
    def __init__(
        self, 
        embedder: Optional[SentenceEmbedder] = None,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        enforce_sequential: bool = ENFORCE_SEQUENTIAL
    ):
        """
        의미 기반 문장 매처 초기화
        
        Args:
            embedder: 문장 임베딩을 위한 SentenceEmbedder 객체
            similarity_threshold: 매핑 최소 유사도 임계값 (0.0~1.0)
            enforce_sequential: 순차적 매핑 강제 여부
        """
        self.embedder = embedder or SentenceEmbedder()
        self.similarity_threshold = similarity_threshold
        self.enforce_sequential = enforce_sequential
    
    def match_sentences(
        self, 
        source_sentences: List[Sentence], 
        target_sentences: List[Sentence]
    ) -> List[SentenceMapping]:
        """
        두 문장 집합 간의 최적 매핑 찾기
        
        Args:
            source_sentences: 소스 문장 리스트 (원본)
            target_sentences: 타겟 문장 리스트 (번역)
            
        Returns:
            SentenceMapping 객체 리스트
        """
        if not source_sentences or not target_sentences:
            logger.warning("빈 문장 리스트가 입력되었습니다.")
            return []
        
        # 1. 문장 임베딩 계산
        source_with_embeddings = self.embedder.embed_sentences(source_sentences)
        target_with_embeddings = self.embedder.embed_sentences(target_sentences)
        
        # 2. 유사도 행렬 계산
        similarity_matrix = self.embedder.compute_similarity_matrix(
            source_with_embeddings, target_with_embeddings)
        
        # 3. 최적 매핑 찾기
        raw_mappings = self._find_optimal_mapping(
            similarity_matrix, source_sentences, target_sentences)
        
        # 4. 필요시 순서 제약 적용
        if self.enforce_sequential:
            refined_mappings = self._refine_mappings_sequential(raw_mappings)
        else:
            refined_mappings = raw_mappings
        
        # 5. SentenceMapping 객체 생성
        result = []
        for source_idx, target_idx, similarity in refined_mappings:
            mapping = SentenceMapping(
                source=source_sentences[source_idx],
                target=target_sentences[target_idx],
                similarity=similarity
            )
            result.append(mapping)
            
            # 로깅
            logger.info(f"매핑: (유사도 {similarity:.4f})")
            logger.info(f"  소스: {source_sentences[source_idx]}")
            logger.info(f"  타겟: {target_sentences[target_idx]}")
        
        return result
    
    def _find_optimal_mapping(
        self, 
        similarity_matrix: np.ndarray,
        source_sentences: List[Sentence],
        target_sentences: List[Sentence]
    ) -> List[Tuple[int, int, float]]:
        """
        헝가리안 알고리즘을 사용해 최적의 매핑 찾기
        
        Args:
            similarity_matrix: 유사도 행렬
            source_sentences: 소스 문장 리스트
            target_sentences: 타겟 문장 리스트
            
        Returns:
            매핑 리스트 [(source_idx, target_idx, similarity), ...]
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            logger.error("scipy 라이브러리가 설치되지 않았습니다.")
            logger.error("pip install scipy 명령어로 설치하세요.")
            raise
            
        # 유사도가 높을수록 좋으므로 음수로 변환하여 비용 행렬로 만듦
        cost_matrix = -similarity_matrix
        
        # 헝가리안 알고리즘으로 최적 매핑 찾기
        source_indices, target_indices = linear_sum_assignment(cost_matrix)
        
        # 결과 형식화 및 임계값 필터링
        mappings = []
        for source_idx, target_idx in zip(source_indices, target_indices):
            similarity = similarity_matrix[source_idx, target_idx]
            if similarity >= self.similarity_threshold:
                mappings.append((source_idx, target_idx, similarity))
        
        # 유사도 기준 내림차순 정렬
        mappings.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"임계값 {self.similarity_threshold} 이상의 매핑 {len(mappings)}개 찾음")
        return mappings
    
    def _refine_mappings_sequential(
        self, 
        mappings: List[Tuple[int, int, float]]
    ) -> List[Tuple[int, int, float]]:
        """
        원본 문장 순서를 유지하도록 매핑 정제
        
        Args:
            mappings: 원본 매핑 리스트 [(source_idx, target_idx, similarity), ...]
            
        Returns:
            정제된 매핑 리스트
        """
        if not mappings:
            return []
            
        # 원본 순서를 유지하는 방향으로 정제
        refined = []
        used_source = set()
        used_target = set()
        
        # 유사도 순으로 정렬된 매핑에서 시작
        for source_idx, target_idx, similarity in mappings:
            # 이미 사용된 인덱스는 건너뜀
            if source_idx in used_source or target_idx in used_target:
                continue
                
            # 이전에 추가된 매핑이 있는지 확인
            if refined:
                last_source, last_target, _ = refined[-1]
                
                # 순서가 뒤집히는 경우 건너뜀 (교차 방지)
                if (source_idx < last_source and target_idx > last_target) or \
                   (source_idx > last_source and target_idx < last_target):
                    continue
            
            refined.append((source_idx, target_idx, similarity))
            used_source.add(source_idx)
            used_target.add(target_idx)
        
        # 원본 문장 순서대로 정렬
        refined.sort(key=lambda x: x[0])
        
        # 정제 결과 로깅
        logger.info(f"순차적 제약 적용 후 매핑 수: {len(refined)} (원본: {len(mappings)})")
        return refined
    
    def match_with_fallback(
        self, 
        source_sentences: List[Sentence], 
        target_sentences: List[Sentence]
    ) -> List[SentenceMapping]:
        """
        의미 기반 매칭을 시도하고, 실패할 경우 순서 기반 폴백 적용
        
        Args:
            source_sentences: 소스 문장 리스트
            target_sentences: 타겟 문장 리스트
            
        Returns:
            SentenceMapping 객체 리스트
        """
        # 먼저 의미 기반 매칭 시도
        mappings = self.match_sentences(source_sentences, target_sentences)
        
        # 매핑 결과가 충분한 경우 그대로 반환
        if len(mappings) >= min(len(source_sentences), len(target_sentences)) * 0.7:
            return mappings
        
        # 매핑 결과가 부족한 경우 순서 기반 폴백
        logger.warning("의미 기반 매핑이 충분하지 않아 순서 기반 폴백을 적용합니다.")
        
        fallback_mappings = []
        min_length = min(len(source_sentences), len(target_sentences))
        
        for i in range(min_length):
            source = source_sentences[i]
            target = target_sentences[i]
            
            # 임베딩이 없는 경우 계산
            if source.embedding is None:
                source.embedding = self.embedder.get_embedding(source.text, source.lang)
            if target.embedding is None:
                target.embedding = self.embedder.get_embedding(target.text, target.lang)
            
            # 유사도 계산
            similarity = self.embedder.compute_similarity(source.embedding, target.embedding)
            
            # SentenceMapping 객체 생성
            mapping = SentenceMapping(
                source=source,
                target=target,
                similarity=similarity
            )
            fallback_mappings.append(mapping)
        
        logger.info(f"순서 기반 폴백으로 {len(fallback_mappings)}개 매핑 생성")
        return fallback_mappings