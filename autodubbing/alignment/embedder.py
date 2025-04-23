"""
다국어 문장 임베딩을 위한 모듈

이 모듈은 다국어 임베딩 모델을 사용하여 문장을 벡터로 변환하는 기능을 제공합니다.
다양한 언어로 된 문장을 같은 벡터 공간에 매핑하여 의미 기반 매칭을 가능하게 합니다.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Union, Any

from ..models.sentence import Sentence
from ..config import EMBEDDING_MODEL, EMBEDDING_MODEL_OPTIONS

logger = logging.getLogger(__name__)


class SentenceEmbedder:
    """문장 임베딩을 처리하는 클래스"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        문장 임베딩 모델 초기화
        
        Args:
            model_name: 사용할 임베딩 모델 이름
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """임베딩 모델 로드"""
        try:
            # 모델 이름이 별칭인 경우 실제 모델 이름으로 변환
            actual_model_name = EMBEDDING_MODEL_OPTIONS.get(self.model_name.lower(), self.model_name)
            
            # 사용할 라이브러리 결정 및 모델 로드
            if 'laser' in actual_model_name.lower():
                self._load_laser_model()
            else:
                self._load_sentence_transformers_model(actual_model_name)
                
            logger.info(f"문장 임베딩 모델 '{self.model_name}' 로드 완료")
        except Exception as e:
            logger.error(f"문장 임베딩 모델 로드 실패: {e}")
            raise
    
    def _load_sentence_transformers_model(self, model_name: str):
        """Sentence Transformers 모델 로드"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._embed_func = self._embed_with_sentence_transformers
        except ImportError:
            logger.error("sentence-transformers 라이브러리가 설치되지 않았습니다.")
            logger.error("pip install sentence-transformers 명령어로 설치하세요.")
            raise
    
    def _load_laser_model(self):
        """LASER 모델 로드"""
        try:
            from laserembeddings import Laser
            self.model = Laser()
            self._embed_func = self._embed_with_laser
        except ImportError:
            logger.error("laserembeddings 라이브러리가 설치되지 않았습니다.")
            logger.error("pip install laserembeddings 명령어로 설치하세요.")
            raise
    
    def _embed_with_sentence_transformers(self, texts: List[str], langs: List[str]) -> np.ndarray:
        """Sentence Transformers를 사용한 임베딩"""
        return self.model.encode(texts, show_progress_bar=False)
    
    def _embed_with_laser(self, texts: List[str], langs: List[str]) -> np.ndarray:
        """LASER를 사용한 임베딩"""
        return self.model.embed_sentences(texts, langs)
    
    def embed_sentences(self, sentences: List[Sentence]) -> List[Sentence]:
        """
        문장 리스트를 임베딩하고 임베딩 벡터를 Sentence 객체에 저장
        
        Args:
            sentences: Sentence 객체 리스트
            
        Returns:
            임베딩이 추가된 Sentence 객체 리스트
        """
        if not sentences:
            return []
        
        # 텍스트와 언어 코드 추출
        texts = [s.text for s in sentences]
        langs = [s.lang for s in sentences]
        
        # 임베딩 계산
        embeddings = self._embed_func(texts, langs)
        
        # 임베딩을 각 문장 객체에 저장
        for i, sentence in enumerate(sentences):
            sentence.embedding = embeddings[i]
        
        logger.info(f"{len(sentences)}개 문장의 임베딩 계산 완료")
        return sentences
    
    def get_embedding(self, text: str, lang: str) -> np.ndarray:
        """
        단일 문장의 임베딩 벡터 반환
        
        Args:
            text: 임베딩할 문장 텍스트
            lang: 언어 코드
            
        Returns:
            임베딩 벡터 (numpy 배열)
        """
        embeddings = self._embed_func([text], [lang])
        return embeddings[0]
    
    @staticmethod
    def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        두 임베딩 벡터 간의 코사인 유사도 계산
        
        Args:
            embedding1: 첫 번째 임베딩 벡터
            embedding2: 두 번째 임베딩 벡터
            
        Returns:
            코사인 유사도 (0.0 ~ 1.0)
        """
        # 벡터 정규화
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        # 0으로 나누기 방지
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 코사인 유사도 계산
        cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # -1.0 ~ 1.0 범위를 0.0 ~ 1.0 범위로 변환 (옵션)
        # similarity = (cosine_sim + 1) / 2
        
        # 또는 그냥 코사인 유사도 반환 (-1.0 ~ 1.0)
        return float(max(0.0, cosine_sim))  # 음수 유사도는 0으로 처리
    
    @staticmethod
    def compute_similarity_matrix(
        source_sentences: List[Sentence], 
        target_sentences: List[Sentence]
    ) -> np.ndarray:
        """
        두 문장 집합 간의 유사도 행렬 계산
        
        Args:
            source_sentences: 소스 문장 리스트
            target_sentences: 타겟 문장 리스트
            
        Returns:
            유사도 행렬 (shape: len(source_sentences) x len(target_sentences))
        """
        # 임베딩이 계산되지 않은 문장이 있는지 확인
        if any(s.embedding is None for s in source_sentences + target_sentences):
            logger.warning("일부 문장의 임베딩이 계산되지 않았습니다. 유사도 계산 전에 embed_sentences()를 호출하세요.")
            return np.zeros((len(source_sentences), len(target_sentences)))
        
        # 유사도 행렬 초기화
        similarity_matrix = np.zeros((len(source_sentences), len(target_sentences)))
        
        # 모든 문장 쌍의 유사도 계산
        for i, source in enumerate(source_sentences):
            for j, target in enumerate(target_sentences):
                similarity_matrix[i, j] = SentenceEmbedder.compute_similarity(
                    source.embedding, target.embedding)
        
        return similarity_matrix