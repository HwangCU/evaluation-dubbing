"""
텍스트 전처리와 문장 분리를 담당하는 모듈

이 모듈은 원본 텍스트를 정규화하고 문장 단위로 분리하는 기능을 제공합니다.
"""

import re
import logging
from typing import List, Dict

from ..models.sentence import Sentence

logger = logging.getLogger(__name__)


class TextProcessor:
    """텍스트 전처리 및 문장 분리를 담당하는 클래스"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        텍스트 정규화: 공백 정리, 특수문자 처리 등
        
        Args:
            text: 정규화할 텍스트
            
        Returns:
            정규화된 텍스트
        """
        # 여러 공백을 하나로 줄이기
        text = re.sub(r'\s+', ' ', text)
        # 양쪽 공백 제거
        text = text.strip()
        # 불필요한 제어 문자 제거
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        return text

    @staticmethod
    def split_into_sentences(text: str, lang: str = 'en') -> List[str]:
        """
        텍스트를 문장 단위로 분리
        
        Args:
            text: 분리할 텍스트
            lang: 언어 코드
            
        Returns:
            문장 텍스트 리스트
        """
        # 텍스트 정규화
        text = TextProcessor.normalize_text(text)
        
        # 언어별 문장 분리 패턴
        patterns = {
            'ko': r'([^.!?…]+[.!?…]+)',    # 한국어
            'en': r'([^.!?]+[.!?]+)',      # 영어
            'es': r'([^.!?¡¿]+[.!?]+)',    # 스페인어
            'fr': r'([^.!?]+[.!?]+)',      # 프랑스어
            'de': r'([^.!?]+[.!?]+)',      # 독일어
            'it': r'([^.!?]+[.!?]+)',      # 이탈리아어
            'ja': r'([^。！？]+[。！？]+)',  # 일본어
            'zh': r'([^。！？]+[。！？]+)',  # 중국어
        }
        
        # 해당 언어 패턴이 없으면 기본 패턴 사용
        pattern = patterns.get(lang, r'([^.!?]+[.!?]+)')
        
        sentences = re.findall(pattern, text)
        
        # 정규식으로 찾지 못한 마지막 부분이 있다면 추가
        leftover = re.sub(pattern, '', text).strip()
        if leftover:
            sentences.append(leftover)
        
        # 추가 정규화 적용
        return [TextProcessor.normalize_text(s) for s in sentences if s.strip()]

    @staticmethod
    def create_sentences_from_text(text: str, lang: str = 'en') -> List[Sentence]:
        """
        텍스트에서 Sentence 객체 리스트 생성
        
        Args:
            text: 원본 텍스트
            lang: 언어 코드
            
        Returns:
            Sentence 객체 리스트
        """
        sentence_texts = TextProcessor.split_into_sentences(text, lang)
        sentences = []
        
        for i, sentence_text in enumerate(sentence_texts):
            words = sentence_text.split()
            sentence = Sentence(
                text=sentence_text,
                words=words,
                lang=lang,
                index=i
            )
            sentences.append(sentence)
        
        logger.info(f"텍스트에서 {len(sentences)}개 문장 추출 완료 (언어: {lang})")
        return sentences
    
    @staticmethod
    def combine_consecutive_short_sentences(sentences: List[Sentence], min_words: int = 3) -> List[Sentence]:
        """
        연속된 짧은 문장들을 하나로 합치기
        
        Args:
            sentences: Sentence 객체 리스트
            min_words: 짧은 문장 기준 단어 수
            
        Returns:
            병합된 Sentence 객체 리스트
        """
        if len(sentences) <= 1:
            return sentences
        
        result = []
        i = 0
        
        while i < len(sentences):
            current = sentences[i]
            
            # 현재 문장이 짧고 다음 문장이 있는 경우
            if len(current.words) < min_words and i + 1 < len(sentences):
                next_sent = sentences[i + 1]
                
                # 두 문장이 같은 언어인 경우에만 병합
                if current.lang == next_sent.lang:
                    merged = current.merge_with(next_sent)
                    result.append(merged)
                    i += 2  # 두 문장을 건너뜀
                else:
                    result.append(current)
                    i += 1
            else:
                result.append(current)
                i += 1
        
        return result
    
    @staticmethod
    def add_timing_to_sentences_proportionally(sentences: List[Sentence], total_duration: float) -> List[Sentence]:
        """
        텍스트만으로 생성된 문장에 비례 시간 정보 추가 (더빙 시뮬레이션용)
        
        Args:
            sentences: 시간 정보가 없는 Sentence 객체 리스트
            total_duration: 전체 오디오 길이 (초)
            
        Returns:
            시간 정보가 추가된 Sentence 객체 리스트
        """
        if not sentences or total_duration <= 0:
            return sentences
        
        # 모든 단어 수 계산
        total_words = sum(len(s.words) for s in sentences)
        
        if total_words == 0:
            return sentences
        
        # 각 문장의 상대적 비율 계산 및 시간 할당
        current_time = 0.0
        result = []
        
        for sentence in sentences:
            word_ratio = len(sentence.words) / total_words
            sentence_duration = word_ratio * total_duration
            
            # 시간 정보 업데이트
            updated_sentence = Sentence(
                text=sentence.text,
                words=sentence.words,
                lang=sentence.lang,
                start_time=current_time,
                end_time=current_time + sentence_duration,
                index=sentence.index
            )
            
            result.append(updated_sentence)
            current_time += sentence_duration
        
        return result