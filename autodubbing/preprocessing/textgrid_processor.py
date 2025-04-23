"""
TextGrid 파일 처리를 위한 모듈

이 모듈은 TextGrid 파일에서 문장 정보를 추출하고 처리하는 기능을 제공합니다.
TextGrid는 음성 파일의 단어별 타임스탬프를 포함하는 파일 형식입니다.
"""

import logging
import re
from typing import List, Tuple, Dict, Optional
from textgrid import TextGrid, IntervalTier

from ..models.sentence import Sentence
from ..config import MIN_PAUSE_THRESHOLD, MIN_PHRASE_LENGTH

logger = logging.getLogger(__name__)


class TextGridProcessor:
    """TextGrid 파일을 처리하는 클래스"""
    
    @staticmethod
    def load_textgrid(file_path: str) -> Optional[TextGrid]:
        """
        TextGrid 파일 로드
        
        Args:
            file_path: TextGrid 파일 경로
            
        Returns:
            TextGrid 객체 또는 로드 실패 시 None
        """
        try:
            tg = TextGrid.fromFile(file_path)
            logger.info(f"TextGrid 파일 로드 성공: {file_path}")
            return tg
        except Exception as e:
            logger.error(f"TextGrid 파일 로드 실패: {file_path}, 오류: {e}")
            return None
    
    @staticmethod
    def extract_words_from_tier(tier: IntervalTier) -> List[Tuple[str, float, float]]:
        """
        TextGrid 계층(tier)에서 단어 및 타임스탬프 추출
        
        Args:
            tier: TextGrid의 IntervalTier 객체
            
        Returns:
            단어, 시작 시간, 종료 시간의 튜플 리스트
        """
        words = []
        for interval in tier:
            # 빈 mark는 건너뜀
            mark = interval.mark.strip()
            if not mark:
                continue
            
            words.append((mark, interval.minTime, interval.maxTime))
        
        return words
    
    @staticmethod
    def segment_into_sentences(
        words: List[Tuple[str, float, float]], 
        lang: str,
        min_pause: float = MIN_PAUSE_THRESHOLD
    ) -> List[Sentence]:
        """
        단어 리스트를 문장으로 세그먼트화
        
        Args:
            words: 단어, 시작 시간, 종료 시간의 튜플 리스트
            lang: 언어 코드
            min_pause: 문장 구분을 위한 최소 휴지 시간(초)
            
        Returns:
            Sentence 객체 리스트
        """
        if not words:
            return []
        
        sentences = []
        current_words = []
        current_word_texts = []
        sentence_start_time = words[0][1]
        last_end_time = words[0][2]
        sentence_idx = 0
        
        for i, (word, start_time, end_time) in enumerate(words):
            # 긴 휴지(pause) 감지 또는 문장 종결 표현 감지
            is_long_pause = (i > 0 and start_time - last_end_time >= min_pause)
            is_end_of_sentence = TextGridProcessor._is_sentence_end(word, lang)
            
            if (is_long_pause or is_end_of_sentence) and current_words:
                # 현재까지의 단어로 문장 생성
                sentence_text = " ".join(current_word_texts)
                
                sentences.append(Sentence(
                    text=sentence_text,
                    words=current_word_texts.copy(),
                    lang=lang,
                    start_time=sentence_start_time,
                    end_time=last_end_time,
                    index=sentence_idx
                ))
                sentence_idx += 1
                
                # 새 문장 시작
                current_words = []
                current_word_texts = []
                sentence_start_time = start_time
            
            # 현재 단어 추가
            current_words.append((word, start_time, end_time))
            current_word_texts.append(word)
            last_end_time = end_time
        
        # 마지막 문장 처리
        if current_words:
            sentence_text = " ".join(current_word_texts)
            
            sentences.append(Sentence(
                text=sentence_text,
                words=current_word_texts,
                lang=lang,
                start_time=sentence_start_time,
                end_time=last_end_time,
                index=sentence_idx
            ))
        
        # 지나치게 짧은 문장 병합 (옵션)
        sentences = TextGridProcessor._merge_short_sentences(sentences)
        
        logger.info(f"TextGrid에서 {len(sentences)}개 문장 추출 완료 (언어: {lang})")
        return sentences
    
    @staticmethod
    def _is_sentence_end(word: str, lang: str) -> bool:
        """
        단어가 문장 종결 표현인지 확인
        
        Args:
            word: 검사할 단어
            lang: 언어 코드
            
        Returns:
            문장 종결 표현이면 True, 아니면 False
        """
        # 언어별 문장 종결 표현 패턴
        end_patterns = {
            'ko': r'[.!?…]$',           # 한국어
            'en': r'[.!?…]$',           # 영어
            'es': r'[.!?…¡¿]$',         # 스페인어
            'fr': r'[.!?…]$',           # 프랑스어
            'de': r'[.!?…]$',           # 독일어
            'it': r'[.!?…]$',           # 이탈리아어
            'ja': r'[。！？…]$',         # 일본어
            'zh': r'[。！？…]$',         # 중국어
        }
        
        # 해당 언어 패턴이 없으면 기본 패턴 사용
        pattern = end_patterns.get(lang, r'[.!?…]$')
        
        return bool(re.search(pattern, word))
    
    @staticmethod
    def _merge_short_sentences(sentences: List[Sentence]) -> List[Sentence]:
        """
        너무 짧은 문장을 앞/뒤 문장과 병합
        
        Args:
            sentences: Sentence 객체 리스트
            
        Returns:
            병합된 Sentence 객체 리스트
        """
        if len(sentences) <= 1:
            return sentences
        
        result = []
        i = 0
        
        while i < len(sentences):
            current = sentences[i]
            
            # 단어 수가 MIN_PHRASE_LENGTH보다 적고, 다음 문장이 존재하는 경우
            if (len(current.words) < MIN_PHRASE_LENGTH and 
                i + 1 < len(sentences)):
                # 다음 문장과 병합
                next_sent = sentences[i + 1]
                merged = current.merge_with(next_sent)
                result.append(merged)
                i += 2  # 두 문장을 건너뜀
            else:
                result.append(current)
                i += 1
        
        return result
    
    @staticmethod
    def extract_sentences_from_textgrid(
        file_path: str, 
        tier_name: str = "words",
        lang: str = "en",
        min_pause: float = MIN_PAUSE_THRESHOLD
    ) -> List[Sentence]:
        """
        TextGrid 파일에서 문장을 추출하는 통합 메서드
        
        Args:
            file_path: TextGrid 파일 경로
            tier_name: 추출할 계층 이름
            lang: 언어 코드
            min_pause: 문장 구분을 위한 최소 휴지 시간(초)
            
        Returns:
            Sentence 객체 리스트
        """
        tg = TextGridProcessor.load_textgrid(file_path)
        if not tg:
            return []
        
        try:
            tier = tg.getFirst(tier_name)
            words = TextGridProcessor.extract_words_from_tier(tier)
            sentences = TextGridProcessor.segment_into_sentences(words, lang, min_pause)
            return sentences
        except Exception as e:
            logger.error(f"TextGrid에서 문장 추출 실패: {e}")
            return []