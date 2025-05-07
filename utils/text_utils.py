# utils/text_utils.py
"""
텍스트 처리 유틸리티 함수
"""
import os
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """
    텍스트 정규화 (공백, 특수문자 정리)
    
    Args:
        text: 정규화할 텍스트
        
    Returns:
        정규화된 텍스트
    """
    # 공백 정리
    text = text.strip()
    
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 텍스트가 비어있지 않은지 확인
    if not text:
        return ""
    
    return text

def extract_sentences(text: str) -> List[str]:
    """
    텍스트를 문장 단위로 분리
    
    Args:
        text: 분리할 텍스트
        
    Returns:
        문장 목록
    """
    # 문장 분리 정규식 패턴 (마침표, 물음표, 느낌표 등 구두점으로 구분)
    pattern = r'(?<=[.!?])\s+'
    
    # 텍스트 분리
    sentences = re.split(pattern, text)
    
    # 빈 문장 제거 및 정규화
    sentences = [normalize_text(sentence) for sentence in sentences if sentence.strip()]
    
    return sentences

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    두 텍스트 간의 유사도 계산
    
    Args:
        text1: 첫 번째 텍스트
        text2: 두 번째 텍스트
        
    Returns:
        유사도 점수 (0.0 ~ 1.0)
    """
    # 텍스트 정규화
    text1 = normalize_text(text1).lower()
    text2 = normalize_text(text2).lower()
    
    # 두 텍스트가 모두 비어있으면 완전 일치
    if not text1 and not text2:
        return 1.0
    
    # 둘 중 하나만 비어있으면 완전 불일치
    if not text1 or not text2:
        return 0.0
    
    # 완전 일치 확인
    if text1 == text2:
        return 1.0
    
    # 단어 집합 비교 (자카드 유사도)
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    jaccard = intersection / union if union > 0 else 0.0
    
    # 문자 집합 비교
    chars1 = set(text1)
    chars2 = set(text2)
    
    char_intersection = len(chars1.intersection(chars2))
    char_union = len(chars1.union(chars2))
    
    char_jaccard = char_intersection / char_union if char_union > 0 else 0.0
    
    # 길이 비율
    len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
    
    # 가중 평균 (단어 유사도 50%, 문자 유사도 30%, 길이 비율 20%)
    return 0.5 * jaccard + 0.3 * char_jaccard + 0.2 * len_ratio

def read_text_file(file_path: str, encoding: str = 'utf-8') -> str:
    """
    텍스트 파일 읽기
    
    Args:
        file_path: 파일 경로
        encoding: 파일 인코딩
        
    Returns:
        파일 내용
    """
    try:
        logger.info(f"텍스트 파일 읽는 중: {file_path}")
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        return content
    except Exception as e:
        logger.error(f"텍스트 파일 읽기 실패: {e}")
        return ""

def write_text_file(text: str, file_path: str, encoding: str = 'utf-8') -> bool:
    """
    텍스트 파일 쓰기
    
    Args:
        text: 저장할 텍스트
        file_path: 파일 경로
        encoding: 파일 인코딩
        
    Returns:
        성공 여부
    """
    try:
        logger.info(f"텍스트 파일 저장 중: {file_path}")
        
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(text)
        
        logger.info(f"텍스트 파일 저장 완료: {file_path}")
        return True
    except Exception as e:
        logger.error(f"텍스트 파일 저장 실패: {e}")
        return False

def extract_text_from_textgrid(textgrid_path: str, tier_name: str = "words") -> str:
    """
    TextGrid 파일에서 특정 계층의 텍스트 추출
    
    Args:
        textgrid_path: TextGrid 파일 경로
        tier_name: 추출할 계층 이름
        
    Returns:
        추출된 텍스트
    """
    try:
        # textgrid 모듈 가져오기
        try:
            import textgrid
        except ImportError:
            logger.error("textgrid 모듈을 가져올 수 없습니다. 다음 명령어로 설치하세요: pip install textgrid")
            return ""
        
        # TextGrid 파일 로드
        tg = textgrid.TextGrid.fromFile(textgrid_path)
        
        # 지정된 계층 찾기
        target_tier = None
        for tier in tg.tiers:
            if tier.name == tier_name:
                target_tier = tier
                break
        
        if target_tier is None:
            logger.warning(f"계층 '{tier_name}'을 TextGrid에서 찾을 수 없습니다")
            return ""
        
        # 텍스트 추출
        words = []
        for interval in target_tier:
            if interval.mark.strip():
                words.append(interval.mark)
        
        return " ".join(words)
        
    except Exception as e:
        logger.error(f"TextGrid에서 텍스트 추출 실패: {e}")
        return ""