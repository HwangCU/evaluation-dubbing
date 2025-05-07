# core/processor.py
"""
TextGrid 및 오디오 파일 처리 모듈
TextGrid 파일을 처리하고 오디오 데이터를 로드하는 모듈입니다
"""
import os
import logging
import numpy as np
import librosa
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class TextGridProcessor:
    """TextGrid 파일에서 시간 정보와 음소 정보를 추출하는 클래스"""
    
    def __init__(self, word_tier_name: str = "words", phone_tier_name: str = "phones"):
        """
        TextGrid 프로세서 초기화
        
        Args:
            word_tier_name: 단어 계층 이름
            phone_tier_name: 음소 계층 이름
        """
        self.word_tier_name = word_tier_name
        self.phone_tier_name = phone_tier_name
        logger.info(f"TextGrid 프로세서 초기화: 단어 계층 '{word_tier_name}', 음소 계층 '{phone_tier_name}'")
    
    def process_textgrid(self, textgrid_path: str) -> List[Dict[str, Any]]:
        """
        TextGrid 파일을 처리하여 세그먼트 정보 추출
        
        Args:
            textgrid_path: TextGrid 파일 경로
            
        Returns:
            시간 정보가 포함된 세그먼트 목록
        """
        try:
            # textgrid 모듈 가져오기
            try:
                import textgrid
            except ImportError:
                logger.error("textgrid 모듈을 가져올 수 없습니다. 다음 명령어로 설치하세요: pip install textgrid")
                raise
            
            # TextGrid 파일 로드
            tg = textgrid.TextGrid.fromFile(textgrid_path)
            
            # 관련 계층 찾기
            word_tier = None
            phone_tier = None
            
            for tier in tg.tiers:
                if tier.name == self.word_tier_name:
                    word_tier = tier
                elif tier.name == self.phone_tier_name:
                    phone_tier = tier
            
            if word_tier is None:
                logger.warning(f"단어 계층 '{self.word_tier_name}'을 TextGrid에서 찾을 수 없습니다")
                word_tier = tg.tiers[0]  # 첫 번째 계층을 대체재로 사용
            
            # 문장 구분 (구두점 또는 긴 무음 기준)
            segments = self._extract_segments(word_tier)
            
            # 각 세그먼트에 단어 단위 정보 추가
            for segment in segments:
                segment["words"] = self._extract_words(
                    word_tier, segment["start"], segment["end"]
                )
                
                # 음소 계층이 있는 경우 음소 정보 추가
                if phone_tier:
                    segment["phones"] = self._extract_phones(
                        phone_tier, segment["start"], segment["end"]
                    )
            
            logger.info(f"TextGrid에서 {len(segments)}개 세그먼트 추출 완료")
            return segments
            
        except Exception as e:
            logger.error(f"TextGrid 파일 처리 중 오류: {e}")
            # 기본 세그먼트 반환
            return [{
                "start": 0.0,
                "end": 60.0,  # 기본 1분 가정
                "text": "",
                "words": []
            }]
    
    def _extract_segments(self, word_tier) -> List[Dict[str, Any]]:
        """
        단어 계층에서 문장 단위 세그먼트를 추출
        
        Args:
            word_tier: TextGrid의 단어 계층
            
        Returns:
            시작/종료 시간과 텍스트가 포함된 세그먼트 목록
        """
        segments = []
        current_segment_start = 0.0
        current_segment_words = []
        
        # 문장의 끝을 나타내는 구두점
        sentence_end_markers = [".", "!", "?", "。", "！", "？"]
        
        # 문장 경계로 간주할 최소 무음 구간 (초)
        min_silence = 0.5
        
        for i, interval in enumerate(word_tier):
            word = interval.mark.strip() if interval.mark else ""
            
            # 빈 구간 건너뛰기
            if not word:
                # 긴 무음 구간이 있고 현재 세그먼트에 단어가 있는 경우
                if interval.duration() >= min_silence and current_segment_words:
                    # 현재 세그먼트 종료
                    segment_text = " ".join(current_segment_words)
                    segments.append({
                        "start": current_segment_start,
                        "end": interval.minTime,
                        "text": segment_text,
                        "duration": interval.minTime - current_segment_start
                    })
                    
                    # 무음 구간 이후 새 세그먼트 시작
                    current_segment_start = interval.maxTime
                    current_segment_words = []
                
                continue
            
            # 현재 세그먼트에 단어 추가
            current_segment_words.append(word)
            
            # 구두점으로 끝나는 단어인지 확인
            if any(word.endswith(marker) for marker in sentence_end_markers):
                # 현재 세그먼트 종료
                segment_text = " ".join(current_segment_words)
                segments.append({
                    "start": current_segment_start,
                    "end": interval.maxTime,
                    "text": segment_text,
                    "duration": interval.maxTime - current_segment_start
                })
                
                # 새 세그먼트 시작
                if i < len(word_tier) - 1:
                    current_segment_start = interval.maxTime
                    current_segment_words = []
        
        # 마지막 세그먼트가 비어있지 않은 경우 추가
        if current_segment_words:
            last_interval = word_tier[-1]
            segment_text = " ".join(current_segment_words)
            segments.append({
                "start": current_segment_start,
                "end": last_interval.maxTime,
                "text": segment_text,
                "duration": last_interval.maxTime - current_segment_start
            })
        
        return segments
    
    def _extract_words(
        self, word_tier, start_time: float, end_time: float
    ) -> List[Dict[str, Any]]:
        """
        세그먼트 내 단어 정보 추출
        
        Args:
            word_tier: TextGrid의 단어 계층
            start_time: 세그먼트 시작 시간
            end_time: 세그먼트 종료 시간
            
        Returns:
            시간 정보가 포함된 단어 정보 목록
        """
        words = []
        
        for interval in word_tier:
            # 세그먼트 시간 범위 밖의 구간 건너뛰기
            if interval.maxTime <= start_time or interval.minTime >= end_time:
                continue
            
            word = interval.mark.strip() if interval.mark else ""
            
            # 빈 구간 건너뛰기
            if not word:
                continue
            
            words.append({
                "word": word,
                "start": interval.minTime,
                "end": interval.maxTime,
                "duration": interval.duration()
            })
        
        return words
    
    def _extract_phones(
        self, phone_tier, start_time: float, end_time: float
    ) -> List[Dict[str, Any]]:
        """
        세그먼트 내 음소 정보 추출
        
        Args:
            phone_tier: TextGrid의 음소 계층
            start_time: 세그먼트 시작 시간
            end_time: 세그먼트 종료 시간
            
        Returns:
            시간 정보가 포함된 음소 정보 목록
        """
        phones = []
        
        for interval in phone_tier:
            # 세그먼트 시간 범위 밖의 구간 건너뛰기
            if interval.maxTime <= start_time or interval.minTime >= end_time:
                continue
            
            phone = interval.mark.strip() if interval.mark else ""
            
            # 빈 구간 건너뛰기 (무음)
            if not phone:
                continue
            
            phones.append({
                "phone": phone,
                "start": interval.minTime,
                "end": interval.maxTime,
                "duration": interval.duration(),
                "is_vowel": self._is_vowel(phone)
            })
        
        return phones
    
    def _is_vowel(self, phone: str) -> bool:
        """
        음소가 모음인지 판별
        
        Args:
            phone: 음소 문자열
            
        Returns:
            모음 여부
        """
        # 영어 모음 목록 (IPA)
        english_vowels = ['a', 'e', 'i', 'o', 'u', 'æ', 'ɑ', 'ɒ', 'ɔ', 'ɛ', 'ə', 'ɪ', 'ʊ', 'ʌ']
        
        # 한국어 모음 목록
        korean_vowels = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        
        # 추가 모음 패턴 (다양한 표기법 고려)
        vowel_patterns = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
        
        # 모음 여부 확인
        return (
            phone in english_vowels or
            phone in korean_vowels or
            any(pattern in phone for pattern in vowel_patterns)
        )


class AudioProcessor:
    """오디오 파일을 처리하고 특징을 추출하는 클래스"""
    
    def __init__(self, sample_rate: int = 22050):
        """
        오디오 프로세서 초기화
        
        Args:
            sample_rate: 샘플링 레이트
        """
        self.sample_rate = sample_rate
        logger.info(f"오디오 프로세서 초기화: 샘플링 레이트 {sample_rate}Hz")
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        오디오 파일 로드
        
        Args:
            audio_path: 오디오 파일 경로
            
        Returns:
            (오디오 데이터, 샘플링 레이트) 튜플
        """
        logger.info(f"오디오 파일 로드 중: {audio_path}")
        try:
            # librosa로 오디오 로드
            y, sr = librosa.load(audio_path, sr=None)
            
            # 원본 샘플링 레이트 유지 (리샘플링 없음)
            logger.info(f"오디오 로드 완료: 길이 {len(y)/sr:.2f}초, 샘플링 레이트 {sr}Hz")
            return y, sr
        except Exception as e:
            logger.error(f"오디오 파일 로드 중 오류: {e}")
            # 빈 오디오 반환
            return np.zeros(self.sample_rate), self.sample_rate
    
    def extract_features(
        self, 
        y: np.ndarray, 
        sr: int, 
        frame_length: int = 1024, 
        hop_length: int = 256
    ) -> Dict[str, Any]:
        """
        오디오에서 특징 추출
        
        Args:
            y: 오디오 데이터
            sr: 샘플링 레이트
            frame_length: 프레임 길이
            hop_length: 홉 길이
            
        Returns:
            오디오 특징 딕셔너리 (피치, 에너지, MFCC 등)
        """
        logger.info("오디오 특징 추출 중...")
        
        features = {}
        
        # 1. 에너지 (RMS) 계산
        features["rms"] = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length
        )[0]
        
        # 2. 피치 (F0) 추출
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=sr, hop_length=hop_length
            )
            features["f0"] = f0
            features["voiced_flag"] = voiced_flag
        except Exception as e:
            logger.warning(f"피치 추출 중 오류: {e}. librosa.piptrack으로 대체합니다.")
            # 대체 피치 추출
            pitches, magnitudes = librosa.piptrack(
                y=y, sr=sr, fmin=80, fmax=400,
                hop_length=hop_length
            )
            
            f0 = []
            for t in range(pitches.shape[1]):
                index = np.argmax(magnitudes[:, t])
                f0.append(pitches[index, t])
            
            features["f0"] = np.array(f0)
        
        # 3. MFCC 계산
        features["mfcc"] = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=13, hop_length=hop_length
        )
        
        # 4. 스펙트럼 중심값 (Spectral Centroid)
        features["spectral_centroid"] = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=hop_length
        )[0]
        
        # 5. 온셋 감지 (Onset Detection)
        features["onset_env"] = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=hop_length
        )
        features["onsets"] = librosa.onset.onset_detect(
            onset_envelope=features["onset_env"], sr=sr, hop_length=hop_length
        )
        
        # 6. 무음 구간 감지 (Silence Detection)
        # RMS 에너지의 평균의 10%를 임계값으로 사용
        threshold = np.mean(features["rms"]) * 0.1
        features["silence_mask"] = features["rms"] < threshold
        
        logger.info("오디오 특징 추출 완료")
        return features