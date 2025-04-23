"""
TTS 제어 및 합성음성 생성 모듈

이 모듈은 프로소딕 얼라인먼트 결과를 기반으로 TTS 엔진을 제어하여
자연스러운 합성 음성을 생성합니다.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional

# TTS 라이브러리 import (사용하는 TTS 엔진에 따라 다름)
# 예: gTTS, pyttsx3, Amazon Polly, Google Cloud TTS 등
import gtts
# 또는 다른 TTS 라이브러리

from ..models.sentence import SentenceMapping
from ..evaluation.metrics import DubbingEvaluator

logger = logging.getLogger(__name__)


class TTSController:
    """TTS 엔진을 제어하여 프로소딕 얼라인먼트 기반 합성 음성을 생성하는 클래스"""
    
    def __init__(self, tts_engine: str = 'gtts', voice_id: str = None):
        """
        TTS 컨트롤러 초기화
        
        Args:
            tts_engine: 사용할 TTS 엔진 ('gtts', 'pyttsx3', 'polly', 'gcloud' 등)
            voice_id: 사용할 음성 ID (엔진에 따라 다름)
        """
        self.tts_engine = tts_engine
        self.voice_id = voice_id
        self._setup_tts_engine()
        
        logger.info(f"TTS 컨트롤러 초기화 완료 (엔진: {tts_engine})")
    
    def _setup_tts_engine(self):
        """TTS 엔진 설정"""
        # 사용하는 TTS 엔진에 따라 구현
        if self.tts_engine == 'gtts':
            self.synthesize_func = self._synthesize_gtts
        # 다른 엔진에 대한 구현 추가
        else:
            logger.warning(f"지원되지 않는 TTS 엔진: {self.tts_engine}")
            self.synthesize_func = self._synthesize_gtts  # 기본값
    
    def _synthesize_gtts(self, text: str, output_path: str, lang: str = 'en', 
                        speed: float = 1.0) -> bool:
        """
        gTTS를 사용한 음성 합성
        
        Args:
            text: 합성할 텍스트
            output_path: 출력 파일 경로
            lang: 언어 코드
            speed: 발화 속도 (1.0 = 기본 속도)
            
        Returns:
            성공 여부
        """
        try:
            # gTTS는 직접적인 속도 조절을 지원하지 않지만,
            # 후처리 단계에서 속도를 조절할 수 있음
            tts = gtts.gTTS(text=text, lang=lang, slow=False)
            tts.save(output_path)
            
            # 필요시 후처리로 속도 조절 (예: pydub 사용)
            if speed != 1.0:
                self._adjust_audio_speed(output_path, speed)
            
            return True
        except Exception as e:
            logger.error(f"음성 합성 실패: {e}")
            return False
    
    def _adjust_audio_speed(self, audio_path: str, speed: float) -> None:
        """
        오디오 파일의 속도 조절
        
        Args:
            audio_path: 오디오 파일 경로
            speed: 속도 비율 (1.0 = 원본 속도)
        """
        try:
            from pydub import AudioSegment
            
            # 오디오 파일 로드
            sound = AudioSegment.from_file(audio_path)
            
            # 속도 조절
            # pydub에서는 샘플 레이트를 조절하여 속도 변경
            new_sample_rate = int(sound.frame_rate * speed)
            sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
            
            # 원래 샘플 레이트로 내보내기 (속도 효과 적용)
            sound = sound.set_frame_rate(sound.frame_rate)
            
            # 원본 파일 덮어쓰기
            sound.export(audio_path, format="mp3")
            
            logger.info(f"오디오 속도 조절 완료: {speed}x")
        except Exception as e:
            logger.error(f"오디오 속도 조절 실패: {e}")
    
    def _add_silence(self, audio_path: str, start_silence: float, end_silence: float) -> None:
        """
        오디오 파일 앞뒤에 묵음 추가
        
        Args:
            audio_path: 오디오 파일 경로
            start_silence: 시작 부분 묵음 길이 (초)
            end_silence: 종료 부분 묵음 길이 (초)
        """
        try:
            from pydub import AudioSegment
            
            # 오디오 파일 로드
            sound = AudioSegment.from_file(audio_path)
            
            # 묵음 생성
            start_silence_segment = AudioSegment.silent(duration=int(start_silence * 1000))
            end_silence_segment = AudioSegment.silent(duration=int(end_silence * 1000))
            
            # 묵음 추가
            new_sound = start_silence_segment + sound + end_silence_segment
            
            # 원본 파일 덮어쓰기
            new_sound.export(audio_path, format="mp3")
            
            logger.info(f"묵음 추가 완료: 시작={start_silence}초, 종료={end_silence}초")
        except Exception as e:
            logger.error(f"묵음 추가 실패: {e}")
    
    def synthesize_aligned_speech(
        self,
        mappings: List[SentenceMapping],
        output_dir: str,
        prefix: str = "segment"
    ) -> List[str]:
        """
        얼라인된 매핑 정보를 기반으로 음성 합성
        
        Args:
            mappings: 얼라인된 SentenceMapping 객체 리스트
            output_dir: 출력 디렉토리
            prefix: 출력 파일 이름 접두어
            
        Returns:
            생성된 오디오 파일 경로 리스트
        """
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        output_files = []
        
        for i, mapping in enumerate(mappings):
            # 타겟 문장 및 언어 정보
            target_text = mapping.target.text
            target_lang = mapping.target.lang
            
            # 출력 파일 경로
            output_path = os.path.join(output_dir, f"{prefix}_{i+1}.mp3")
            
            # 발화 속도 계산 (time_dilation 적용)
            speed = 1.0 / mapping.time_dilation if mapping.time_dilation > 0 else 1.0
            
            # 음성 합성
            success = self.synthesize_func(target_text, output_path, target_lang, speed)
            
            if success:
                # 필요한 묵음 추가 (relaxation 적용)
                if mapping.start_relaxation != 0 or mapping.end_relaxation != 0:
                    # 양수 값은 묵음 추가, 음수 값은 묵음 제거 (실제 오디오 트리밍 필요)
                    start_silence = max(0, mapping.start_relaxation)
                    end_silence = max(0, mapping.end_relaxation)
                    
                    if start_silence > 0 or end_silence > 0:
                        self._add_silence(output_path, start_silence, end_silence)
                
                logger.info(f"세그먼트 {i+1} 합성 완료: {output_path}")
                output_files.append(output_path)
            else:
                logger.error(f"세그먼트 {i+1} 합성 실패")
        
        return output_files
    
    def concatenate_audio_segments(
        self,
        audio_files: List[str],
        output_path: str
    ) -> bool:
        """
        개별 오디오 세그먼트를 하나로 연결
        
        Args:
            audio_files: 오디오 파일 경로 리스트
            output_path: 최종 출력 파일 경로
            
        Returns:
            성공 여부
        """
        try:
            from pydub import AudioSegment
            
            # 첫 번째 파일로 시작
            if not audio_files:
                logger.error("연결할 오디오 파일이 없습니다.")
                return False
            
            combined = AudioSegment.from_file(audio_files[0])
            
            # 나머지 파일 추가
            for audio_file in audio_files[1:]:
                segment = AudioSegment.from_file(audio_file)
                combined += segment
            
            # 최종 파일 저장
            combined.export(output_path, format="mp3")
            
            logger.info(f"{len(audio_files)}개 오디오 세그먼트 연결 완료: {output_path}")
            return True
        except Exception as e:
            logger.error(f"오디오 세그먼트 연결 실패: {e}")
            return False
    
    def evaluate_synthesis_quality(
        self,
        original_audio_path: str,
        synthesized_audio_path: str
    ) -> Dict[str, float]:
        """
        합성된 오디오의 품질 평가
        
        Args:
            original_audio_path: 원본 오디오 파일 경로
            synthesized_audio_path: 합성된 오디오 파일 경로
            
        Returns:
            평가 지표 사전
        """
        # 실제 구현은 오디오 분석 라이브러리 사용 필요
        # 여기서는 더미 구현만 제공
        
        logger.info(f"합성 오디오 품질 평가: {synthesized_audio_path}")
        
        # 더미 평가 결과
        metrics = {
            'naturalness': 0.8,
            'intelligibility': 0.85,
            'timing_accuracy': 0.9,
            'overall': 0.85
        }
        
        return metrics