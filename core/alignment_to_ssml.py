# alignment_to_ssml.py
"""
Alignment 결과를 SSML로 변환하는 모듈
"""
import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# 로깅 설정
logger = logging.getLogger(__name__)

class AlignmentToSSML:
    """
    Segment 정렬 결과를 TTS용 SSML로 변환하는 클래스
    """
    
    def __init__(self, 
                 min_speaking_rate: float = 0.7, 
                 max_speaking_rate: float = 1.5,
                 lang_coefficient: Dict[str, float] = None):
        """
        초기화
        
        Args:
            min_speaking_rate: 최소 발화 속도 (기본값: 0.7)
            max_speaking_rate: 최대 발화 속도 (기본값: 1.5)
            lang_coefficient: 언어별 발화 속도 계수 (기본값: None)
        """
        self.min_speaking_rate = min_speaking_rate
        self.max_speaking_rate = max_speaking_rate
        
        # 언어별 발화 속도 계수 (기본값)
        if lang_coefficient is None:
            self.lang_coefficient = {
                "en": 1.0,    # 영어 (기준)
                "ko": 0.85,   # 한국어 (영어보다 약간 느림)
                "ja": 0.9,    # 일본어
                "zh": 0.8,    # 중국어
                "es": 1.1,    # 스페인어 (영어보다 약간 빠름)
                "fr": 1.05,   # 프랑스어
                "de": 0.95,   # 독일어
                "it": 1.15    # 이탈리아어 (영어보다 빠름)
            }
        else:
            self.lang_coefficient = lang_coefficient
        
        logger.info(f"AlignmentToSSML 초기화: 속도 범위 {min_speaking_rate}-{max_speaking_rate}")
    
    def convert_to_ssml(self, 
                       aligned_segments: List[Dict[str, Any]],
                       src_lang: str = "en",
                       tgt_lang: str = "ko") -> str:
        """
        정렬된 세그먼트를 SSML로 변환
        
        Args:
            aligned_segments: 정렬된 세그먼트 목록
            src_lang: 소스 언어 코드
            tgt_lang: 타겟 언어 코드
            
        Returns:
            SSML 문자열
        """
        if not aligned_segments:
            logger.warning("변환할 정렬 세그먼트가 없습니다.")
            return "<speak></speak>"
        
        # 언어별 계수 가져오기
        src_coef = self.lang_coefficient.get(src_lang, 1.0)
        tgt_coef = self.lang_coefficient.get(tgt_lang, 1.0)
        lang_ratio = tgt_coef / src_coef
        
        # SSML 파트 구성
        ssml_parts = []
        
        # 화면 표시 타입 (on-screen, off-screen)
        current_display_type = None
        
        for i, segment in enumerate(aligned_segments):
            # 세그먼트 정보 추출
            src_duration = segment.get("src_duration", 0)
            tgt_duration = segment.get("tgt_duration", 0)
            tgt_text = segment.get("tgt_text", "")
            
            # 세그먼트가 비어있으면 건너뛰기
            if not tgt_text.strip():
                continue
            
            # 화면 표시 타입 확인 (있는 경우)
            display_type = segment.get("display_type", "on-screen")
            
            # 화면 표시 타입이 변경된 경우 주석 추가
            if display_type != current_display_type:
                ssml_parts.append(f"\n  <!-- {display_type} 구간 시작 -->")
                current_display_type = display_type
            
            # 발화 속도 계산
            speaking_rate = self._calculate_speaking_rate(
                segment, lang_ratio, display_type == "off-screen"
            )
            
            # 앞뒤 휴지(pause) 시간 계산
            break_before, break_after = self._calculate_breaks(segment, i, aligned_segments)
            
            # 앞 휴지 추가 (첫 세그먼트가 아니고 휴지가 필요한 경우)
            if i > 0 and break_before > 0:
                ssml_parts.append(f'\n  <break time="{break_before}ms"/>')
            
            # 억양(pitch) 및 볼륨(volume) 계산 (필요한 경우)
            pitch_attr = ""
            volume_attr = ""
            
            # 억양(pitch) 변경 - 항상 추가
            pitch_attr = f' pitch="+0%"'  # 기본값 (변화 없음)
            if "pitch_similarity" in segment and segment["pitch_similarity"] < 0.7:
                pitch_adjust = "+10%" if segment.get("src_high_pitch", False) else "-10%"
                pitch_attr = f' pitch="{pitch_adjust}"'

            # 볼륨(volume) 변경 - 항상 추가
            volume_attr = f' volume="medium"'  # 기본값
            if "energy_similarity" in segment and segment["energy_similarity"] < 0.7:
                volume_adjust = "loud" if segment.get("src_high_energy", False) else "soft"
                volume_attr = f' volume="{volume_adjust}"'
            
            # 발화 텍스트에 prosody 태그 적용
            if speaking_rate != 1.0 or pitch_attr or volume_attr:
                rate_str = f' rate="{speaking_rate:.2f}"' if speaking_rate != 1.0 else ""
                ssml_parts.append(f'\n  <prosody{rate_str}{pitch_attr}{volume_attr}>{tgt_text}</prosody>')
            else:
                ssml_parts.append(f'\n  {tgt_text}')
            
            # 뒤 휴지 추가 (필요한 경우)
            if break_after > 0:
                ssml_parts.append(f'\n  <break time="{break_after}ms"/>')
        
        # 최종 SSML 생성
        ssml = f"<speak>{''.join(ssml_parts)}\n</speak>"
        
        return ssml
    
    def _calculate_speaking_rate(self, 
                               segment: Dict[str, Any], 
                               lang_ratio: float,
                               is_off_screen: bool = False) -> float:
        """
        세그먼트에 대한 적절한 발화 속도 계산
        
        Args:
            segment: 세그먼트 정보
            lang_ratio: 언어 간 비율 계수
            is_off_screen: 화면 밖 표시 여부
            
        Returns:
            발화 속도 (1.0이 기본 속도)
        """
        # 기본 발화 속도 계산
        src_duration = segment.get("src_duration", 0)
        tgt_duration = segment.get("tgt_duration", 0)
        
        # 원본/타겟 텍스트 길이 비율
        src_text = segment.get("src_text", "")
        tgt_text = segment.get("tgt_text", "")
        
        # 단어 수 비율 (근사치)
        src_word_count = len(src_text.split())
        tgt_word_count = len(tgt_text.split())
        text_ratio = tgt_word_count / max(1, src_word_count)
        
        if src_duration <= 0 or tgt_duration <= 0:
            # 시간 정보가 없는 경우 기본값 반환
            return 1.0
        
        # 발화 속도 계산 로직
        # 1. 세그먼트 지속 시간 비율
        duration_ratio = src_duration / tgt_duration
        
        # 2. 텍스트 및 언어 특성 반영
        adjusted_ratio = duration_ratio * (1.0 / text_ratio) * lang_ratio
        
        # 3. 화면 표시 여부에 따른 조정
        # off-screen의 경우 더 자유롭게 조정 가능
        if is_off_screen:
            # 허용 범위 확장
            min_rate = self.min_speaking_rate * 0.8
            max_rate = self.max_speaking_rate * 1.2
        else:
            # on-screen은 좁은 범위로 유지
            min_rate = self.min_speaking_rate
            max_rate = self.max_speaking_rate
        
        # 4. 최종 속도 제한
        final_rate = max(min_rate, min(adjusted_ratio, max_rate))
        
        return final_rate
    
    def _calculate_breaks(self, 
                        segment: Dict[str, Any], 
                        index: int,
                        all_segments: List[Dict[str, Any]]) -> tuple:
        """
        앞뒤 휴지(pause) 시간 계산
        
        Args:
            segment: 현재 세그먼트
            index: 현재 세그먼트 인덱스
            all_segments: 모든 세그먼트 목록
            
        Returns:
            (앞 휴지 시간(ms), 뒤 휴지 시간(ms))
        """
        # 휴지 기본값
        break_before = 0
        break_after = 0
        
        # 앞 휴지 계산
        if index > 0:
            prev_segment = all_segments[index - 1]
            
            # 현재 세그먼트 시작과 이전 세그먼트 종료 사이의 시간
            src_gap = segment.get("src_start", 0) - prev_segment.get("src_end", 0)
            
            if src_gap > 0:
                # 최소 휴지 (300ms 이상인 경우만 적용)
                if src_gap >= 0.3:
                    break_before = int(src_gap * 1000)  # 초 -> 밀리초 변환
        
        # 뒤 휴지 계산
        if index < len(all_segments) - 1:
            next_segment = all_segments[index + 1]
            
            # 현재 세그먼트 종료와 다음 세그먼트 시작 사이의 시간
            src_gap = next_segment.get("src_start", 0) - segment.get("src_end", 0)
            
            if src_gap > 0:
                # 최소 휴지 (300ms 이상인 경우만 적용)
                if src_gap >= 0.3:
                    break_after = int(src_gap * 1000)  # 초 -> 밀리초 변환
        
        return break_before, break_after
    
    def save_ssml(self, 
                ssml: str, 
                output_path: str) -> None:
        """
        SSML을 파일로 저장
        
        Args:
            ssml: SSML 문자열
            output_path: 저장 경로
        """
        # 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ssml)
        
        logger.info(f"SSML이 {output_path}에 저장되었습니다.")

    def convert_alignment_file(self,
                             alignment_file: str,
                             output_file: str,
                             src_lang: str = "en",
                             tgt_lang: str = "ko") -> str:
        """
        JSON 형식의 alignment 파일을 SSML로 변환
        
        Args:
            alignment_file: 정렬 결과 JSON 파일 경로
            output_file: 출력 SSML 파일 경로
            src_lang: 소스 언어 코드
            tgt_lang: 타겟 언어 코드
            
        Returns:
            생성된 SSML 문자열
        """
        try:
            # 정렬 결과 파일 로드
            with open(alignment_file, 'r', encoding='utf-8') as f:
                aligned_segments = json.load(f)
            
            # SSML 변환
            ssml = self.convert_to_ssml(aligned_segments, src_lang, tgt_lang)
            
            # 파일로 저장
            self.save_ssml(ssml, output_file)
            
            return ssml
            
        except Exception as e:
            logger.error(f"정렬 파일 변환 중 오류 발생: {e}")
            return "<speak>오류가 발생했습니다.</speak>"