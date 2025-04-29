# prosody_analyzer.py
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

class ProsodyAnalyzer:
    """
    원본 음성과 합성 음성 간의 음향적 유사도를 분석하는 클래스.
    pause, vowel 위치, 억양 패턴 등을 비교하여 유사도를 평가합니다.
    """
    
    def __init__(self):
        """프로소디 분석기를 초기화합니다."""
        logger.info("Initializing Prosody Analyzer")
        
    def analyze(
        self,
        src_audio_path: str,
        tgt_audio_path: str,
        src_segments: List[Dict[str, Any]],
        tgt_segments: List[Dict[str, Any]],
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        원본 음성과 합성 음성의 프로소디를 분석하고 유사도를 계산합니다.
        
        Args:
            src_audio_path: 원본 오디오 파일 경로
            tgt_audio_path: 합성 오디오 파일 경로
            src_segments: 원본 세그먼트 정보
            tgt_segments: 합성 세그먼트 정보
            output_dir: 결과를 저장할 디렉토리
            
        Returns:
            분석 결과를 담은 딕셔너리
        """
        if output_dir is None:
            output_dir = Path("output/analysis")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 오디오 파일 로드
        src_y, src_sr = librosa.load(src_audio_path, sr=None)
        tgt_y, tgt_sr = librosa.load(tgt_audio_path, sr=None)
        
        # 오디오 시각화 및 저장
        self._visualize_audio_comparison(
            src_y, src_sr, src_segments,
            tgt_y, tgt_sr, tgt_segments,
            output_dir / "audio_comparison.png"
        )
        
        # 유사도 점수 계산
        scores = {}
        
        # 1. 일시 정지(pause) 패턴 유사도
        pause_similarity = self._analyze_pause_similarity(
            src_y, src_sr, src_segments,
            tgt_y, tgt_sr, tgt_segments
        )
        scores["pause_similarity"] = pause_similarity
        
        # 2. 음높이(pitch) 패턴 유사도
        pitch_similarity = self._analyze_pitch_similarity(
            src_y, src_sr, src_segments,
            tgt_y, tgt_sr, tgt_segments
        )
        scores["pitch_similarity"] = pitch_similarity
        
        # 3. 에너지(energy) 패턴 유사도
        energy_similarity = self._analyze_energy_similarity(
            src_y, src_sr, src_segments,
            tgt_y, tgt_sr, tgt_segments
        )
        scores["energy_similarity"] = energy_similarity
        
        # 4. 리듬(rhythm) 패턴 유사도
        rhythm_similarity = self._analyze_rhythm_similarity(
            src_y, src_sr, src_segments,
            tgt_y, tgt_sr, tgt_segments
        )
        scores["rhythm_similarity"] = rhythm_similarity
        
        # 5. 모음(vowel) 길이 패턴 유사도 (가능한 경우)
        if all('phones' in segment for segment in src_segments + tgt_segments):
            vowel_similarity = self._analyze_vowel_similarity(
                src_segments, tgt_segments
            )
            scores["vowel_similarity"] = vowel_similarity
        
        # 종합 점수 계산 (가중 평균)
        weights = {
            "pause_similarity": 0.3,
            "pitch_similarity": 0.2,
            "energy_similarity": 0.2,
            "rhythm_similarity": 0.2,
            "vowel_similarity": 0.1  # vowel_similarity가 없는 경우 나머지에 분배
        }
        
        if "vowel_similarity" not in scores:
            # vowel_similarity가 없는 경우 가중치 재분배
            del weights["vowel_similarity"]
            weight_sum = sum(weights.values())
            weights = {k: v / weight_sum for k, v in weights.items()}
        
        overall_similarity = sum(
            scores[key] * weights[key] for key in weights.keys()
        )
        scores["overall"] = overall_similarity
        
        # 결과 로깅
        logger.info(f"Prosody analysis completed with overall similarity: {overall_similarity:.4f}")
        for key, value in scores.items():
            if key != "overall":
                logger.info(f"  - {key}: {value:.4f}")
        
        # 분석 결과 시각화
        self._visualize_similarity_scores(scores, output_dir / "similarity_scores.png")
        
        return scores
    
    def generate_recommendations(
        self,
        scores: Dict[str, float],
        current_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        분석 결과를 기반으로 TTS 파라미터 개선 권장사항을 생성합니다.
        
        Args:
            scores: 유사도 점수
            current_params: 현재 TTS 파라미터
            
        Returns:
            개선된 TTS 파라미터 및 변경 이유를 담은 딕셔너리
        """
        recommendations = {}
        changes = {}
        
        # 1. 일시 정지(pause) 패턴 개선
        if scores.get("pause_similarity", 1.0) < 0.7:
            # pause_similarity가 낮으면 speaking_rate 조정
            current_rate = current_params.get("speaking_rate", 1.0)
            
            if current_rate > 1.0:
                # 말하는 속도가 빠르면 줄임
                new_rate = max(0.8, current_rate - 0.1)
                recommendations["speaking_rate"] = new_rate
                changes["speaking_rate"] = "일시 정지 패턴 개선을 위해 말하는 속도 감소"
            else:
                # 말하는 속도가 이미 느리면 약간 증가
                new_rate = min(1.2, current_rate + 0.05)
                recommendations["speaking_rate"] = new_rate
                changes["speaking_rate"] = "일시 정지 패턴 개선을 위해 말하는 속도 소폭 증가"
        
        # 2. 음높이(pitch) 패턴 개선
        if scores.get("pitch_similarity", 1.0) < 0.7:
            current_pitch = current_params.get("pitch", 0.0)
            
            # 유사도가 낮으면 피치 조정 (현재 피치에 따라 방향 결정)
            if current_pitch < 0:
                new_pitch = min(2.0, current_pitch + 1.0)
            else:
                new_pitch = max(-2.0, current_pitch - 1.0)
                
            recommendations["pitch"] = new_pitch
            changes["pitch"] = "음높이 패턴 유사도 개선을 위한 피치 조정"
        
        # 3. 에너지(energy) 패턴 개선
        if scores.get("energy_similarity", 1.0) < 0.7:
            current_volume = current_params.get("volume", 1.0)
            
            # 에너지 유사도가 낮으면 볼륨 조정
            new_volume = min(1.5, current_volume + 0.2)
            recommendations["volume"] = new_volume
            changes["volume"] = "에너지 패턴 개선을 위한 볼륨 증가"
        
        # 4. 리듬(rhythm) 패턴 개선
        if scores.get("rhythm_similarity", 1.0) < 0.7:
            # 리듬 유사도가 낮으면 음성 스타일 변경
            current_style = current_params.get("voice_style", "neutral")
            
            if current_style == "neutral":
                recommendations["voice_style"] = "conversational"
                changes["voice_style"] = "리듬 패턴 개선을 위한 대화체 스타일 적용"
            elif current_style == "conversational":
                recommendations["voice_style"] = "formal"
                changes["voice_style"] = "리듬 패턴 개선을 위한 격식체 스타일 적용"
        
        # 5. 모음(vowel) 길이 패턴 개선
        if scores.get("vowel_similarity", 1.0) < 0.7:
            # 모음 길이 패턴 유사도가 낮으면 speaking_rate와 함께 pitch_range 조정
            if "speaking_rate" not in recommendations:
                current_rate = current_params.get("speaking_rate", 1.0)
                new_rate = max(0.8, current_rate - 0.05)
                recommendations["speaking_rate"] = new_rate
                changes["speaking_rate"] = "모음 길이 패턴 개선을 위한 말하는 속도 소폭 감소"
        
        # 종합 점수가 매우 낮은 경우 보다 과감한 변경
        if scores.get("overall", 1.0) < 0.5:
            # TTS 엔진 자체를 변경하는 것을 추천
            current_engine = current_params.get("engine", "neural")
            
            if current_engine == "neural":
                recommendations["engine"] = "google"
                changes["engine"] = "전반적인 유사도가 낮아 다른 TTS 엔진으로 변경 권장"
            elif current_engine == "google":
                recommendations["engine"] = "amazon"
                changes["engine"] = "전반적인 유사도가 낮아 다른 TTS 엔진으로 변경 권장"
            else:
                recommendations["engine"] = "neural"
                changes["engine"] = "전반적인 유사도가 낮아 다른 TTS 엔진으로 변경 권장"
        
        # 변경사항이 없으면 작은 변화라도 추천
        if not recommendations and scores.get("overall", 1.0) < 0.85:
            current_rate = current_params.get("speaking_rate", 1.0)
            recommendations["speaking_rate"] = current_rate * 0.95  # 5% 감소
            changes["speaking_rate"] = "미세 조정: 말하는 속도 소폭 감소"
        
        return {
            "parameters": {**current_params, **recommendations},
            "changes": changes,
            "expected_improvement": min(0.15, 1.0 - scores.get("overall", 0.5))
        }
    
    def _analyze_pause_similarity(
        self,
        src_y: np.ndarray,
        src_sr: int,
        src_segments: List[Dict[str, Any]],
        tgt_y: np.ndarray,
        tgt_sr: int,
        tgt_segments: List[Dict[str, Any]]
    ) -> float:
        """
        원본과 합성 오디오 간의 일시 정지(pause) 패턴 유사도를 분석합니다.
        
        Returns:
            일시 정지 패턴 유사도 점수 (0.0 ~ 1.0)
        """
        # 일시 정지 위치 추출 (무음 구간)
        src_pauses = self._extract_pauses(src_y, src_sr)
        tgt_pauses = self._extract_pauses(tgt_y, tgt_sr)
        
        # 각 일시 정지의 상대적 위치 계산 (전체 길이에 대한 비율)
        src_relative_pauses = [p / len(src_y) for p in src_pauses]
        tgt_relative_pauses = [p / len(tgt_y) for p in tgt_pauses]
        
        # 유사한 수의 일시 정지가 있는지 확인
        count_similarity = min(len(src_relative_pauses), len(tgt_relative_pauses)) / max(1, max(len(src_relative_pauses), len(tgt_relative_pauses)))
        
        # 일시 정지 위치의 유사도 계산
        position_similarity = 0.0
        if src_relative_pauses and tgt_relative_pauses:
            # 가장 가까운 일시 정지 쌍 찾기
            matched_distances = []
            
            for src_pos in src_relative_pauses:
                min_distance = min([abs(src_pos - tgt_pos) for tgt_pos in tgt_relative_pauses])
                matched_distances.append(min_distance)
            
            # 평균 거리를 유사도로 변환 (거리가 작을수록 유사도가 높음)
            avg_distance = sum(matched_distances) / len(matched_distances)
            position_similarity = max(0.0, 1.0 - avg_distance * 10)  # 거리에 가중치를 두어 변환
        
        # 최종 유사도: 개수 유사도와 위치 유사도의 가중 평균
        similarity = 0.4 * count_similarity + 0.6 * position_similarity
        
        logger.debug(f"Pause similarity: {similarity:.4f} (count: {count_similarity:.4f}, position: {position_similarity:.4f})")
        return similarity
    
    def _extract_pauses(self, y: np.ndarray, sr: int) -> List[int]:
        """
        오디오에서 일시 정지(무음 구간) 위치를 추출합니다.
        
        Returns:
            일시 정지 위치의 샘플 인덱스 목록
        """
        # 에너지(RMS) 계산
        frame_length = int(0.025 * sr)  # 25ms 프레임
        hop_length = int(0.010 * sr)    # 10ms 홉
        
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # 무음 임계값 (RMS 에너지의 10%)
        threshold = np.mean(rms) * 0.1
        
        # 무음 구간 찾기
        is_silent = rms < threshold
        
        # 연속 무음 구간의 중간점 찾기
        pause_positions = []
        in_pause = False
        pause_start = 0
        
        for i, silent in enumerate(is_silent):
            if silent and not in_pause:
                # 무음 구간 시작
                in_pause = True
                pause_start = i
            elif not silent and in_pause:
                # 무음 구간 종료
                pause_end = i
                if pause_end - pause_start > 3:  # 최소 3프레임(~30ms) 이상의 무음만 고려
                    pause_center = (pause_start + pause_end) // 2
                    # 샘플 인덱스로 변환
                    pause_positions.append(pause_center * hop_length)
                in_pause = False
        
        return pause_positions
    
    def _analyze_pitch_similarity(
        self,
        src_y: np.ndarray,
        src_sr: int,
        src_segments: List[Dict[str, Any]],
        tgt_y: np.ndarray,
        tgt_sr: int,
        tgt_segments: List[Dict[str, Any]]
    ) -> float:
        """
        원본과 합성 오디오 간의 음높이(pitch) 패턴 유사도를 분석합니다.
        
        Returns:
            음높이 패턴 유사도 점수 (0.0 ~ 1.0)
        """
        # 피치 추출
        src_pitches, src_magnitudes = librosa.piptrack(y=src_y, sr=src_sr)
        tgt_pitches, tgt_magnitudes = librosa.piptrack(y=tgt_y, sr=tgt_sr)
        
        # 각 프레임의 주요 피치 추출
        src_pitch = []
        for t in range(src_pitches.shape[1]):
            index = np.argmax(src_magnitudes[:, t])
            src_pitch.append(src_pitches[index, t])
        
        tgt_pitch = []
        for t in range(tgt_pitches.shape[1]):
            index = np.argmax(tgt_magnitudes[:, t])
            tgt_pitch.append(tgt_pitches[index, t])
        
        # 피치 정보가 너무 적으면 (무음 또는 오류) 기본값 반환
        if len(src_pitch) < 10 or len(tgt_pitch) < 10:
            return 0.75  # 중간 값보다 약간 높게 설정
        
        # 0이 아닌 피치만 선택 (유효한 피치)
        src_pitch = np.array([p for p in src_pitch if p > 0])
        tgt_pitch = np.array([p for p in tgt_pitch if p > 0])
        
        if len(src_pitch) == 0 or len(tgt_pitch) == 0:
            return 0.75
        
        # 피치 정규화 (서로 다른 화자의 목소리 높이 차이 보정)
        src_pitch_norm = (src_pitch - np.mean(src_pitch)) / np.std(src_pitch)
        tgt_pitch_norm = (tgt_pitch - np.mean(tgt_pitch)) / np.std(tgt_pitch)
        
        # 피치 길이 맞추기 (짧은 쪽에 맞춤)
        min_len = min(len(src_pitch_norm), len(tgt_pitch_norm))
        src_pitch_resized = librosa.util.fix_length(src_pitch_norm, size=min_len)
        tgt_pitch_resized = librosa.util.fix_length(tgt_pitch_norm, size=min_len)
        
        # 음높이 변화(contour) 유사도 계산
        correlation = np.corrcoef(src_pitch_resized, tgt_pitch_resized)[0, 1]
        
        # NaN 처리
        if np.isnan(correlation):
            correlation = 0.5  # 중간값 할당
        
        # 유사도 범위를 0-1로 조정
        similarity = (correlation + 1) / 2
        
        logger.debug(f"Pitch similarity: {similarity:.4f}")
        return similarity
    
    def _analyze_energy_similarity(
        self,
        src_y: np.ndarray,
        src_sr: int,
        src_segments: List[Dict[str, Any]],
        tgt_y: np.ndarray,
        tgt_sr: int,
        tgt_segments: List[Dict[str, Any]]
    ) -> float:
        """
        원본과 합성 오디오 간의 에너지(loudness) 패턴 유사도를 분석합니다.
        
        Returns:
            에너지 패턴 유사도 점수 (0.0 ~ 1.0)
        """
        # 에너지(RMS) 계산
        src_rms = librosa.feature.rms(y=src_y)[0]
        tgt_rms = librosa.feature.rms(y=tgt_y)[0]
        
        # 에너지 정규화
        src_rms_norm = src_rms / np.max(src_rms)
        tgt_rms_norm = tgt_rms / np.max(tgt_rms)
        
        # 길이 맞추기
        min_len = min(len(src_rms_norm), len(tgt_rms_norm))
        src_rms_resized = librosa.util.fix_length(src_rms_norm, size=min_len)
        tgt_rms_resized = librosa.util.fix_length(tgt_rms_norm, size=min_len)
        
        # 에너지 패턴 유사도 계산
        correlation = np.corrcoef(src_rms_resized, tgt_rms_resized)[0, 1]
        
        # NaN 처리
        if np.isnan(correlation):
            correlation = 0.5
        
        # 유사도 범위를 0-1로 조정
        similarity = (correlation + 1) / 2
        
        logger.debug(f"Energy similarity: {similarity:.4f}")
        return similarity
    
    def _analyze_rhythm_similarity(
        self,
        src_y: np.ndarray,
        src_sr: int,
        src_segments: List[Dict[str, Any]],
        tgt_y: np.ndarray,
        tgt_sr: int,
        tgt_segments: List[Dict[str, Any]]
    ) -> float:
        """
        원본과 합성 오디오 간의 리듬 패턴 유사도를 분석합니다.
        
        Returns:
            리듬 패턴 유사도 점수 (0.0 ~ 1.0)
        """
        try:
            # 템포와 비트 추출 - beat_track 사용을 피함
            src_onset_env = librosa.onset.onset_strength(y=src_y, sr=src_sr)
            tgt_onset_env = librosa.onset.onset_strength(y=tgt_y, sr=tgt_sr)
            
            # onset_detect 함수로 대체
            src_onsets = librosa.onset.onset_detect(onset_envelope=src_onset_env, sr=src_sr)
            tgt_onsets = librosa.onset.onset_detect(onset_envelope=tgt_onset_env, sr=tgt_sr)
            
            # 프레임 인덱스를 시간으로 변환
            src_times = librosa.frames_to_time(src_onsets, sr=src_sr)
            tgt_times = librosa.frames_to_time(tgt_onsets, sr=tgt_sr)
            
            # 비트 수 유사도
            beat_count_similarity = min(len(src_times), len(tgt_times)) / max(1, max(len(src_times), len(tgt_times)))
            
            # 간격 패턴 유사도 계산
            src_intervals = np.diff(src_times) if len(src_times) > 1 else [0]
            tgt_intervals = np.diff(tgt_times) if len(tgt_times) > 1 else [0]
            
            # 길이 맞추기
            min_intervals = min(len(src_intervals), len(tgt_intervals))
            if min_intervals > 1:
                src_intervals = src_intervals[:min_intervals]
                tgt_intervals = tgt_intervals[:min_intervals]
                
                # 정규화 (평균으로 나누기)
                src_intervals_norm = src_intervals / np.mean(src_intervals)
                tgt_intervals_norm = tgt_intervals / np.mean(tgt_intervals)
                
                # 간격 패턴 상관계수
                interval_correlation = np.corrcoef(src_intervals_norm, tgt_intervals_norm)[0, 1]
                
                # NaN 처리
                if np.isnan(interval_correlation):
                    interval_correlation = 0.5
                    
                interval_similarity = (interval_correlation + 1) / 2
            else:
                interval_similarity = 0.5  # 충분한 비트가 없는 경우 중간값
            
            # 최종 리듬 유사도
            rhythm_similarity = 0.4 * beat_count_similarity + 0.6 * interval_similarity
            
            logger.debug(f"Rhythm similarity: {rhythm_similarity:.4f} (beat_count: {beat_count_similarity:.4f}, interval: {interval_similarity:.4f})")
            return rhythm_similarity
        except Exception as e:
            logger.error(f"Error in rhythm analysis: {e}")
            # 오류 발생 시 기본값 반환
            return 0.7  # 약간 높은 기본값 제공
    
    def _is_vowel(self, phone: str) -> bool:
        """
        주어진 음소가 모음인지 판별합니다.
        
        Args:
            phone: 음소 문자열
            
        Returns:
            모음 여부
        """
        # 영어 모음 목록 (IPA)
        english_vowels = ['a', 'e', 'i', 'o', 'u', 'æ', 'ɑ', 'ɒ', 'ɔ', 'ɛ', 'ə', 'ɪ', 'ʊ', 'ʌ']
        
        # 한국어 모음 목록
        korean_vowels = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        
        # 추가 모음 확인 패턴 (다양한 음소 표기 시스템 고려)
        vowel_patterns = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
        
        # 음소가 모음 목록에 있거나 패턴을 포함하는지 확인
        is_vowel = (
            phone in english_vowels or
            phone in korean_vowels or
            any(pattern in phone for pattern in vowel_patterns)
        )
        
        return is_vowel
    
    def _visualize_audio_comparison(
        self,
        src_y: np.ndarray,
        src_sr: int,
        src_segments: List[Dict[str, Any]],
        tgt_y: np.ndarray,
        tgt_sr: int,
        tgt_segments: List[Dict[str, Any]],
        output_path: Path
    ) -> None:
        """
        원본과 합성 오디오의 비교 시각화를 생성합니다.
        
        Args:
            src_y: 원본 오디오 파형
            src_sr: 원본 샘플레이트
            src_segments: 원본 세그먼트 정보
            tgt_y: 합성 오디오 파형
            tgt_sr: 합성 샘플레이트
            tgt_segments: 합성 세그먼트 정보
            output_path: 시각화 저장 경로
        """
        plt.figure(figsize=(12, 8))
        
        # 파형 비교
        plt.subplot(3, 1, 1)
        plt.title("Waveform Comparison")
        
        # 파형 시간 정규화
        src_times = np.arange(len(src_y)) / src_sr
        tgt_times = np.arange(len(tgt_y)) / tgt_sr
        
        max_time = max(src_times[-1], tgt_times[-1])
        
        # 원본 파형
        plt.plot(src_times, src_y, 'b-', alpha=0.7, label='Source')
        
        # 합성 파형
        plt.plot(tgt_times, tgt_y, 'r-', alpha=0.7, label='Target')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        
        # 스펙트로그램 비교
        plt.subplot(3, 1, 2)
        plt.title("Source Spectrogram")
        D_src = librosa.amplitude_to_db(np.abs(librosa.stft(src_y)), ref=np.max)
        librosa.display.specshow(D_src, sr=src_sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        
        plt.subplot(3, 1, 3)
        plt.title("Target Spectrogram")
        D_tgt = librosa.amplitude_to_db(np.abs(librosa.stft(tgt_y)), ref=np.max)
        librosa.display.specshow(D_tgt, sr=tgt_sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Audio comparison visualization saved to {output_path}")
    
    def _visualize_similarity_scores(
        self,
        scores: Dict[str, float],
        output_path: Path
    ) -> None:
        """
        유사도 점수를 시각화합니다.
        
        Args:
            scores: 유사도 점수 딕셔너리
            output_path: 시각화 저장 경로
        """
        # 'overall' 항목 제외
        plot_scores = {k: v for k, v in scores.items() if k != 'overall'}
        
        # 점수가 없으면 반환
        if not plot_scores:
            return
        
        plt.figure(figsize=(10, 6))
        
        # 막대 그래프
        categories = list(plot_scores.keys())
        values = list(plot_scores.values())
        
        # 색상 설정 (점수에 따라)
        colors = ['#ff9999' if v < 0.6 else '#66b3ff' if v < 0.8 else '#99ff99' for v in values]
        
        bars = plt.bar(categories, values, color=colors)
        
        # 전체 점수 표시
        plt.axhline(y=scores.get('overall', 0), color='r', linestyle='-', label=f"Overall: {scores.get('overall', 0):.2f}")
        
        # 그래프 설정
        plt.ylim(0, 1.0)
        plt.title('Prosody Similarity Scores')
        plt.ylabel('Similarity (0-1)')
        plt.legend()
        
        # 막대 위에 값 표시
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height * 1.01,
                f'{height:.2f}',
                ha='center',
                va='bottom'
            )
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Similarity scores visualization saved to {output_path}")