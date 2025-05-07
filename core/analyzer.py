# core/analyzer.py
"""
음성 유사도 분석 모듈
음성 유사도를 분석하는 핵심 모듈입니다:
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import librosa
import os
import matplotlib
from config import EVAL_CONFIG

logger = logging.getLogger(__name__)
# IPA 문자(발음 기호) 지원을 위해 DejaVu Sans 폰트 사용
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic']  # 폰트 설정
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
class ProsodyAnalyzer:
    """
    원본 음성과 합성 음성 간의 프로소디 유사도를 분석하는 클래스
    """
    
    def __init__(self, feature_weights: Optional[Dict[str, float]] = None):
        """
        프로소디 분석기 초기화
        
        Args:
            feature_weights: 특성별 가중치 (기본값: config.py에서 설정)
        """
        self.feature_weights = feature_weights or EVAL_CONFIG["feature_weights"]
        logger.info("프로소디 분석기 초기화")
    
    def analyze(
        self,
        src_audio: np.ndarray,
        src_sr: int,
        src_segments: List[Dict[str, Any]],
        tgt_audio: np.ndarray,
        tgt_sr: int,
        tgt_segments: List[Dict[str, Any]],
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        원본 음성과 합성 음성의 프로소디 유사도 분석
        
        Args:
            src_audio: 원본 오디오 데이터
            src_sr: 원본 샘플링 레이트
            src_segments: 원본 세그먼트 정보
            tgt_audio: 합성 오디오 데이터
            tgt_sr: 합성 샘플링 레이트
            tgt_segments: 합성 세그먼트 정보
            output_dir: 결과 저장 디렉토리
            
        Returns:
            분석 결과 및 유사도 점수를 포함한 딕셔너리
        """
        if output_dir:
            output_dir.mkdir(exist_ok=True, parents=True)
        
        # 유사도 점수 계산
        scores = {}
        
        # 1. 휴지(일시정지) 패턴 유사도
        pause_similarity = self._analyze_pause_similarity(
            src_audio, src_sr, src_segments,
            tgt_audio, tgt_sr, tgt_segments
        )
        scores["pause_similarity"] = pause_similarity
        
        # 2. 음높이(피치) 패턴 유사도
        pitch_similarity = self._analyze_pitch_similarity(
            src_audio, src_sr, src_segments,
            tgt_audio, tgt_sr, tgt_segments
        )
        scores["pitch_similarity"] = pitch_similarity
        
        # 3. 에너지(강세) 패턴 유사도
        energy_similarity = self._analyze_energy_similarity(
            src_audio, src_sr, src_segments,
            tgt_audio, tgt_sr, tgt_segments
        )
        scores["energy_similarity"] = energy_similarity
        
        # 4. 리듬 패턴 유사도
        rhythm_similarity = self._analyze_rhythm_similarity(
            src_audio, src_sr, src_segments,
            tgt_audio, tgt_sr, tgt_segments
        )
        scores["rhythm_similarity"] = rhythm_similarity
        
        # 5. 모음 길이 패턴 유사도 (가능한 경우)
        if all('phones' in segment for segment in src_segments + tgt_segments):
            vowel_similarity = self._analyze_vowel_similarity(
                src_segments, tgt_segments
            )
            scores["vowel_similarity"] = vowel_similarity
        
        # 종합 점수 계산 (가중 평균)
        weights = self.feature_weights.copy()
        
        if "vowel_similarity" not in scores and "vowel" in weights:
            # 모음 유사도가 없는 경우 가중치 재분배
            vowel_weight = weights.pop("vowel")
            weight_sum = sum(weights.values())
            weights = {k: v / weight_sum * (weight_sum + vowel_weight) for k, v in weights.items()}
        
        # 가중 평균 계산
        overall_similarity = 0
        weight_sum = 0
        
        for feature, weight_key in [
            ("pause_similarity", "pause"),
            ("pitch_similarity", "pitch"),
            ("energy_similarity", "energy"),
            ("rhythm_similarity", "rhythm"),
            ("vowel_similarity", "vowel")
        ]:
            if feature in scores and weight_key in weights:
                overall_similarity += scores[feature] * weights[weight_key]
                weight_sum += weights[weight_key]
        
        # 가중치 합이 0이 아닌 경우에만 정규화
        if weight_sum > 0:
            overall_similarity /= weight_sum
        
        scores["overall"] = overall_similarity
        
        # 결과 로깅
        logger.info(f"프로소디 분석 완료: 전체 유사도 {overall_similarity:.4f}")
        for key, value in scores.items():
            if key != "overall":
                logger.info(f"  - {key}: {value:.4f}")
        
        # 결과를 파일로 저장 및 시각화
        if output_dir:
            self._save_results(scores, output_dir / "similarity_scores.json")
            self._visualize_similarity_scores(scores, output_dir / "similarity_scores.png")
            self._visualize_audio_comparison(
                src_audio, src_sr, src_segments,
                tgt_audio, tgt_sr, tgt_segments,
                output_dir / "audio_comparison.png"
            )
        
        return scores
    
    def _analyze_pause_similarity(
        self,
        src_audio: np.ndarray,
        src_sr: int,
        src_segments: List[Dict[str, Any]],
        tgt_audio: np.ndarray,
        tgt_sr: int,
        tgt_segments: List[Dict[str, Any]]
    ) -> float:
        """
        원본과 합성 오디오 간의 휴지(일시정지) 패턴 유사도 분석
        
        Returns:
            휴지 패턴 유사도 점수 (0.0 ~ 1.0)
        """
        # 일시정지 위치 추출 (무음 구간)
        src_pauses = self._extract_pauses(src_audio, src_sr)
        tgt_pauses = self._extract_pauses(tgt_audio, tgt_sr)
        
        # 각 일시정지의 상대적 위치 계산 (전체 길이에 대한 비율)
        src_relative_pauses = [p / len(src_audio) for p in src_pauses]
        tgt_relative_pauses = [p / len(tgt_audio) for p in tgt_pauses]
        
        # 유사한 수의 일시정지가 있는지 확인
        count_similarity = min(len(src_relative_pauses), len(tgt_relative_pauses)) / max(1, max(len(src_relative_pauses), len(tgt_relative_pauses)))
        
        # 일시정지 위치의 유사도 계산
        position_similarity = 0.0
        if src_relative_pauses and tgt_relative_pauses:
            # 가장 가까운 일시정지 쌍 찾기
            matched_distances = []
            
            for src_pos in src_relative_pauses:
                min_distance = min([abs(src_pos - tgt_pos) for tgt_pos in tgt_relative_pauses])
                matched_distances.append(min_distance)
            
            # 평균 거리를 유사도로 변환 (거리가 작을수록 유사도가 높음)
            avg_distance = sum(matched_distances) / len(matched_distances)
            position_similarity = max(0.0, 1.0 - avg_distance * 10)  # 거리에 가중치를 두어 변환
        
        # 최종 유사도: 개수 유사도와 위치 유사도의 가중 평균
        similarity = 0.4 * count_similarity + 0.6 * position_similarity
        
        logger.debug(f"휴지 유사도: {similarity:.4f} (개수: {count_similarity:.4f}, 위치: {position_similarity:.4f})")
        return similarity
    
    def _extract_pauses(self, audio: np.ndarray, sr: int) -> List[int]:
        """
        오디오에서 일시정지(무음 구간) 위치 추출
        
        Returns:
            일시정지 위치의 샘플 인덱스 목록
        """
        # 에너지(RMS) 계산
        frame_length = int(0.025 * sr)  # 25ms 프레임
        hop_length = int(0.010 * sr)    # 10ms 홉
        
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
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
        src_audio: np.ndarray,
        src_sr: int,
        src_segments: List[Dict[str, Any]],
        tgt_audio: np.ndarray,
        tgt_sr: int,
        tgt_segments: List[Dict[str, Any]]
    ) -> float:
        """
        원본과 합성 오디오 간의 음높이(피치) 패턴 유사도 분석
        
        Returns:
            음높이 패턴 유사도 점수 (0.0 ~ 1.0)
        """
        # 피치 추출
        src_pitches, src_magnitudes = librosa.piptrack(y=src_audio, sr=src_sr)
        tgt_pitches, tgt_magnitudes = librosa.piptrack(y=tgt_audio, sr=tgt_sr)
        
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
        
        logger.debug(f"피치 유사도: {similarity:.4f}")
        return similarity
    
    def _analyze_energy_similarity(
        self,
        src_audio: np.ndarray,
        src_sr: int,
        src_segments: List[Dict[str, Any]],
        tgt_audio: np.ndarray,
        tgt_sr: int,
        tgt_segments: List[Dict[str, Any]]
    ) -> float:
        """
        원본과 합성 오디오 간의 에너지(강세) 패턴 유사도 분석
        
        Returns:
            에너지 패턴 유사도 점수 (0.0 ~ 1.0)
        """
        # 에너지(RMS) 계산
        src_rms = librosa.feature.rms(y=src_audio)[0]
        tgt_rms = librosa.feature.rms(y=tgt_audio)[0]
        
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
        
        logger.debug(f"에너지 유사도: {similarity:.4f}")
        return similarity
    
    def _analyze_rhythm_similarity(
        self,
        src_audio: np.ndarray,
        src_sr: int,
        src_segments: List[Dict[str, Any]],
        tgt_audio: np.ndarray,
        tgt_sr: int,
        tgt_segments: List[Dict[str, Any]]
    ) -> float:
        """
        원본과 합성 오디오 간의 리듬 패턴 유사도 분석
        
        Returns:
            리듬 패턴 유사도 점수 (0.0 ~ 1.0)
        """
        try:
            # 템포와 비트 추출
            src_onset_env = librosa.onset.onset_strength(y=src_audio, sr=src_sr)
            tgt_onset_env = librosa.onset.onset_strength(y=tgt_audio, sr=tgt_sr)
            
            # onset_detect 함수로 비트 위치 검출
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
            
            logger.debug(f"리듬 유사도: {rhythm_similarity:.4f} (비트 개수: {beat_count_similarity:.4f}, 간격: {interval_similarity:.4f})")
            return rhythm_similarity
        except Exception as e:
            logger.error(f"리듬 분석 중 오류: {e}")
            # 오류 발생 시 기본값 반환
            return 0.7  # 약간 높은 기본값 제공
    
    def _analyze_vowel_similarity(
        self,
        src_segments: List[Dict[str, Any]],
        tgt_segments: List[Dict[str, Any]]
    ) -> float:
        """
        원본과 합성 오디오 간의 모음 길이 패턴 유사도 분석
        
        Returns:
            모음 길이 패턴 유사도 점수 (0.0 ~ 1.0)
        """
        # 원본과 타겟에서 모음 추출
        src_vowels = []
        for segment in src_segments:
            if 'phones' not in segment:
                continue
            for phone in segment['phones']:
                if phone.get('is_vowel', False):
                    src_vowels.append(phone)
        
        tgt_vowels = []
        for segment in tgt_segments:
            if 'phones' not in segment:
                continue
            for phone in segment['phones']:
                if phone.get('is_vowel', False):
                    tgt_vowels.append(phone)
        
        # 모음 개수가 너무 적으면 기본값 반환
        if len(src_vowels) < 3 or len(tgt_vowels) < 3:
            return 0.7
        
        # 모음 개수 유사도
        count_similarity = min(len(src_vowels), len(tgt_vowels)) / max(len(src_vowels), len(tgt_vowels))
        
        # 모음 길이 추출
        src_durations = [v['duration'] for v in src_vowels]
        tgt_durations = [v['duration'] for v in tgt_vowels]
        
        # 길이 정규화 (평균으로 나누기)
        src_norm_durations = np.array(src_durations) / np.mean(src_durations)
        tgt_norm_durations = np.array(tgt_durations) / np.mean(tgt_durations)
        
        # 길이 맞추기 (짧은 쪽에 맞춤)
        min_len = min(len(src_norm_durations), len(tgt_norm_durations))
        src_resized = src_norm_durations[:min_len]
        tgt_resized = tgt_norm_durations[:min_len]
        
        # 모음 길이 패턴 유사도 계산
        try:
            correlation = np.corrcoef(src_resized, tgt_resized)[0, 1]
            
            # NaN 처리
            if np.isnan(correlation):
                duration_similarity = 0.5
            else:
                duration_similarity = (correlation + 1) / 2
        except Exception:
            duration_similarity = 0.5
        
        # 최종 모음 유사도
        vowel_similarity = 0.3 * count_similarity + 0.7 * duration_similarity
        
        logger.debug(f"모음 유사도: {vowel_similarity:.4f} (개수: {count_similarity:.4f}, 길이: {duration_similarity:.4f})")
        return vowel_similarity
    
    def _save_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """
        분석 결과를 JSON 파일로 저장
        
        Args:
            results: 저장할 결과 딕셔너리
            output_path: 저장 경로
        """
        import json
        
        # 결과 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"분석 결과가 {output_path}에 저장되었습니다.")
    
    def _visualize_similarity_scores(
        self,
        scores: Dict[str, float],
        output_path: Path
    ) -> None:
        """
        유사도 점수를 시각화
        
        Args:
            scores: 유사도 점수 딕셔너리
            output_path: 저장 경로
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
        plt.axhline(y=scores.get('overall', 0), color='r', linestyle='-', label=f"전체: {scores.get('overall', 0):.2f}")
        
        # 그래프 설정
        plt.ylim(0, 1.0)
        plt.title('프로소디 유사도 점수')
        plt.ylabel('유사도 (0-1)')
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
        
        logger.info(f"유사도 점수 시각화가 {output_path}에 저장되었습니다.")
    
    def _visualize_audio_comparison(
        self,
        src_audio: np.ndarray,
        src_sr: int,
        src_segments: List[Dict[str, Any]],
        tgt_audio: np.ndarray,
        tgt_sr: int,
        tgt_segments: List[Dict[str, Any]],
        output_path: Path
    ) -> None:
        """
        원본과 합성 오디오의 비교 시각화 생성
        
        Args:
            src_audio: 원본 오디오 파형
            src_sr: 원본 샘플레이트
            src_segments: 원본 세그먼트 정보
            tgt_audio: 합성 오디오 파형
            tgt_sr: 합성 샘플레이트
            tgt_segments: 합성 세그먼트 정보
            output_path: 시각화 저장 경로
        """
        plt.figure(figsize=(12, 8))
        
        # 파형 비교
        plt.subplot(3, 1, 1)
        plt.title("파형 비교")
        
        # 파형 시간 정규화
        src_times = np.arange(len(src_audio)) / src_sr
        tgt_times = np.arange(len(tgt_audio)) / tgt_sr
        
        max_time = max(src_times[-1], tgt_times[-1])
        
        # 원본 파형
        plt.plot(src_times, src_audio, 'b-', alpha=0.7, label='원본')
        
        # 합성 파형
        plt.plot(tgt_times, tgt_audio, 'r-', alpha=0.7, label='합성')
        
        plt.xlabel('시간 (초)')
        plt.ylabel('진폭')
        plt.legend()
        plt.grid(True)
        
        # 스펙트로그램 비교
        plt.subplot(3, 1, 2)
        plt.title("원본 스펙트로그램")
        D_src = librosa.amplitude_to_db(np.abs(librosa.stft(src_audio)), ref=np.max)
        librosa.display.specshow(D_src, sr=src_sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        
        plt.subplot(3, 1, 3)
        plt.title("합성 스펙트로그램")
        D_tgt = librosa.amplitude_to_db(np.abs(librosa.stft(tgt_audio)), ref=np.max)
        librosa.display.specshow(D_tgt, sr=tgt_sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"오디오 비교 시각화가 {output_path}에 저장되었습니다.")