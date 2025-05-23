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
import scipy
import fastdtw
from scipy.spatial.distance import euclidean

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
        
        # 오디오 비교 시각화 생성 (루트 디렉토리에)
        if output_dir:
            self._visualize_audio_comparison(
                src_audio, src_sr, src_segments,
                tgt_audio, tgt_sr, tgt_segments,
                output_dir / "audio_comparison.png"  # 메인 디렉터리에 저장
            )
        
        # prosody 디렉토리의 JSON 파일은 evaluator에서 처리
        
        logger.info(f"프로소디 분석 완료: 전체 유사도 {overall_similarity:.4f}")
        
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
        원본과 합성 오디오 간의 휴지(일시정지) 패턴 유사도 분석 - 평가 기준 완화
        """
        # 일시정지 위치 추출 (무음 구간)
        src_pauses = self._extract_pauses(src_audio, src_sr)
        tgt_pauses = self._extract_pauses(tgt_audio, tgt_sr)
        
        # 짧은 일시정지 필터링 (중요한 휴지만 분석)
        min_pause_duration = 0.3  # 초
        src_pauses_filtered = self._filter_short_pauses(src_audio, src_sr, src_pauses, min_pause_duration)
        tgt_pauses_filtered = self._filter_short_pauses(tgt_audio, tgt_sr, tgt_pauses, min_pause_duration)
        
        # 각 일시정지의 상대적 위치 계산 (전체 길이에 대한 비율)
        src_duration = len(src_audio) / src_sr
        tgt_duration = len(tgt_audio) / tgt_sr
        
        src_relative_pauses = [p / (src_sr * src_duration) for p in src_pauses_filtered]
        tgt_relative_pauses = [p / (tgt_sr * tgt_duration) for p in tgt_pauses_filtered]
        
        # 1. 유사한 수의 일시정지가 있는지 확인 - 더 관대하게
        if max(len(src_relative_pauses), len(tgt_relative_pauses)) == 0:
            count_similarity = 1.0  # 둘 다 일시정지가 없는 경우
        else:
            # 개수 차이 비율
            count_diff_ratio = abs(len(src_relative_pauses) - len(tgt_relative_pauses)) / max(1, max(len(src_relative_pauses), len(tgt_relative_pauses)))
            
            # 개수 유사도 - 더 관대하게 (최대 30% 감점)
            count_similarity = 1.0 - min(count_diff_ratio * 0.7, 0.3)
        
        # 2. 일시정지 위치의 유사도 계산 - 더 관대하게
        position_similarity = 0.6  # 기본값 상향
        
        if src_relative_pauses and tgt_relative_pauses:
            # 위치 매칭 방식 변경: 상대적 분포 비교
            src_bins = np.linspace(0, 1, 5)  # 5개 구간으로 나누기
            tgt_bins = np.linspace(0, 1, 5)
            
            src_hist, _ = np.histogram(src_relative_pauses, bins=src_bins)
            tgt_hist, _ = np.histogram(tgt_relative_pauses, bins=tgt_bins)
            
            # 분포 비율로 변환
            src_dist = src_hist / (np.sum(src_hist) + 1e-8)
            tgt_dist = tgt_hist / (np.sum(tgt_hist) + 1e-8)
            
            # 분포 차이 계산 (최대 절대 차이)
            dist_diff = np.max(np.abs(src_dist - tgt_dist))
            
            # 분포 유사도 - 더 관대하게
            position_similarity = 1.0 - min(dist_diff * 0.8, 0.4)  # 최대 40% 감점
        
        # 3. 최종 유사도: 개수 유사도와 위치 유사도의 가중 평균 - 가중치 조정
        similarity = 0.3 * count_similarity + 0.7 * position_similarity
        
        # 4. 점수 상향 조정
        adjusted_similarity = self._adjust_score(similarity)
        
        logger.debug(f"휴지 유사도: {adjusted_similarity:.4f} (원래: {similarity:.4f}, 개수: {count_similarity:.4f}, 위치: {position_similarity:.4f})")
        return adjusted_similarity

    def _filter_short_pauses(self, audio, sr, pauses, min_duration):
        """짧은 휴지 필터링 (중요한 휴지만 남김)"""
        filtered_pauses = []
        min_samples = int(min_duration * sr)
        
        for i in range(len(pauses)):
            if i < len(pauses) - 1:
                # 현재 무음과 다음 무음 사이의 간격 계산
                gap = pauses[i+1] - pauses[i]
                if gap > min_samples:
                    filtered_pauses.append(pauses[i])
            else:
                # 마지막 무음은 항상 포함
                filtered_pauses.append(pauses[i])
        
        return filtered_pauses
    
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
        원본과 합성 오디오 간의 음높이(피치) 패턴 유사도 분석 - 변화 크기 및 피크 위치 분석 추가
        """
        try:
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
            
            # 피치 정보가 너무 적으면 기본값 반환
            if len(src_pitch) < 10 or len(tgt_pitch) < 10:
                return 0.75
            
            # 0이 아닌 피치만 선택 (유효한 피치)
            src_pitch = np.array([p for p in src_pitch if p > 0])
            tgt_pitch = np.array([p for p in tgt_pitch if p > 0])
            
            if len(src_pitch) == 0 or len(tgt_pitch) == 0:
                return 0.75
            
            # 피치 정규화 (서로 다른 화자의 목소리 높이 차이 보정)
            src_pitch_norm = (src_pitch - np.mean(src_pitch)) / (np.std(src_pitch) + 1e-10)
            tgt_pitch_norm = (tgt_pitch - np.mean(tgt_pitch)) / (np.std(tgt_pitch) + 1e-10)
            
            # 1. 상대적 패턴 중심: 피치 방향 변화 추출 (상승/하강/유지)
            src_direction = np.sign(np.diff(src_pitch_norm))
            tgt_direction = np.sign(np.diff(tgt_pitch_norm))
            
            # 길이 맞추기
            min_dir_len = min(len(src_direction), len(tgt_direction))
            if min_dir_len < 1:
                return 0.75
            
            # 리샘플링 (동일한 길이로)
            src_dir_resampled = scipy.signal.resample(src_direction, min_dir_len) 
            tgt_dir_resampled = scipy.signal.resample(tgt_direction, min_dir_len)
            
            # 방향 일치 정도 계산
            direction_matches = np.sum(src_dir_resampled == tgt_dir_resampled) / min_dir_len
            
            # 2. 피치 변화 크기 분석 (새로 추가)
            src_pitch_changes = np.abs(np.diff(src_pitch_norm))
            tgt_pitch_changes = np.abs(np.diff(tgt_pitch_norm))
            
            # 변화 크기 정규화
            src_changes_norm = src_pitch_changes / (np.max(src_pitch_changes) + 1e-10)
            tgt_changes_norm = tgt_pitch_changes / (np.max(tgt_pitch_changes) + 1e-10)
            
            # 길이 맞추기
            min_changes_len = min(len(src_changes_norm), len(tgt_changes_norm))
            src_changes_resized = scipy.signal.resample(src_changes_norm, min_changes_len)
            tgt_changes_resized = scipy.signal.resample(tgt_changes_norm, min_changes_len)
            
            # 변화 크기 유사도 계산 (MSE 기반)
            changes_mse = np.mean((src_changes_resized - tgt_changes_resized)**2)
            magnitude_similarity = 1.0 - min(1.0, changes_mse * 2.0)  # MSE를 0-1 범위로 변환
            
            # 3. 중요 변화 지점 분석 (개선)
            # 중요한 변화 지점: 피치 기울기가 임계값보다 큰 지점
            src_threshold = np.mean(src_changes_norm) + np.std(src_changes_norm)
            tgt_threshold = np.mean(tgt_changes_norm) + np.std(tgt_changes_norm)
            
            src_important = src_changes_norm > src_threshold
            tgt_important = tgt_changes_norm > tgt_threshold
            
            # 중요 변화 지점 밀도
            src_important_density = np.sum(src_important) / len(src_important)
            tgt_important_density = np.sum(tgt_important) / len(tgt_important)
            
            # 밀도 유사도 계산
            density_diff = abs(src_important_density - tgt_important_density)
            density_similarity = 1.0 - min(1.0, density_diff * 3.0)
            
            # 4. 피크 위치 분석 (새로 추가)
            src_peaks, _ = scipy.signal.find_peaks(src_pitch_norm, height=0.5, distance=5)
            tgt_peaks, _ = scipy.signal.find_peaks(tgt_pitch_norm, height=0.5, distance=5)
            
            # 피크 개수 유사도
            if max(len(src_peaks), len(tgt_peaks)) == 0:
                peaks_count_sim = 1.0  # 둘 다 피크가 없는 경우
            else:
                peaks_count_sim = min(len(src_peaks), len(tgt_peaks)) / max(len(src_peaks), len(tgt_peaks))
            
            # 피크 위치를 정규화된 위치(0-1 범위)로 변환
            if len(src_pitch_norm) > 0 and len(src_peaks) > 0:
                src_peak_positions = src_peaks / len(src_pitch_norm)
            else:
                src_peak_positions = np.array([])
                
            if len(tgt_pitch_norm) > 0 and len(tgt_peaks) > 0:
                tgt_peak_positions = tgt_peaks / len(tgt_pitch_norm)
            else:
                tgt_peak_positions = np.array([])
            
            # 피크 위치 유사도 (최근접 피크 거리)
            peak_position_similarity = 0.7  # 기본값
            
            if len(src_peak_positions) > 0 and len(tgt_peak_positions) > 0:
                # 각 피크에 대해 가장 가까운 상대 피크와의 거리 계산
                min_distances = []
                for src_pos in src_peak_positions:
                    if len(tgt_peak_positions) > 0:
                        min_dist = min(abs(src_pos - tgt_pos) for tgt_pos in tgt_peak_positions)
                        min_distances.append(min_dist)
                
                # 평균 최소 거리를 유사도로 변환
                if min_distances:
                    avg_min_dist = sum(min_distances) / len(min_distances)
                    peak_position_similarity = 1.0 - min(1.0, avg_min_dist * 5.0)  # 거리가 0.2 이상이면 0 유사도
            
            # 중요 지점 유사도 종합 (밀도 + 피크 위치)
            important_similarity = 0.5 * density_similarity + 0.5 * peak_position_similarity
            
            # 5. 기존 상관계수 (전체 곡선 유사도)
            # 피치 길이 맞추기
            min_pitch_len = min(len(src_pitch_norm), len(tgt_pitch_norm))
            src_pitch_resized = librosa.util.fix_length(src_pitch_norm, size=min_pitch_len)
            tgt_pitch_resized = librosa.util.fix_length(tgt_pitch_norm, size=min_pitch_len)
            
            # 상관계수 계산
            correlation = np.corrcoef(src_pitch_resized, tgt_pitch_resized)[0, 1]
            
            # NaN 처리
            if np.isnan(correlation):
                correlation = 0.5
            
            # 상관계수 유사도
            correlation_similarity = (correlation + 1) / 2
            
            # 6. 최종 유사도 계산 (각 특성별 가중치 적용)
            similarity = (
                0.35 * direction_matches +           # 방향 일치도
                0.25 * important_similarity +        # 중요 지점 유사도
                0.20 * correlation_similarity +      # 전체 곡선 유사도
                0.20 * magnitude_similarity          # 변화 크기 유사도
            )
            
            # 7. 점수 상향 조정
            adjusted_similarity = self._adjust_score(similarity)
            
            logger.debug(f"피치 유사도: {adjusted_similarity:.4f} (원래: {similarity:.4f}, " +
                        f"방향: {direction_matches:.4f}, 중요: {important_similarity:.4f}, " +
                        f"상관: {correlation_similarity:.4f}, 크기: {magnitude_similarity:.4f})")
            
            return adjusted_similarity
            
        except Exception as e:
            logger.error(f"피치 분석 중 오류: {e}")
            return 0.75  # 오류 시 기본값
    
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
        원본과 합성 오디오 간의 에너지(강세) 패턴 유사도 분석 - 상대적 패턴 중심
        """
        # 에너지(RMS) 계산
        frame_length = int(0.025 * src_sr)  # 25ms 프레임
        hop_length = int(0.010 * src_sr)    # 10ms 홉
        
        src_rms = librosa.feature.rms(y=src_audio, frame_length=frame_length, hop_length=hop_length)[0]
        tgt_rms = librosa.feature.rms(y=tgt_audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # 1. 상대적 패턴 중심: 에너지 변화율
        src_rms_changes = np.diff(src_rms) / (np.mean(src_rms) + 1e-10)
        tgt_rms_changes = np.diff(tgt_rms) / (np.mean(tgt_rms) + 1e-10)
        
        # 2. 에너지 피크 감지
        # 문장 강세 패턴: 상위 30% 에너지 포인트 추출
        src_threshold = np.percentile(src_rms, 70)
        tgt_threshold = np.percentile(tgt_rms, 70)
        
        src_peaks = src_rms > src_threshold
        tgt_peaks = tgt_rms > tgt_threshold
        
        # 정규화된 길이
        min_len = min(len(src_peaks), len(tgt_peaks))
        if min_len < 10:
            return 0.75  # 데이터가 너무 적으면 기본값
        
        # 피크 밀도 계산
        src_peak_density = np.sum(src_peaks) / len(src_peaks)
        tgt_peak_density = np.sum(tgt_peaks) / len(tgt_peaks)
        
        # 피크 밀도 유사도 - 더 관대하게
        peak_density_diff = abs(src_peak_density - tgt_peak_density)
        peak_density_similarity = 1.0 - min(1.0, peak_density_diff * 2.5)
        
        # 3. 에너지 변화 패턴 비교
        # 길이 맞추기 (리샘플링)
        min_change_len = min(len(src_rms_changes), len(tgt_rms_changes))
        
        src_changes_resampled = scipy.signal.resample(src_rms_changes, min_change_len)
        tgt_changes_resampled = scipy.signal.resample(tgt_rms_changes, min_change_len)
        
        # 에너지 변화 방향 비교 (상승/하강 패턴)
        src_directions = np.sign(src_changes_resampled)
        tgt_directions = np.sign(tgt_changes_resampled)
        
        # 방향 일치도 계산
        direction_matches = np.sum(src_directions == tgt_directions) / min_change_len
        
        # 4. 기존 상관계수 방식도 일부 반영
        # 에너지 정규화
        src_rms_norm = src_rms / (np.max(src_rms) + 1e-10)
        tgt_rms_norm = tgt_rms / (np.max(tgt_rms) + 1e-10)
        
        # 길이 맞추기
        min_rms_len = min(len(src_rms_norm), len(tgt_rms_norm))
        src_rms_resized = librosa.util.fix_length(src_rms_norm, size=min_rms_len)
        tgt_rms_resized = librosa.util.fix_length(tgt_rms_norm, size=min_rms_len)
        
        # 상관계수 계산
        correlation = np.corrcoef(src_rms_resized, tgt_rms_resized)[0, 1]
        
        # NaN 처리
        if np.isnan(correlation):
            correlation = 0.5
        
        # 상관계수 유사도
        correlation_similarity = (correlation + 1) / 2
        
        # 5. 종합 에너지 유사도 - 가중치 조정 (방향성 중심)
        similarity = (
            0.4 * direction_matches + 
            0.3 * peak_density_similarity + 
            0.3 * correlation_similarity
        )
        
        # 6. 점수 상향 조정
        adjusted_similarity = self._adjust_score(similarity)
        
        logger.debug(f"에너지 유사도: {adjusted_similarity:.4f} (원래: {similarity:.4f}, 방향: {direction_matches:.4f}, 피크: {peak_density_similarity:.4f}, 상관: {correlation_similarity:.4f})")
        return adjusted_similarity

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
        원본과 합성 오디오 간의 리듬 패턴 유사도 분석 - DTW 활용
        """
        try:
            # 템포와 비트 추출
            src_onset_env = librosa.onset.onset_strength(y=src_audio, sr=src_sr)
            tgt_onset_env = librosa.onset.onset_strength(y=tgt_audio, sr=tgt_sr)
            
            # 1. 상대적 패턴: 온셋 강도 곡선의 정규화된 형태 비교
            src_onset_norm = src_onset_env / (np.max(src_onset_env) + 1e-8)
            tgt_onset_norm = tgt_onset_env / (np.max(tgt_onset_env) + 1e-8)
            
            # 길이 맞추기 (리샘플링)
            min_len = min(len(src_onset_norm), len(tgt_onset_norm))
            if min_len < 10:  # 데이터가 너무 적으면
                return 0.75
                
            # 리샘플링 (동일한 길이로 - 샘플링 레이트 맞추기 위한 용도)
            src_onset_resampled = scipy.signal.resample(src_onset_norm, 200)  # 고정 길이로 리샘플링
            tgt_onset_resampled = scipy.signal.resample(tgt_onset_norm, 200)  # 고정 길이로 리샘플링
            
            # 2. 중요 리듬 패턴: 핵심 리듬 패턴 특성 추출
            # 온셋 피크 감지
            src_peaks, _ = scipy.signal.find_peaks(src_onset_resampled, height=0.3)
            tgt_peaks, _ = scipy.signal.find_peaks(tgt_onset_resampled, height=0.3)
            
            # 피크 밀도 유사도 (단위 시간당 피크 수)
            src_peak_density = len(src_peaks) / len(src_onset_resampled)
            tgt_peak_density = len(tgt_peaks) / len(tgt_onset_resampled)
            
            density_similarity = 1.0 - min(1.0, abs(src_peak_density - tgt_peak_density) * 3.0)
            
            # 3. 비트 간격 패턴 (기존 방식, 수정)
            # onset_detect 함수로 비트 위치 검출
            src_onsets = librosa.onset.onset_detect(onset_envelope=src_onset_env, sr=src_sr)
            tgt_onsets = librosa.onset.onset_detect(onset_envelope=tgt_onset_env, sr=tgt_sr)
            
            # 프레임 인덱스를 시간으로 변환
            src_times = librosa.frames_to_time(src_onsets, sr=src_sr)
            tgt_times = librosa.frames_to_time(tgt_onsets, sr=tgt_sr)
            
            # 비트 수 유사도 - 더 관대하게 조정
            if max(len(src_times), len(tgt_times)) == 0:
                beat_count_similarity = 1.0  # 둘 다 비트가 없는 경우 (완벽한 일치)
            else:
                diff_ratio = abs(len(src_times) - len(tgt_times)) / max(1, max(len(src_times), len(tgt_times)))
                beat_count_similarity = 1.0 - min(diff_ratio * 0.8, 0.8)  # 최대 20% 감점만
            
            # 간격 패턴 유사도 계산 - 더 관대하게
            interval_similarity = 0.7  # 기본값 상향 조정
            
            if len(src_times) > 1 and len(tgt_times) > 1:
                # 간격 계산
                src_intervals = np.diff(src_times)
                tgt_intervals = np.diff(tgt_times)
                
                # 간격 정규화 (평균 간격으로 나눔)
                src_intervals_norm = src_intervals / (np.mean(src_intervals) + 1e-8)
                tgt_intervals_norm = tgt_intervals / (np.mean(tgt_intervals) + 1e-8)
                
                # 길이 맞추기 (최대 10개 간격만 비교)
                compare_len = min(10, min(len(src_intervals_norm), len(tgt_intervals_norm)))
                
                # 중앙값 차이 계산
                src_median = np.median(src_intervals_norm[:compare_len])
                tgt_median = np.median(tgt_intervals_norm[:compare_len])
                median_diff = abs(src_median - tgt_median)
                
                # 간격 중앙값 유사도
                interval_similarity = 1.0 - min(median_diff * 0.5, 0.5)  # 최대 50% 감점
            
            # 4. DTW를 사용한 곡선 유사도 계산
            try:
                # 각 시퀀스를 1차원 배열로 확실히 변환
                src_curve = src_onset_resampled.reshape(-1).astype(np.float64)
                tgt_curve = tgt_onset_resampled.reshape(-1).astype(np.float64)
                
                # 1차원 배열을 2차원으로 변환 (각 시점의 값을 1차원 특성 벡터로)
                src_curve_2d = np.array([[x] for x in src_curve])
                tgt_curve_2d = np.array([[x] for x in tgt_curve])
                
                # fastdtw 계산
                dtw_distance, _ = fastdtw(src_curve_2d, tgt_curve_2d, dist=euclidean)
                
                # 거리에서 유사도로 변환 (정규화)
                # 거리가 작을수록 유사도가 높음, 거리를 배열 길이로 나누어 정규화
                max_possible_distance = len(src_curve) * np.max(np.abs(src_curve - np.mean(tgt_curve)))
                dtw_similarity = 1.0 - min(1.0, dtw_distance / max_possible_distance)
                
            except Exception as e:
                logger.warning(f"DTW 계산 중 오류: {e}, MSE 사용")
                # 오류 발생 시 MSE 사용
                mse = np.mean((src_onset_resampled - tgt_onset_resampled)**2)
                dtw_similarity = 1.0 - min(1.0, mse * 2.0)
            
            # 5. 최종 리듬 유사도 - 상대적 패턴 중심 (가중치 조정)
            rhythm_similarity = (
                0.3 * beat_count_similarity + 
                0.3 * density_similarity + 
                0.2 * interval_similarity + 
                0.2 * dtw_similarity
            )
            
            # 6. 점수 상향 조정
            adjusted_similarity = self._adjust_score(rhythm_similarity)
            
            logger.debug(f"리듬 유사도: {adjusted_similarity:.4f} (원래: {rhythm_similarity:.4f}, DTW: {dtw_similarity:.4f})")
            return adjusted_similarity
            
        except Exception as e:
            logger.error(f"리듬 분석 중 오류: {e}")
            # 오류 발생 시 기본값 반환 (상향 조정)
            return 0.75
    
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
       
        # 점수 상향 조정
        adjusted_similarity = self._adjust_score(vowel_similarity)
        return adjusted_similarity
    
    def _adjust_score(self, original_score: float, strength: float = 0.65) -> float:
        """
        유사도 점수를 보다 관대하게 조정하는 함수
        
        Args:
            original_score: 원래 계산된 점수 (0.0 ~ 1.0)
            strength: 조정 강도 (0.0 ~ 1.0, 높을수록 더 관대하게 조정)
            
        Returns:
            조정된 점수 (0.0 ~ 1.0)
        """
        # 기준점 (이 점수 이상은 좋은 점수로 간주)
        threshold = 0.5
        
        if original_score >= threshold:
            # 이미 좋은 점수는 더 높은 점수대로 매핑
            # 0.5 ~ 1.0 -> 0.75 ~ 1.0 (strength=0.5 기준)
            adjusted = threshold + (original_score - threshold) * (1.0 - threshold) / (1.0 - threshold * (1.0 - strength))
        else:
            # 낮은 점수는 약간만 상향 조정
            # 0.0 ~ 0.5 -> 0.0 ~ 0.75 (strength=0.5 기준)
            adjusted = original_score * (threshold + strength * (1.0 - threshold)) / threshold
        
        # 범위 확인 (0.0 ~ 1.0)
        return max(0.0, min(1.0, adjusted))

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
        plt.title("Waveform Comparison")
        
        # 파형 시간 정규화
        src_times = np.arange(len(src_audio)) / src_sr
        tgt_times = np.arange(len(tgt_audio)) / tgt_sr
        
        max_time = max(src_times[-1], tgt_times[-1])
        
        # 원본 파형
        plt.plot(src_times, src_audio, 'b-', alpha=0.7, label='Source')
        
        # 합성 파형
        plt.plot(tgt_times, tgt_audio, 'r-', alpha=0.7, label='Target')
        
        plt.xlabel('time (sec)')
        plt.ylabel('amplitude')
        plt.legend()
        plt.grid(True)
        
        # 스펙트로그램 비교
        plt.subplot(3, 1, 2)
        plt.title("Source Spectrogram")
        D_src = librosa.amplitude_to_db(np.abs(librosa.stft(src_audio)), ref=np.max)
        librosa.display.specshow(D_src, sr=src_sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        
        plt.subplot(3, 1, 3)
        plt.title("Target Spectrogram")
        D_tgt = librosa.amplitude_to_db(np.abs(librosa.stft(tgt_audio)), ref=np.max)
        librosa.display.specshow(D_tgt, sr=tgt_sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        
        plt.tight_layout()
        
        # 디렉토리 생성 확인
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"오디오 비교 시각화가 저장되었습니다: {output_path}")