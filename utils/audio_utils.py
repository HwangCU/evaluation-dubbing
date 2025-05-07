# utils/audio_utils.py
"""
오디오 처리 유틸리티 함수
"""
import os
import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
logger = logging.getLogger(__name__)
# IPA 문자(발음 기호) 지원을 위해 DejaVu Sans 폰트 사용
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic']  # 폰트 설정
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
def load_audio(file_path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    오디오 파일 로드
    
    Args:
        file_path: 오디오 파일 경로
        sr: 목표 샘플링 레이트 (None이면 원본 유지)
        
    Returns:
        (오디오 데이터, 샘플링 레이트) 튜플
    """
    try:
        logger.info(f"오디오 파일 로드 중: {file_path}")
        y, sr_orig = librosa.load(file_path, sr=sr)
        logger.info(f"오디오 파일 로드 완료: 길이 {len(y)/sr_orig:.2f}초, 샘플레이트 {sr_orig}Hz")
        return y, sr_orig
    except Exception as e:
        logger.error(f"오디오 파일 로드 실패: {e}")
        # 실패 시 빈 오디오 반환
        return np.zeros(1000), 22050

def save_audio(audio: np.ndarray, sr: int, file_path: str) -> str:
    """
    오디오 데이터를 파일로 저장
    
    Args:
        audio: 오디오 데이터
        sr: 샘플링 레이트
        file_path: 저장할 파일 경로
        
    Returns:
        저장된 파일 경로
    """
    try:
        logger.info(f"오디오 파일 저장 중: {file_path}")
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 오디오 저장
        sf.write(file_path, audio, sr)
        logger.info(f"오디오 파일 저장 완료: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"오디오 파일 저장 실패: {e}")
        return ""

def extract_audio_features(
    audio: np.ndarray, 
    sr: int, 
    frame_length: int = 1024,
    hop_length: int = 256
) -> Dict[str, Any]:
    """
    오디오에서 특징 추출
    
    Args:
        audio: 오디오 데이터
        sr: 샘플링 레이트
        frame_length: 프레임 길이
        hop_length: 홉 길이
        
    Returns:
        오디오 특징 딕셔너리
    """
    features = {}
    
    # 1. 스펙트로그램
    spec = np.abs(librosa.stft(audio, n_fft=frame_length, hop_length=hop_length))
    features["spectrogram"] = librosa.amplitude_to_db(spec, ref=np.max)
    
    # 2. 멜 스펙트로그램
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)
    features["mel_spectrogram"] = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 3. MFCC
    features["mfcc"] = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=hop_length)
    
    # 4. 크로마그램
    features["chroma"] = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop_length)
    
    # 5. 온셋 강도
    features["onset_env"] = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    
    # 6. 피치 (F0)
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sr, hop_length=hop_length
        )
        features["f0"] = f0
        features["voiced_flag"] = voiced_flag
    except Exception:
        # 피치 추출 실패 시 대체 방법
        pitches, magnitudes = librosa.piptrack(
            y=audio, sr=sr, hop_length=hop_length
        )
        f0 = []
        for t in range(pitches.shape[1]):
            index = np.argmax(magnitudes[:, t])
            f0.append(pitches[index, t])
        features["f0"] = np.array(f0)
    
    return features

def plot_waveform(
    audio: np.ndarray, 
    sr: int, 
    title: str = "파형",
    file_path: Optional[str] = None
) -> None:
    """
    오디오 파형 시각화
    
    Args:
        audio: 오디오 데이터
        sr: 샘플링 레이트
        title: 그래프 제목
        file_path: 저장할 파일 경로 (None이면 저장하지 않음)
    """
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.xlabel("시간 (초)")
    plt.ylabel("진폭")
    plt.tight_layout()
    
    if file_path:
        plt.savefig(file_path)
        plt.close()
        logger.info(f"파형 시각화가 {file_path}에 저장되었습니다.")
    else:
        plt.show()

def plot_spectrogram(
    audio: np.ndarray, 
    sr: int, 
    title: str = "스펙트로그램",
    file_path: Optional[str] = None
) -> None:
    """
    오디오 스펙트로그램 시각화
    
    Args:
        audio: 오디오 데이터
        sr: 샘플링 레이트
        title: 그래프 제목
        file_path: 저장할 파일 경로 (None이면 저장하지 않음)
    """
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if file_path:
        plt.savefig(file_path)
        plt.close()
        logger.info(f"스펙트로그램 시각화가 {file_path}에 저장되었습니다.")
    else:
        plt.show()

def plot_pitch(
    audio: np.ndarray, 
    sr: int, 
    title: str = "피치",
    file_path: Optional[str] = None
) -> None:
    """
    오디오 피치 시각화
    
    Args:
        audio: 오디오 데이터
        sr: 샘플링 레이트
        title: 그래프 제목
        file_path: 저장할 파일 경로 (None이면 저장하지 않음)
    """
    plt.figure(figsize=(10, 4))
    
    try:
        # pYIN 알고리즘으로 피치 추출
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        times = librosa.times_like(f0)
        
        # 유효한 피치 값만 선택 (0이 아닌 값)
        valid_indices = ~np.isnan(f0)
        valid_times = times[valid_indices]
        valid_f0 = f0[valid_indices]
        
        # 피치 그래프 그리기
        plt.semilogy(valid_times, valid_f0, label='F0')
        plt.grid(True, alpha=0.3)
        plt.title(title)
        plt.xlabel("시간 (초)")
        plt.ylabel("주파수 (Hz)")
        plt.legend()
        
    except Exception as e:
        logger.warning(f"pYIN 피치 추출 실패: {e}. 대체 방법을 사용합니다.")
        
        # 대체 방법: piptrack 사용
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        
        times = librosa.times_like(pitches[0])
        
        # 각 타임 프레임에서 최대 진폭을 갖는 피치 찾기
        f0 = []
        for t in range(pitches.shape[1]):
            index = np.argmax(magnitudes[:, t])
            f0.append(pitches[index, t])
        
        # 유효한 피치 값만 선택
        valid_indices = np.array(f0) > 0
        valid_times = times[valid_indices]
        valid_f0 = np.array(f0)[valid_indices]
        
        plt.semilogy(valid_times, valid_f0, label='F0 (piptrack)')
        plt.grid(True, alpha=0.3)
        plt.title(title)
        plt.xlabel("시간 (초)")
        plt.ylabel("주파수 (Hz)")
        plt.legend()
    
    plt.tight_layout()
    
    if file_path:
        plt.savefig(file_path)
        plt.close()
        logger.info(f"피치 시각화가 {file_path}에 저장되었습니다.")
    else:
        plt.show()

def plot_audio_comparison(
    src_audio: np.ndarray,
    src_sr: int,
    tgt_audio: np.ndarray,
    tgt_sr: int,
    title: str = "원본 vs 합성 오디오 비교",
    file_path: Optional[str] = None
) -> None:
    """
    두 오디오 파형 및 스펙트로그램 비교 시각화
    
    Args:
        src_audio: 원본 오디오 데이터
        src_sr: 원본 샘플링 레이트
        tgt_audio: 합성 오디오 데이터
        tgt_sr: 합성 샘플링 레이트
        title: 그래프 제목
        file_path: 저장할 파일 경로 (None이면 저장하지 않음)
    """
    plt.figure(figsize=(12, 10))
    
    # 파형 비교 (위쪽 두 개)
    plt.subplot(4, 1, 1)
    librosa.display.waveshow(src_audio, sr=src_sr)
    plt.title("원본 오디오 파형")
    plt.ylabel("진폭")
    
    plt.subplot(4, 1, 2)
    librosa.display.waveshow(tgt_audio, sr=tgt_sr)
    plt.title("합성 오디오 파형")
    plt.ylabel("진폭")
    
    # 스펙트로그램 비교 (아래쪽 두 개)
    plt.subplot(4, 1, 3)
    D_src = librosa.amplitude_to_db(np.abs(librosa.stft(src_audio)), ref=np.max)
    librosa.display.specshow(D_src, sr=src_sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("원본 오디오 스펙트로그램")
    
    plt.subplot(4, 1, 4)
    D_tgt = librosa.amplitude_to_db(np.abs(librosa.stft(tgt_audio)), ref=np.max)
    librosa.display.specshow(D_tgt, sr=tgt_sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("합성 오디오 스펙트로그램")
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if file_path:
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
        logger.info(f"오디오 비교 시각화가 {file_path}에 저장되었습니다.")
    else:
        plt.show()