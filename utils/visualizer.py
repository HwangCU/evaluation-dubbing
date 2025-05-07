# utils/visualizer.py
"""
결과 시각화 유틸리티 함수
"""
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# 폰트 설정 (한글 지원)
# IPA 문자(발음 기호) 지원을 위해 DejaVu Sans 폰트 사용
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic']  # 폰트 설정
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

def visualize_scores(scores: Dict[str, float], title: str = "유사도 점수", file_path: Optional[str] = None) -> None:
    """
    유사도 점수 시각화
    
    Args:
        scores: 점수 딕셔너리
        title: 그래프 제목
        file_path: 저장할 파일 경로 (None이면 화면에 표시)
    """
    # 최종 점수와 overall 제외
    plot_scores = {}
    for k, v in scores.items():
        # 숫자 타입이고 특정 키가 아닌 경우만 포함
        if isinstance(v, (int, float)) and k not in ('final_score', 'overall', 'grade'):
            plot_scores[k] = v
    
    if not plot_scores:
        logger.warning("시각화할 점수가 없습니다.")
        return
    
    # 막대 그래프
    plt.figure(figsize=(10, 6))
    
    # 점수와 키 추출
    categories = list(plot_scores.keys())
    values = list(plot_scores.values())
    
    # 색상 설정 (점수에 따라)
    colors = ['#ff9999' if v < 0.6 else '#66b3ff' if v < 0.8 else '#99ff99' for v in values]
    
    # 막대 그래프 그리기
    bars = plt.bar(categories, values, color=colors)
    
    # 전체 점수 선 추가
    if 'overall' in scores and isinstance(scores['overall'], (int, float)):
        plt.axhline(y=scores['overall'], color='r', linestyle='-', label=f"전체: {scores['overall']:.2f}")
    
    if 'final_score' in scores and isinstance(scores['final_score'], (int, float)):
        plt.axhline(y=scores['final_score'], color='g', linestyle='--', label=f"최종: {scores['final_score']:.2f}")
    
    # 등급 표시
    if 'grade' in scores:
        plt.figtext(0.5, 0.01, f"등급: {scores['grade']}", ha='center', fontsize=12)
    
    # 그래프 설정
    plt.ylim(0, 1.0)
    plt.title(title)
    plt.ylabel('점수 (0-1)')
    plt.grid(axis='y', alpha=0.3)
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
    
    if file_path:
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        plt.savefig(file_path)
        plt.close()
        logger.info(f"점수 시각화가 {file_path}에 저장되었습니다.")
    else:
        plt.show()

def visualize_radar_chart(
    scores: Dict[str, float],
    title: str = "유사도 레이더 차트",
    file_path: Optional[str] = None
) -> None:
    """
    유사도 점수를 레이더 차트로 시각화
    
    Args:
        scores: 점수 딕셔너리
        title: 그래프 제목
        file_path: 저장할 파일 경로 (None이면 화면에 표시)
    """
    # 최종 점수와 overall 제외하고 숫자 타입만 필터링
    plot_scores = {}
    for k, v in scores.items():
        if isinstance(v, (int, float)) and k not in ('final_score', 'overall', 'grade'):
            plot_scores[k] = v
    
    if len(plot_scores) < 3:
        logger.warning("레이더 차트에는 최소 3개 이상의 특성이 필요합니다.")
        return
    
    # 레이더 차트
    plt.figure(figsize=(8, 8))
    
    # 특성과 점수 추출
    categories = [k.replace('_similarity', '').replace('_score', '').capitalize() for k in plot_scores.keys()]
    values = list(plot_scores.values())
    
    # 각도 계산
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    
    # 첫 번째 값을 마지막에 복사하여 도형 닫기
    values.append(values[0])
    angles.append(angles[0])
    categories.append(categories[0])
    
    # 레이더 차트 그리기
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    
    # 축 설정
    ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True)
    
    # 제목 설정
    plt.title(title)
    
    # 등급 및 최종 점수 표시
    if 'grade' in scores and 'final_score' in scores and isinstance(scores['final_score'], (int, float)):
        plt.figtext(0.5, 0.01, f"최종 점수: {scores['final_score']:.2f} (등급: {scores['grade']})", 
                   ha='center', fontsize=12)
    
    if file_path:
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        plt.savefig(file_path)
        plt.close()
        logger.info(f"레이더 차트가 {file_path}에 저장되었습니다.")
    else:
        plt.show()

def visualize_waveform_alignment(
    src_audio: np.ndarray,
    src_sr: int,
    src_segments: List[Dict[str, Any]],
    tgt_audio: np.ndarray,
    tgt_sr: int,
    tgt_segments: List[Dict[str, Any]],
    file_path: Optional[str] = None
) -> None:
    """
    원본과 합성 오디오의 파형 정렬 시각화
    
    Args:
        src_audio: 원본 오디오 데이터
        src_sr: 원본 샘플링 레이트
        src_segments: 원본 세그먼트 정보
        tgt_audio: 합성 오디오 데이터
        tgt_sr: 합성 샘플링 레이트
        tgt_segments: 합성 세그먼트 정보
        file_path: 저장할 파일 경로 (None이면 화면에 표시)
    """
    plt.figure(figsize=(12, 8))
    
    # 위쪽 - 원본 파형
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(src_audio, sr=src_sr)
    plt.title("원본 오디오")
    
    # 세그먼트 경계 표시
    for segment in src_segments:
        plt.axvline(x=segment["start"], color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=segment["end"], color='r', linestyle='--', alpha=0.5)
        
        # 세그먼트 번호 표시
        plt.text((segment["start"] + segment["end"]) / 2, 0, str(segment.get("idx", "-")),
                 ha='center', bbox=dict(facecolor='white', alpha=0.7))
    
    # 아래쪽 - 합성 파형
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(tgt_audio, sr=tgt_sr)
    plt.title("합성 오디오")
    
    # 세그먼트 경계 표시
    for segment in tgt_segments:
        plt.axvline(x=segment["start"], color='b', linestyle='--', alpha=0.5)
        plt.axvline(x=segment["end"], color='b', linestyle='--', alpha=0.5)
        
        # 세그먼트 번호 표시
        plt.text((segment["start"] + segment["end"]) / 2, 0, str(segment.get("idx", "-")),
                 ha='center', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if file_path:
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        plt.savefig(file_path)
        plt.close()
        logger.info(f"파형 정렬 시각화가 {file_path}에 저장되었습니다.")
    else:
        plt.show()

def create_summary_report(
    evaluation: Dict[str, Any],
    file_path: str
) -> None:
    """
    평가 결과 요약 보고서 생성
    
    Args:
        evaluation: 평가 결과 딕셔너리
        file_path: 저장할 파일 경로
    """
    try:
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("=== 음성 유사도 평가 요약 보고서 ===\n\n")
            
            # 최종 점수 및 등급
            f.write(f"최종 점수: {evaluation.get('final_score', 0):.4f}\n")
            f.write(f"등급: {evaluation.get('grade', 'N/A')}\n\n")
            
            # 세부 점수
            f.write("== 세부 점수 ==\n")
            for key, value in evaluation.items():
                if isinstance(value, float) and key not in ('final_score', 'overall', 'grade'):
                    f.write(f"- {key}: {value:.4f}\n")
            
            # 전체 프로소디 점수
            if 'overall' in evaluation:
                f.write(f"\n프로소디 전체 점수: {evaluation['overall']:.4f}\n")
            
            # 정렬 점수
            if 'alignment_score' in evaluation:
                f.write(f"정렬 점수: {evaluation['alignment_score']:.4f}\n\n")
            
            # 개선 제안사항
            if 'improvement_suggestions' in evaluation and evaluation['improvement_suggestions']:
                f.write("== 개선 제안사항 ==\n")
                for i, suggestion in enumerate(evaluation['improvement_suggestions'], 1):
                    f.write(f"{i}. {suggestion}\n")
            
            f.write("\n=== 평가 완료 ===\n")
        
        logger.info(f"요약 보고서가 {file_path}에 저장되었습니다.")
        
    except Exception as e:
        logger.error(f"요약 보고서 생성 실패: {e}")