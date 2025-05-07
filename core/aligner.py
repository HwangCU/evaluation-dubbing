# core/aligner.py
"""
음성-텍스트 정렬 및 세그먼트 유사도 평가 모듈
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from config import EVAL_CONFIG

logger = logging.getLogger(__name__)
# IPA 문자(발음 기호) 지원을 위해 DejaVu Sans 폰트 사용
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic']  # 폰트 설정
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
class SegmentAligner:
    """
    원본 세그먼트와 합성 세그먼트를 정렬하고 유사도를 평가하는 클래스
    """
    
    def __init__(self, min_silence: float = None):
        """
        세그먼트 정렬기 초기화
        
        Args:
            min_silence: 최소 무음 구간 (초) (기본값: config.py에서 설정)
        """
        self.min_silence = min_silence or EVAL_CONFIG["min_silence"]
        logger.info(f"세그먼트 정렬기 초기화: 최소 무음 구간 {self.min_silence}초")
    
    def align_segments(
        self,
        src_segments: List[Dict[str, Any]],
        tgt_segments: List[Dict[str, Any]],
        output_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        원본 세그먼트와 합성 세그먼트를 정렬하고 유사도 평가
        
        Args:
            src_segments: 원본 세그먼트 목록
            tgt_segments: 합성 세그먼트 목록
            output_dir: 결과 저장 디렉토리
            
        Returns:
            정렬된 세그먼트 쌍과 유사도 점수
        """
        if output_dir:
            output_dir.mkdir(exist_ok=True, parents=True)
        
        # 텍스트 기반 세그먼트 매칭
        matched_segments = self._match_segments_by_text(src_segments, tgt_segments)
        
        # 매칭이 없는 경우 시간 기반 매칭 시도
        if not matched_segments:
            logger.warning("텍스트 기반 매칭 실패, 시간 기반 매칭으로 대체합니다.")
            matched_segments = self._match_segments_by_time(src_segments, tgt_segments)
        
        # 각 매칭된 세그먼트 쌍의 유사도 계산
        aligned_segments = []
        
        for src_idx, tgt_idx, text_similarity in matched_segments:
            src_segment = src_segments[src_idx]
            tgt_segment = tgt_segments[tgt_idx]
            
            # 시간 정보 추출
            src_start = src_segment["start"]
            src_end = src_segment["end"]
            src_duration = src_segment["duration"]
            
            tgt_start = tgt_segment["start"]
            tgt_end = tgt_segment["end"]
            tgt_duration = tgt_segment["duration"]
            
            # 시간 정렬 유사도 계산
            temporal_similarity = self._calculate_temporal_similarity(
                src_start, src_end, tgt_start, tgt_end
            )
            
            # 길이 비율 유사도 계산
            duration_ratio = min(src_duration, tgt_duration) / max(src_duration, tgt_duration)
            
            # 세그먼트 정보 저장
            aligned_segment = {
                "src_idx": src_idx,
                "tgt_idx": tgt_idx,
                "src_text": src_segment["text"],
                "tgt_text": tgt_segment["text"],
                "src_start": src_start,
                "src_end": src_end,
                "src_duration": src_duration,
                "tgt_start": tgt_start,
                "tgt_end": tgt_end,
                "tgt_duration": tgt_duration,
                "text_similarity": text_similarity,
                "temporal_similarity": temporal_similarity,
                "duration_ratio": duration_ratio,
                "overall_similarity": (0.4 * text_similarity + 0.4 * temporal_similarity + 0.2 * duration_ratio)
            }
            
            aligned_segments.append(aligned_segment)
        
        # 세그먼트 정렬 결과 저장 및 시각화
        if output_dir:
            self._save_alignment_results(aligned_segments, output_dir)
            self._visualize_alignment(aligned_segments, output_dir)
        
        logger.info(f"세그먼트 정렬 완료: {len(aligned_segments)}개 세그먼트 쌍")
        return aligned_segments
    
    def _match_segments_by_text(
        self, 
        src_segments: List[Dict[str, Any]], 
        tgt_segments: List[Dict[str, Any]]
    ) -> List[Tuple[int, int, float]]:
        """
        텍스트 유사도 기반으로 세그먼트 매칭
        
        Args:
            src_segments: 원본 세그먼트 목록
            tgt_segments: 합성 세그먼트 목록
            
        Returns:
            (원본 인덱스, 합성 인덱스, 유사도) 튜플 목록
        """
        # 모든 세그먼트 쌍의 텍스트 유사도 계산
        similarity_matrix = np.zeros((len(src_segments), len(tgt_segments)))
        
        for i, src_segment in enumerate(src_segments):
            for j, tgt_segment in enumerate(tgt_segments):
                similarity_matrix[i, j] = self._calculate_text_similarity(
                    src_segment["text"], tgt_segment["text"]
                )
        
        # Hungarian 알고리즘으로 최적 매칭 찾기
        from scipy.optimize import linear_sum_assignment
        
        # 유사도를 비용으로 변환 (최대화 대신 최소화)
        cost_matrix = 1 - similarity_matrix
        
        # 최적 매칭 계산
        src_indices, tgt_indices = linear_sum_assignment(cost_matrix)
        
        # 결과 변환
        matches = []
        for src_idx, tgt_idx in zip(src_indices, tgt_indices):
            similarity = similarity_matrix[src_idx, tgt_idx]
            
            # 유사도가 너무 낮은 매칭은 제외
            if similarity > 0.2:  # 최소 20% 이상 유사도
                matches.append((src_idx, tgt_idx, similarity))
        
        return matches
    
    def _match_segments_by_time(
        self, 
        src_segments: List[Dict[str, Any]], 
        tgt_segments: List[Dict[str, Any]]
    ) -> List[Tuple[int, int, float]]:
        """
        시간 중첩도 기반으로 세그먼트 매칭
        
        Args:
            src_segments: 원본 세그먼트 목록
            tgt_segments: 합성 세그먼트 목록
            
        Returns:
            (원본 인덱스, 합성 인덱스, 유사도) 튜플 목록
        """
        matches = []
        
        for i, src_segment in enumerate(src_segments):
            src_start = src_segment["start"]
            src_end = src_segment["end"]
            
            best_overlap = 0
            best_tgt_idx = -1
            
            for j, tgt_segment in enumerate(tgt_segments):
                tgt_start = tgt_segment["start"]
                tgt_end = tgt_segment["end"]
                
                # 시간 중첩 계산
                overlap_start = max(src_start, tgt_start)
                overlap_end = min(src_end, tgt_end)
                
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    
                    # 중첩 비율 (중첩 / 총 기간)
                    total_duration = max(src_end, tgt_end) - min(src_start, tgt_start)
                    overlap_ratio = overlap_duration / total_duration
                    
                    if overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        best_tgt_idx = j
            
            # 최선의 매칭이 있으면 추가
            if best_tgt_idx >= 0:
                # 텍스트 유사도 계산
                text_similarity = self._calculate_text_similarity(
                    src_segment["text"], 
                    tgt_segments[best_tgt_idx]["text"]
                )
                
                # 종합 유사도 (중첩 비율과 텍스트 유사도의 가중 평균)
                similarity = 0.7 * best_overlap + 0.3 * text_similarity
                
                matches.append((i, best_tgt_idx, similarity))
        
        return matches
    
    def _calculate_text_similarity(self, src_text: str, tgt_text: str) -> float:
        """
        두 텍스트 간의 유사도 계산 (단순 문자 수준)
        
        Args:
            src_text: 원본 텍스트
            tgt_text: 대상 텍스트
            
        Returns:
            텍스트 유사도 (0.0 ~ 1.0)
        """
        # 텍스트 정규화
        src_text = self._normalize_text(src_text)
        tgt_text = self._normalize_text(tgt_text)
        
        # 두 텍스트가 모두 비어있으면 1.0 반환
        if not src_text and not tgt_text:
            return 1.0
        
        # 둘 중 하나만 비어있으면 0.0 반환
        if not src_text or not tgt_text:
            return 0.0
        
        # 간단한 자카드 유사도 계산
        src_chars = set(src_text)
        tgt_chars = set(tgt_text)
        
        intersection = len(src_chars.intersection(tgt_chars))
        union = len(src_chars.union(tgt_chars))
        
        jaccard = intersection / union if union > 0 else 0
        
        # 길이 유사도 계산
        len_ratio = min(len(src_text), len(tgt_text)) / max(len(src_text), len(tgt_text))
        
        # 자카드 유사도와 길이 유사도의 가중 평균
        return 0.7 * jaccard + 0.3 * len_ratio
    
    def _normalize_text(self, text: str) -> str:
        """
        텍스트 정규화 (소문자 변환, 특수 문자 제거)
        
        Args:
            text: 정규화할 텍스트
            
        Returns:
            정규화된 텍스트
        """
        import re
        
        # 공백 정리 및 소문자 변환
        text = text.strip().lower()
        
        # 특수 문자 제거 (알파벳, 숫자, 한글만 유지)
        text = re.sub(r'[^\w\s가-힣]', '', text)
        
        # 연속 공백 정리
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _calculate_temporal_similarity(
        self, 
        src_start: float, 
        src_end: float,
        tgt_start: float,
        tgt_end: float
    ) -> float:
        """
        두 세그먼트의 시간적 유사도 계산
        
        Args:
            src_start: 원본 세그먼트 시작 시간
            src_end: 원본 세그먼트 종료 시간
            tgt_start: 합성 세그먼트 시작 시간
            tgt_end: 합성 세그먼트 종료 시간
            
        Returns:
            시간적 유사도 (0.0 ~ 1.0)
        """
        # 시작 시간 차이
        start_diff = abs(src_start - tgt_start)
        
        # 종료 시간 차이
        end_diff = abs(src_end - tgt_end)
        
        # 세그먼트 길이
        src_duration = src_end - src_start
        
        # 시간 차이를 세그먼트 길이로 정규화 (0~1 범위)
        if src_duration > 0:
            normalized_start_diff = min(1.0, start_diff / src_duration)
            normalized_end_diff = min(1.0, end_diff / src_duration)
            
            # 시작/종료 시간 유사도 (1 - 정규화된 차이)
            start_similarity = 1.0 - normalized_start_diff
            end_similarity = 1.0 - normalized_end_diff
            
            # 시작/종료 시간 유사도의 평균
            return (start_similarity + end_similarity) / 2
        else:
            return 0.0
    
    def _save_alignment_results(
        self, 
        aligned_segments: List[Dict[str, Any]],
        output_dir: Path
    ) -> None:
        """
        정렬 결과를 JSON 파일로 저장
        
        Args:
            aligned_segments: 정렬된 세그먼트 목록
            output_dir: 결과 저장 디렉토리
        """
        import json
        import numpy as np
        
        # NumPy 타입을 처리할 수 있는 사용자 정의 JSON 인코더
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
    
        # 결과 저장
        alignment_path = output_dir / "segment_alignment.json"
        with open(alignment_path, 'w', encoding='utf-8') as f:
            json.dump(aligned_segments, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        logger.info(f"세그먼트 정렬 결과가 {alignment_path}에 저장되었습니다.")
    
    def _visualize_alignment(
        self, 
        aligned_segments: List[Dict[str, Any]],
        output_dir: Path
    ) -> None:
        """
        세그먼트 정렬 결과 시각화
        
        Args:
            aligned_segments: 정렬된 세그먼트 목록
            output_dir: 결과 저장 디렉토리
        """
        plt.figure(figsize=(12, 6))
        
        # 시간축 최대값 계산
        max_time = 0
        for segment in aligned_segments:
            max_time = max(max_time, segment["src_end"], segment["tgt_end"])
        
        # 여백 추가
        max_time += 1
        
        # 세그먼트 그리기
        for i, segment in enumerate(aligned_segments):
            # 원본 세그먼트 (위쪽)
            src_y = 0.7
            src_start = segment["src_start"]
            src_duration = segment["src_duration"]
            
            plt.barh(
                src_y, src_duration, left=src_start, height=0.2,
                color='blue', alpha=0.6, label='Source' if i == 0 else ""
            )
            
            # 세그먼트 번호 표시
            plt.text(
                src_start + src_duration / 2, src_y, f"{i+1}",
                ha='center', va='center', color='white', fontweight='bold'
            )
            
            # 합성 세그먼트 (아래쪽)
            tgt_y = 0.3
            tgt_start = segment["tgt_start"]
            tgt_duration = segment["tgt_duration"]
            
            plt.barh(
                tgt_y, tgt_duration, left=tgt_start, height=0.2,
                color='red', alpha=0.6, label='Target' if i == 0 else ""
            )
            
            # 세그먼트 번호 표시
            plt.text(
                tgt_start + tgt_duration / 2, tgt_y, f"{i+1}",
                ha='center', va='center', color='white', fontweight='bold'
            )
            
            # 연결선 그리기
            plt.plot(
                [src_start + src_duration/2, tgt_start + tgt_duration/2],
                [src_y - 0.1, tgt_y + 0.1],
                'k--', alpha=0.5
            )
            
            # 유사도 점수 표시
            similarity = segment["overall_similarity"]
            color = 'green' if similarity > 0.7 else 'orange' if similarity > 0.5 else 'red'
            
            plt.text(
                (src_start + src_duration/2 + tgt_start + tgt_duration/2) / 2,
                0.5,
                f"{similarity:.2f}",
                ha='center', va='center', color=color, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3')
            )
        
        # 축 설정
        plt.xlim(0, max_time)
        plt.ylim(0, 1)
        plt.xlabel('시간 (초)')
        plt.yticks([0.3, 0.7], ['합성', '원본'])
        plt.title('세그먼트 정렬 시각화')
        plt.grid(axis='x', alpha=0.3)
        plt.legend()
        
        # 저장
        plt.tight_layout()
        chart_path = output_dir / "segment_alignment.png"
        plt.savefig(chart_path)
        plt.close()
        
        logger.info(f"세그먼트 정렬 시각화가 {chart_path}에 저장되었습니다.")