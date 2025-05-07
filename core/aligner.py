# core/aligner.py
"""
음성-텍스트 정렬 및 세그먼트 유사도 평가 모듈
"""
import logging
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import json
from config import EVAL_CONFIG

# 다국어 임베딩 라이브러리 임포트
try:
    import laserembeddings
    LASER_AVAILABLE = True
except ImportError:
    LASER_AVAILABLE = False
    logging.warning("LASER 임베딩 라이브러리를 찾을 수 없습니다. pip install laserembeddings로 설치하세요.")

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    logging.warning("SentenceTransformer 라이브러리를 찾을 수 없습니다. pip install sentence-transformers로 설치하세요.")

logger = logging.getLogger(__name__)
# IPA 문자(발음 기호) 지원을 위해 DejaVu Sans 폰트 사용
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic']  # 폰트 설정
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

class SegmentAligner:
    """
    원본 세그먼트와 합성 세그먼트를 정렬하고 유사도를 평가하는 클래스
    """
    
    def __init__(self, 
                 min_silence: float = None,
                 embedding_model: str = "laser",
                 similarity_threshold: float = 0.3,
                 enable_n_to_m_mapping: bool = True,  # N:M 매핑 활성화
                 min_segment_duration: float = 0.5):  # 최소 세그먼트 길이
        """
        세그먼트 정렬기 초기화
        
        Args:
            min_silence: 최소 무음 구간 (초) (기본값: config.py에서 설정)
            embedding_model: 사용할 임베딩 모델 ('laser', 'sbert')
            similarity_threshold: 의미 유사도 임계값 (이 값 미만은 매칭에서 제외)
            enable_n_to_m_mapping: N:M 매핑 활성화 여부
            min_segment_duration: 최소 세그먼트 길이 (초), 이보다 짧은 세그먼트는 병합 후보
        """
        self.min_silence = min_silence or EVAL_CONFIG.get("min_silence", 0.3)
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.enable_n_to_m_mapping = enable_n_to_m_mapping
        self.min_segment_duration = min_segment_duration
        
        # 임베딩 모델 초기화
        self.laser = None
        self.sbert = None
        
        if embedding_model == "laser" and LASER_AVAILABLE:
            self.laser = laserembeddings.Laser()
            logger.info("LASER 임베딩 모델을 사용합니다.")
        elif embedding_model == "sbert" and SBERT_AVAILABLE:
            self.sbert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            logger.info("Sentence-BERT 임베딩 모델을 사용합니다.")
        else:
            logger.warning(f"요청한 임베딩 모델({embedding_model})을 사용할 수 없습니다. 기본 텍스트 유사도 방식으로 대체합니다.")
        
        logger.info(f"세그먼트 정렬기 초기화: 최소 무음 구간 {self.min_silence}초, 유사도 임계값 {self.similarity_threshold}")
        logger.info(f"N:M 매핑: {'활성화' if enable_n_to_m_mapping else '비활성화'}, 최소 세그먼트 길이: {min_segment_duration}초")
    
    def align_segments(
        self,
        src_segments: List[Dict[str, Any]],
        tgt_segments: List[Dict[str, Any]],
        src_lang: str = "ko",  # 소스 언어 코드 (ISO 639-1)
        tgt_lang: str = "en",  # 타겟 언어 코드 (ISO 639-1)
        output_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        원본 세그먼트와 합성 세그먼트를 정렬하고 유사도 평가
        
        Args:
            src_segments: 원본 세그먼트 목록
            tgt_segments: 합성 세그먼트 목록
            src_lang: 소스 언어 코드 (ISO 639-1)
            tgt_lang: 타겟 언어 코드 (ISO 639-1)
            output_dir: 결과 저장 디렉토리
            
        Returns:
            정렬된 세그먼트 쌍과 유사도 점수
        """
        if output_dir:
            output_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"세그먼트 정렬 시작: {src_lang} -> {tgt_lang}, 소스 세그먼트 {len(src_segments)}개, 타겟 세그먼트 {len(tgt_segments)}개")
        
        # 1. 전처리: 세그먼트 인덱스 추가 및 짧은 세그먼트 처리
        for i, seg in enumerate(src_segments):
            seg["idx"] = i
            seg["duration"] = seg["end"] - seg["start"]
        
        for i, seg in enumerate(tgt_segments):
            seg["idx"] = i
            seg["duration"] = seg["end"] - seg["start"]
        
        # 짧은 세그먼트 처리
        original_tgt_segments = tgt_segments.copy()  # 원본 보존
        processed_tgt_segments = self._preprocess_segments(tgt_segments, self.min_segment_duration)
        
        logger.info(f"전처리 후 타겟 세그먼트: {len(processed_tgt_segments)}개 (원래: {len(tgt_segments)}개)")
        tgt_segments = processed_tgt_segments
        
        # 2. 세그먼트 매칭
        if self.enable_n_to_m_mapping:
            # N:M 매핑 수행
            matched_segments = self._match_segments_with_n_to_m_mapping(
                src_segments, tgt_segments, src_lang, tgt_lang
            )
        else:
            # 기존 1:1 매핑 수행
            matched_segments = self._match_segments_by_text(
                src_segments, tgt_segments, src_lang, tgt_lang
            )
        
        # 3. 매칭된 세그먼트 정보 생성
        aligned_segments = []
        
        for item in matched_segments:
            if self.enable_n_to_m_mapping:
                src_idx, tgt_indices, text_similarity = item
                
                # 다중 타겟 세그먼트 정보 통합
                tgt_indices = sorted(tgt_indices)  # 순서대로 정렬
                tgt_start = tgt_segments[tgt_indices[0]]["start"]
                tgt_end = tgt_segments[tgt_indices[-1]]["end"]
                tgt_duration = tgt_end - tgt_start
                
                # 통합된 타겟 텍스트
                tgt_text = " ".join([tgt_segments[idx]["text"] for idx in tgt_indices])
                
                # 시간 정렬 유사도 계산
                src_segment = src_segments[src_idx]
                src_start = src_segment["start"]
                src_end = src_segment["end"]
                src_duration = src_segment["duration"]
                
                temporal_similarity = self._calculate_temporal_similarity(
                    src_start, src_end, tgt_start, tgt_end
                )
                
                # 발화 속도 유사도 계산
                speaking_rate_similarity = self._calculate_speaking_rate_match(
                    src_segment, 
                    {"text": tgt_text, "duration": tgt_duration},  # 통합된 타겟 세그먼트
                    src_lang, tgt_lang
                )
                
                # 세그먼트 정보 저장
                aligned_segment = {
                    "src_idx": src_idx,
                    "tgt_idx": tgt_indices,  # 리스트로 저장
                    "src_text": src_segment["text"],
                    "tgt_text": tgt_text,
                    "src_start": src_start,
                    "src_end": src_end,
                    "src_duration": src_duration,
                    "tgt_start": tgt_start,
                    "tgt_end": tgt_end,
                    "tgt_duration": tgt_duration,
                    "text_similarity": text_similarity,
                    "temporal_similarity": temporal_similarity,
                    "speaking_rate_similarity": speaking_rate_similarity,
                    "overall_similarity": (
                        0.5 * text_similarity + 
                        0.3 * temporal_similarity + 
                        0.2 * speaking_rate_similarity
                    ),
                    "mapping_type": "n_to_m" if len(tgt_indices) > 1 else "one_to_one"
                }
                
                aligned_segments.append(aligned_segment)
                
            else:
                # 기존 1:1 매핑 처리
                src_idx, tgt_idx, text_similarity = item
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
                
                # 말하기 속도 유사도 계산
                speaking_rate_similarity = self._calculate_speaking_rate_match(
                    src_segment, tgt_segment, src_lang, tgt_lang
                )
                
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
                    "speaking_rate_similarity": speaking_rate_similarity,
                    "overall_similarity": (
                        0.5 * text_similarity + 
                        0.3 * temporal_similarity + 
                        0.2 * speaking_rate_similarity
                    ),
                    "mapping_type": "one_to_one"
                }
                
                aligned_segments.append(aligned_segment)
        
        # 4. 누락된 원본 세그먼트 확인
        aligned_src_indices = {seg["src_idx"] for seg in aligned_segments}
        missing_src_indices = set(range(len(src_segments))) - aligned_src_indices
        
        if missing_src_indices:
            logger.warning(f"{len(missing_src_indices)}개의 원본 세그먼트가 매칭되지 않았습니다. 강제 매칭을 시도합니다.")
            
            # 매칭된 타겟 인덱스 추적
            matched_tgt_indices = set()
            for segment in aligned_segments:
                if isinstance(segment["tgt_idx"], int):
                    matched_tgt_indices.add(segment["tgt_idx"])
                else:  # 리스트인 경우 (N:M 매핑)
                    matched_tgt_indices.update(segment["tgt_idx"])
            
            # 강제 매칭 추가
            for src_idx in missing_src_indices:
                # 가장 적합한 타겟 세그먼트 찾기
                best_match = self._find_best_fallback_match(
                    src_segments[src_idx], tgt_segments, aligned_segments, matched_tgt_indices
                )
                
                if best_match:
                    aligned_segments.append(best_match)
                    
                    # 매칭된 타겟 인덱스 추가
                    if isinstance(best_match["tgt_idx"], int):
                        matched_tgt_indices.add(best_match["tgt_idx"])
                    else:
                        matched_tgt_indices.update(best_match["tgt_idx"])
        
        # 5. 누락된 타겟 세그먼트 확인
        # 매칭된 타겟 인덱스 추적
        matched_tgt_indices = set()
        for segment in aligned_segments:
            if isinstance(segment["tgt_idx"], int):
                matched_tgt_indices.add(segment["tgt_idx"])
            else:  # 리스트인 경우 (N:M 매핑)
                matched_tgt_indices.update(segment["tgt_idx"])
        
        # 누락된 타겟 인덱스
        missing_tgt_indices = set(range(len(tgt_segments))) - matched_tgt_indices
        
        if missing_tgt_indices:
            logger.warning(f"{len(missing_tgt_indices)}개의 타겟 세그먼트가 매칭되지 않았습니다. 누락 정보를 추가합니다.")
            
            # 누락된 타겟 세그먼트 정보 추가
            aligned_segments = self._add_missing_target_info(
                aligned_segments, tgt_segments, missing_tgt_indices
            )
        
        # 6. 최종 정렬 결과 정렬 (원본 세그먼트 순서대로)
        aligned_segments.sort(key=lambda x: x["src_idx"])
        
        # 7. 세그먼트 정렬 결과 저장 및 시각화
        if output_dir:
            self._save_alignment_results(aligned_segments, output_dir)
            self._visualize_alignment_with_missing(
                aligned_segments, tgt_segments, original_tgt_segments, output_dir
            )
        
        logger.info(f"세그먼트 정렬 완료: {len(aligned_segments)}개 세그먼트 쌍")
        return aligned_segments
    
    def _preprocess_segments(
        self, 
        segments: List[Dict[str, Any]], 
        min_duration: float
    ) -> List[Dict[str, Any]]:
        """
        세그먼트 전처리 (짧은 세그먼트 병합 등)
        
        Args:
            segments: 세그먼트 목록
            min_duration: 최소 세그먼트 길이 (초)
            
        Returns:
            전처리된 세그먼트 목록
        """
        # 짧은 세그먼트 식별
        short_indices = [i for i, seg in enumerate(segments) 
                        if seg["duration"] < min_duration]
        
        if not short_indices:
            return segments.copy()  # 병합할 세그먼트가 없음
        
        logger.info(f"{len(short_indices)}개의 짧은 세그먼트를 병합 처리합니다.")
        
        merged_segments = []
        skip_indices = set()
        
        for i, segment in enumerate(segments):
            if i in skip_indices:
                continue  # 이미 병합된 세그먼트는 건너뛰기
            
            if i in short_indices:
                # 짧은 세그먼트 - 인접한 세그먼트와 병합
                merge_candidates = []
                
                # 앞쪽 세그먼트
                if i > 0 and i-1 not in skip_indices:
                    merge_candidates.append(i-1)
                
                # 뒤쪽 세그먼트
                if i < len(segments)-1 and i+1 not in skip_indices:
                    merge_candidates.append(i+1)
                
                if merge_candidates:
                    # 가장 가까운 세그먼트와 병합
                    closest_idx = min(merge_candidates, 
                                    key=lambda j: abs(segments[j]["start"] - segment["start"]))
                    
                    other_segment = segments[closest_idx]
                    
                    # 새 병합 세그먼트 생성
                    merged_segment = {
                        "idx": segment["idx"],  # 원래 인덱스 유지
                        "start": min(segment["start"], other_segment["start"]),
                        "end": max(segment["end"], other_segment["end"]),
                        "text": segment["text"] + " " + other_segment["text"] 
                                if closest_idx > i else 
                                other_segment["text"] + " " + segment["text"],
                        "duration": max(segment["end"], other_segment["end"]) - 
                                min(segment["start"], other_segment["start"]),
                        "merged_from": [segment["idx"], other_segment["idx"]]  # 병합 정보 추가
                    }
                    
                    merged_segments.append(merged_segment)
                    skip_indices.add(i)
                    skip_indices.add(closest_idx)
                else:
                    # 병합할 세그먼트가 없으면 그대로 유지
                    merged_segments.append(segment.copy())
            elif i not in skip_indices:
                # 일반 세그먼트는 그대로 유지
                merged_segments.append(segment.copy())
        
        return merged_segments
    
    def _match_segments_with_n_to_m_mapping(
        self, 
        src_segments: List[Dict[str, Any]], 
        tgt_segments: List[Dict[str, Any]],
        src_lang: str,
        tgt_lang: str
    ) -> List[Tuple[int, List[int], float]]:
        """
        N:M 매핑을 지원하는 세그먼트 매칭 함수
        
        Returns:
            (원본 인덱스, 매핑된 타겟 인덱스 목록, 유사도) 튜플 목록
        """
        # 1. 기본 1:1 매핑 먼저 수행
        one_to_one_matches = self._match_segments_by_text(src_segments, tgt_segments, src_lang, tgt_lang)
        
        # 매칭된 타겟 인덱스 추적
        matched_tgt_indices = set(tgt_idx for _, tgt_idx, _ in one_to_one_matches)
        
        # 최종 매칭 결과 (N:M 매핑)
        n_to_m_matches = []
        
        # 2. 원본 세그먼트별로 접근
        for src_idx, src_segment in enumerate(src_segments):
            # 이 원본 세그먼트와 매칭된 타겟 인덱스 목록
            matched_targets = []
            matched_similarity = 0.0
            
            # 이미 1:1 매핑에서 매칭된 경우
            for s_idx, t_idx, sim in one_to_one_matches:
                if s_idx == src_idx:
                    matched_targets.append(t_idx)
                    matched_similarity = sim
                    break
                    
            # 매칭되지 않은 타겟 세그먼트 중에서 추가로 매칭될 수 있는 세그먼트 찾기
            if matched_targets:  # 이미 기본 매칭이 있는 경우
                # 매칭되지 않은 타겟 세그먼트 중 인접하고 의미적으로 유사한 세그먼트 찾기
                for tgt_idx, tgt_segment in enumerate(tgt_segments):
                    if tgt_idx in matched_tgt_indices:
                        continue  # 이미 매칭된 타겟은 건너뛰기
                    
                    # 기존 매칭된 타겟과 인접한지 확인
                    is_adjacent = False
                    for mt_idx in matched_targets:
                        if abs(mt_idx - tgt_idx) == 1:  # 인접 인덱스
                            is_adjacent = True
                            break
                    
                    if is_adjacent:
                        # 유사도 계산
                        similarity = self._calculate_text_similarity(
                            src_segment["text"], tgt_segment["text"], src_lang, tgt_lang
                        )
                        
                        # 임계값 이상이면 추가 매칭
                        if similarity >= self.similarity_threshold * 0.7:  # 낮은 임계값 적용
                            matched_targets.append(tgt_idx)
                            matched_tgt_indices.add(tgt_idx)
                
                if matched_targets:
                    # 매칭된 타겟이 있으면 추가
                    n_to_m_matches.append((src_idx, matched_targets, matched_similarity))
            else:
                # 아직 매칭되지 않은 원본 세그먼트는 일단 그냥 두고, 나중에 fallback 처리
                pass
        
        # 3. 아직 매칭되지 않은 원본 세그먼트 확인
        matched_src_indices = set(src_idx for src_idx, _, _ in n_to_m_matches)
        for s_idx, t_idx, sim in one_to_one_matches:
            if s_idx not in matched_src_indices:
                n_to_m_matches.append((s_idx, [t_idx], sim))
                matched_src_indices.add(s_idx)
        
        return n_to_m_matches
    
    def _match_segments_by_text(
        self, 
        src_segments: List[Dict[str, Any]], 
        tgt_segments: List[Dict[str, Any]],
        src_lang: str,
        tgt_lang: str
    ) -> List[Tuple[int, int, float]]:
        """
        텍스트 유사도 기반으로 세그먼트 매칭
        
        Args:
            src_segments: 원본 세그먼트 목록
            tgt_segments: 합성 세그먼트 목록
            src_lang: 소스 언어 코드
            tgt_lang: 타겟 언어 코드
            
        Returns:
            (원본 인덱스, 합성 인덱스, 유사도) 튜플 목록
        """
        # 모든 세그먼트 쌍의 텍스트 유사도 계산
        similarity_matrix = np.zeros((len(src_segments), len(tgt_segments)))
        
        for i, src_segment in enumerate(src_segments):
            for j, tgt_segment in enumerate(tgt_segments):
                similarity_matrix[i, j] = self._calculate_text_similarity(
                    src_segment["text"], tgt_segment["text"], src_lang, tgt_lang
                )
        
        # 유사도를 비용으로 변환 (최대화 대신 최소화)
        cost_matrix = 1 - similarity_matrix.copy()
        
        # 임계값 미만인 경우 높은 비용 할당
        cost_matrix[similarity_matrix < self.similarity_threshold] = 10.0
        
        # Hungarian 알고리즘으로 최적 매칭 찾기
        try:
            from scipy.optimize import linear_sum_assignment
            src_indices, tgt_indices = linear_sum_assignment(cost_matrix)
        except ImportError:
            logger.error("scipy 패키지를 찾을 수 없습니다. pip install scipy로 설치하세요.")
            # 대체 매칭 방법 (그리디)
            src_indices, tgt_indices = self._greedy_assignment(cost_matrix)
        
        # 결과 변환 및 임계값 적용
        matches = []
        for src_idx, tgt_idx in zip(src_indices, tgt_indices):
            similarity = similarity_matrix[src_idx, tgt_idx]
            if similarity >= self.similarity_threshold:
                matches.append((src_idx, tgt_idx, similarity))
        
        return matches
    
    def _greedy_assignment(self, cost_matrix):
        """
        scipy가 없을 경우 대체할 그리디 매칭 알고리즘
        """
        n_rows, n_cols = cost_matrix.shape
        row_indices = []
        col_indices = []
        
        # 비용이 낮은 순서대로 인덱스 정렬
        indices = np.argsort(cost_matrix.flatten())
        row_idx = indices // n_cols
        col_idx = indices % n_cols
        
        used_rows = set()
        used_cols = set()
        
        for i in range(len(indices)):
            r, c = row_idx[i], col_idx[i]
            if r not in used_rows and c not in used_cols:
                row_indices.append(r)
                col_indices.append(c)
                used_rows.add(r)
                used_cols.add(c)
                
                # 필요한 모든 행/열이 매칭되면 종료
                if len(row_indices) == min(n_rows, n_cols):
                    break
        
        return np.array(row_indices), np.array(col_indices)
    
    def _calculate_text_similarity(self, src_text: str, tgt_text: str, src_lang: str, tgt_lang: str) -> float:
        """
        다국어 임베딩을 사용하여 두 텍스트 간의 유사도 계산
        
        Args:
            src_text: 원본 텍스트
            tgt_text: 타겟 텍스트
            src_lang: 소스 언어 코드
            tgt_lang: 타겟 언어 코드
            
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
        
        # LASER 임베딩 사용
        if self.laser is not None:
            try:
                src_embedding = self.laser.embed_sentences([src_text], lang=src_lang)[0]
                tgt_embedding = self.laser.embed_sentences([tgt_text], lang=tgt_lang)[0]
                
                # 코사인 유사도 계산
                similarity = np.dot(src_embedding, tgt_embedding) / (
                    np.linalg.norm(src_embedding) * np.linalg.norm(tgt_embedding)
                )
                
                return float(similarity)
            except Exception as e:
                logger.warning(f"LASER 임베딩 계산 중 오류 발생: {e}. 대체 방법을 사용합니다.")
        
        # Sentence-BERT 임베딩 사용
        if self.sbert is not None:
            try:
                src_embedding = self.sbert.encode(src_text)
                tgt_embedding = self.sbert.encode(tgt_text)
                
                # 코사인 유사도 계산
                similarity = np.dot(src_embedding, tgt_embedding) / (
                    np.linalg.norm(src_embedding) * np.linalg.norm(tgt_embedding)
                )
                
                return float(similarity)
            except Exception as e:
                logger.warning(f"Sentence-BERT 임베딩 계산 중 오류 발생: {e}. 대체 방법을 사용합니다.")
        
        # 임베딩 모델을 사용할 수 없는 경우 기본 문자 수준 유사도 계산
        # 자카드 유사도 계산
        src_chars = set(src_text)
        tgt_chars = set(tgt_text)
        
        intersection = len(src_chars.intersection(tgt_chars))
        union = len(src_chars.union(tgt_chars))
        
        jaccard = intersection / union if union > 0 else 0.0
        
        # 단어 자카드 유사도 계산
        src_words = set(src_text.split())
        tgt_words = set(tgt_text.split())
        
        word_intersection = len(src_words.intersection(tgt_words))
        word_union = len(src_words.union(tgt_words))
        
        word_jaccard = word_intersection / word_union if word_union > 0 else 0.0
        
        # 길이 유사도 계산
        len_ratio = min(len(src_text), len(tgt_text)) / max(len(src_text), len(tgt_text))
        
        # 자카드 유사도와 길이 유사도의 가중 평균
        return 0.4 * jaccard + 0.4 * word_jaccard + 0.2 * len_ratio
    
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
            if best_tgt_idx >= 0 and best_overlap > 0.1:  # 최소한의 중첩 필요
                matches.append((i, best_tgt_idx, best_overlap))
        
        return matches
    
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
        
        # 특수 문자 제거 (알파벳, 숫자, 다국어 유니코드 문자 유지)
        text = re.sub(r'[^\w\s\u0080-\uFFFF]', '', text)
        
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
        두 세그먼트의 시간적 유사도 계산 (논문의 δl, δr relaxations 적용)
        
        Args:
            src_start: 원본 세그먼트 시작 시간
            src_end: 원본 세그먼트 종료 시간
            tgt_start: 합성 세그먼트 시작 시간
            tgt_end: 합성 세그먼트 종료 시간
            
        Returns:
            시간적 유사도 (0.0 ~ 1.0)
        """
        # 원본 및 타겟 세그먼트 길이
        src_duration = src_end - src_start
        tgt_duration = tgt_end - tgt_start
        
        # 허용 가능한 relaxation 범위 계산 (±50%까지 허용)
        max_relaxation = 0.5 * src_duration
        
        # 시작 및 종료 시간 차이 계산
        start_diff = abs(src_start - tgt_start)
        end_diff = abs(src_end - tgt_end)
        
        # Relaxation 범위 내에 있는지 확인
        start_within_relaxation = start_diff <= max_relaxation
        end_within_relaxation = end_diff <= max_relaxation
        
        # 정규화된 유사도 계산
        if src_duration > 0:
            normalized_start_diff = min(1.0, start_diff / src_duration)
            normalized_end_diff = min(1.0, end_diff / src_duration)
            
            # 가중치 적용 (논문에서는 시작 시간이 더 중요)
            α = 0.6  # 시작 시간 가중치
            β = 0.4  # 종료 시간 가중치
            weighted_similarity = α * (1.0 - normalized_start_diff) + β * (1.0 - normalized_end_diff)
            
            # Relaxation 범위를 벗어나면 페널티 적용
            if not start_within_relaxation or not end_within_relaxation:
                weighted_similarity *= 0.7
            
            return weighted_similarity
        else:
            return 0.0
    
    def _calculate_speaking_rate_match(
        self,
        src_segment: Dict[str, Any],
        tgt_segment: Dict[str, Any],
        src_lang: str,
        tgt_lang: str
    ) -> float:
        """
        언어별 특성을 고려한 발화 속도 유사도 계산
        
        Args:
            src_segment: 원본 세그먼트
            tgt_segment: 타겟 세그먼트
            src_lang: 원본 언어 코드
            tgt_lang: 타겟 언어 코드
            
        Returns:
            발화 속도 유사도 (0.0 ~ 1.0)
        """
        # 단어 수 계산
        src_text = src_segment["text"]
        tgt_text = tgt_segment["text"]
        
        src_word_count = len(src_text.split())
        tgt_word_count = len(tgt_text.split())
        
        # 지속 시간
        src_duration = src_segment.get("duration", src_segment.get("end", 0) - src_segment.get("start", 0))
        tgt_duration = tgt_segment.get("duration", tgt_segment.get("end", 0) - tgt_segment.get("start", 0))
        
        if src_duration <= 0 or tgt_duration <= 0 or src_word_count == 0 or tgt_word_count == 0:
            return 0.5  # 기본값
        
        # 언어별 계수 (언어마다 다름)
        language_coefficients = {
            "en": 1.0,    # 영어 (기준)
            "ko": 0.85,   # 한국어 (영어보다 약간 느림)
            "ja": 0.9,    # 일본어
            "zh": 0.8,    # 중국어
            "es": 1.1,    # 스페인어 (영어보다 약간 빠름)
            "fr": 1.05,   # 프랑스어
            "de": 0.95,   # 독일어
            "it": 1.15,   # 이탈리아어 (영어보다 빠름)
            # 기타 언어...
        }
        
        # 기본 계수 (언어 코드가 없는 경우)
        src_coef = language_coefficients.get(src_lang, 1.0)
        tgt_coef = language_coefficients.get(tgt_lang, 1.0)
        
        # 언어별 특성을 고려한 발화 속도 beta 계수 계산
        beta = tgt_coef / src_coef
        
        # 단어당 지속 시간 계산
        src_word_duration = src_duration / src_word_count
        tgt_word_duration = tgt_duration / tgt_word_count
        
        # 발화 속도 유사도 계산 (논문의 speaking rate match 개선)
        normalized_tgt_word_duration = tgt_word_duration / beta
        
        # 정규화된 유사도 (1에 가까울수록 유사)
        similarity = 1.0 - min(1.0, abs(normalized_tgt_word_duration - src_word_duration) / src_word_duration)
        
        return similarity
    
    def _find_best_fallback_match(
        self, 
        src_segment: Dict[str, Any],
        tgt_segments: List[Dict[str, Any]],
        aligned_segments: List[Dict[str, Any]],
        matched_tgt_indices: Set[int]
    ) -> Optional[Dict[str, Any]]:
        """
        매칭되지 않은 세그먼트에 대한 최적의 대체 매칭 찾기
        
        Args:
            src_segment: 매칭되지 않은 원본 세그먼트
            tgt_segments: 모든 타겟 세그먼트 목록
            aligned_segments: 현재까지의 매칭 결과
            matched_tgt_indices: 이미 매칭된 타겟 인덱스 집합
            
        Returns:
            최적의 대체 매칭 또는 None
        """
        # 원본 세그먼트의 시간 정보
        src_start = src_segment["start"]
        src_end = src_segment["end"]
        src_duration = src_segment["duration"]
        src_idx = src_segment.get("idx", -1)
        
        # 시간 거리 기반 최적 매칭 찾기
        best_match = None
        best_score = -1
        best_tgt_indices = []
        
        # 1. 시간적으로 가까운 단일 매칭 시도
        for tgt_idx, tgt_segment in enumerate(tgt_segments):
            # 이미 사용된 타겟 세그먼트는 건너뛰기
            if tgt_idx in matched_tgt_indices:
                continue
            
            # 시간 거리 계산
            tgt_start = tgt_segment["start"]
            tgt_end = tgt_segment["end"]
            tgt_duration = tgt_segment["duration"]
            
            # 시간적 거리 및 지속 시간 비율
            time_distance = min(
                abs(src_start - tgt_start) + abs(src_end - tgt_end),
                abs(src_start - tgt_end) + abs(src_end - tgt_start)
            )
            
            duration_ratio = min(src_duration, tgt_duration) / max(src_duration, tgt_duration)
            
            # 정규화된 시간 거리 점수 (거리가 작을수록 점수가 높음)
            max_allowed_distance = src_duration * 2  # 최대 허용 거리
            time_score = max(0, 1 - time_distance / max_allowed_distance)
            
            # 최종 점수 (시간 거리 70%, 지속 시간 비율 30%)
            match_score = 0.7 * time_score + 0.3 * duration_ratio
            
            if match_score > best_score:
                best_score = match_score
                best_tgt_indices = [tgt_idx]  # 단일 인덱스 저장
        
        # 2. 인접한 미사용 세그먼트들을 모아서 매칭 시도
        if self.enable_n_to_m_mapping:
            # 미사용 세그먼트를 시간순으로 정렬
            unused_segments = []
            for tgt_idx, tgt_segment in enumerate(tgt_segments):
                if tgt_idx not in matched_tgt_indices:
                    unused_segments.append((tgt_idx, tgt_segment))
            
            unused_segments.sort(key=lambda x: x[1]["start"])
            
            # 연속된 미사용 세그먼트 그룹 찾기
            segment_groups = []
            current_group = []
            
            for i, (tgt_idx, tgt_segment) in enumerate(unused_segments):
                if not current_group:
                    current_group.append((tgt_idx, tgt_segment))
                elif i > 0:
                    prev_idx, prev_segment = unused_segments[i-1]
                    
                    # 인접한 세그먼트인지 확인 (시간 간격이 작거나 인덱스가 연속적인지)
                    is_adjacent = (tgt_segment["start"] - prev_segment["end"] < 0.5) or (tgt_idx == prev_idx + 1)
                    
                    if is_adjacent:
                        current_group.append((tgt_idx, tgt_segment))
                    else:
                        # 현재 그룹 저장하고 새 그룹 시작
                        if len(current_group) > 1:  # 2개 이상의 세그먼트가 있는 그룹만 저장
                            segment_groups.append(current_group)
                        current_group = [(tgt_idx, tgt_segment)]
            
            # 마지막 그룹 처리
            if len(current_group) > 1:
                segment_groups.append(current_group)
            
            # 각 그룹에 대해 통합 매칭 평가
            for group in segment_groups:
                group_indices = [idx for idx, _ in group]
                
                # 통합 세그먼트 정보 계산
                group_start = min(seg["start"] for _, seg in group)
                group_end = max(seg["end"] for _, seg in group)
                group_duration = group_end - group_start
                group_text = " ".join(seg["text"] for _, seg in group)
                
                # 시간 거리 및 지속 시간 비율
                time_distance = min(
                    abs(src_start - group_start) + abs(src_end - group_end),
                    abs(src_start - group_end) + abs(src_end - group_start)
                )
                
                duration_ratio = min(src_duration, group_duration) / max(src_duration, group_duration)
                
                # 정규화된 시간 거리 점수
                max_allowed_distance = src_duration * 2
                time_score = max(0, 1 - time_distance / max_allowed_distance)
                
                # 텍스트 유사도 계산
                from_src_lang = "ko"  # 기본값
                to_tgt_lang = "en"  # 기본값
                
                # 최종 점수 (시간 거리 60%, 지속 시간 비율 20%, 텍스트 유사도 20%)
                match_score = 0.6 * time_score + 0.4 * duration_ratio
                
                if match_score > best_score:
                    best_score = match_score
                    best_tgt_indices = group_indices
        
        # 최적의 매칭이 있는 경우
        if best_tgt_indices and best_score > 0.1:  # 최소 점수 임계값
            # 단일 매칭인 경우
            if len(best_tgt_indices) == 1:
                tgt_idx = best_tgt_indices[0]
                tgt_segment = tgt_segments[tgt_idx]
                
                # 대체 매칭 생성
                return {
                    "src_idx": src_idx,
                    "tgt_idx": tgt_idx,
                    "src_text": src_segment["text"],
                    "tgt_text": tgt_segment["text"],
                    "src_start": src_start,
                    "src_end": src_end,
                    "src_duration": src_duration,
                    "tgt_start": tgt_segment["start"],
                    "tgt_end": tgt_segment["end"],
                    "tgt_duration": tgt_segment["duration"],
                    "text_similarity": 0.2,  # 기본값 (낮음)
                    "temporal_similarity": best_score,
                    "speaking_rate_similarity": min(src_duration, tgt_segment["duration"]) / max(src_duration, tgt_segment["duration"]),
                    "overall_similarity": 0.5 * 0.2 + 0.3 * best_score + 0.2 * (min(src_duration, tgt_segment["duration"]) / max(src_duration, tgt_segment["duration"])),
                    "is_fallback": True,  # 이것이 대체 매칭임을 표시
                    "mapping_type": "one_to_one"
                }
            # N:M 매칭인 경우
            else:
                # 타겟 세그먼트 그룹 정보 계산
                tgt_indices = sorted(best_tgt_indices)
                tgt_start = tgt_segments[tgt_indices[0]]["start"]
                tgt_end = tgt_segments[tgt_indices[-1]]["end"]
                tgt_duration = tgt_end - tgt_start
                tgt_text = " ".join([tgt_segments[idx]["text"] for idx in tgt_indices])
                
                # 대체 매칭 생성
                return {
                    "src_idx": src_idx,
                    "tgt_idx": tgt_indices,  # 리스트로 저장
                    "src_text": src_segment["text"],
                    "tgt_text": tgt_text,
                    "src_start": src_start,
                    "src_end": src_end,
                    "src_duration": src_duration,
                    "tgt_start": tgt_start,
                    "tgt_end": tgt_end,
                    "tgt_duration": tgt_duration,
                    "text_similarity": 0.2,  # 기본값 (낮음)
                    "temporal_similarity": best_score,
                    "speaking_rate_similarity": min(src_duration, tgt_duration) / max(src_duration, tgt_duration),
                    "overall_similarity": 0.5 * 0.2 + 0.3 * best_score + 0.2 * (min(src_duration, tgt_duration) / max(src_duration, tgt_duration)),
                    "is_fallback": True,  # 이것이 대체 매칭임을 표시
                    "mapping_type": "n_to_m"
                }
        
        return None
    
    def _add_missing_target_info(
        self,
        aligned_segments: List[Dict[str, Any]],
        tgt_segments: List[Dict[str, Any]],
        missing_tgt_indices: Set[int]
    ) -> List[Dict[str, Any]]:
        """
        누락된 타겟 세그먼트 정보를 정렬 결과에 추가
        
        Args:
            aligned_segments: 정렬된 세그먼트 목록
            tgt_segments: 모든 타겟 세그먼트 목록
            missing_tgt_indices: 누락된 타겟 인덱스 집합
            
        Returns:
            누락 정보가 추가된 정렬 결과
        """
        if not missing_tgt_indices:
            return aligned_segments
        
        enhanced_segments = []
        
        # 각 정렬 결과마다 확인
        for alignment in aligned_segments:
            # 현재 정렬 결과의 타겟 인덱스
            if isinstance(alignment["tgt_idx"], int):
                alignment_tgt_indices = [alignment["tgt_idx"]]
            else:
                alignment_tgt_indices = alignment["tgt_idx"]
            
            # 이 정렬 결과에 가까운 누락된 타겟 세그먼트 찾기
            nearby_missing = []
            
            for tgt_idx in missing_tgt_indices:
                tgt_segment = tgt_segments[tgt_idx]
                
                # 현재 정렬된 타겟 세그먼트와 인접한지 확인
                for aligned_idx in alignment_tgt_indices:
                    if abs(aligned_idx - tgt_idx) == 1:  # 인접 인덱스
                        nearby_missing.append((tgt_idx, tgt_segment))
                        break
            
            # 이 정렬 결과의 복사본 생성
            enhanced_alignment = alignment.copy()
            
            # 인접한 누락 세그먼트가 있으면 정보 추가
            if nearby_missing:
                missing_info = []
                
                for tgt_idx, tgt_segment in nearby_missing:
                    missing_info.append({
                        "missing_tgt_idx": tgt_idx,
                        "missing_tgt_text": tgt_segment["text"],
                        "missing_tgt_start": tgt_segment["start"],
                        "missing_tgt_end": tgt_segment["end"],
                        "missing_tgt_duration": tgt_segment["duration"]
                    })
                
                enhanced_alignment["missing_segments"] = missing_info
            
            enhanced_segments.append(enhanced_alignment)
        
        # 아직 처리되지 않은 누락 세그먼트가 있는지 확인
        processed_missing = set()
        for segment in enhanced_segments:
            if "missing_segments" in segment:
                for missing in segment["missing_segments"]:
                    processed_missing.add(missing["missing_tgt_idx"])
        
        still_missing = missing_tgt_indices - processed_missing
        
        # 아직 처리되지 않은 누락 세그먼트가 있으면 가장 가까운 정렬 결과에 추가
        if still_missing:
            for tgt_idx in still_missing:
                tgt_segment = tgt_segments[tgt_idx]
                
                # 가장 가까운 정렬 결과 찾기
                closest_idx = -1
                min_distance = float('inf')
                
                for i, alignment in enumerate(enhanced_segments):
                    # 시간 거리 계산
                    tgt_time = (tgt_segment["start"] + tgt_segment["end"]) / 2
                    align_tgt_time = (alignment["tgt_start"] + alignment["tgt_end"]) / 2
                    
                    distance = abs(tgt_time - align_tgt_time)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_idx = i
                
                # 가장 가까운 정렬 결과에 추가
                if closest_idx >= 0:
                    if "missing_segments" not in enhanced_segments[closest_idx]:
                        enhanced_segments[closest_idx]["missing_segments"] = []
                    
                    enhanced_segments[closest_idx]["missing_segments"].append({
                        "missing_tgt_idx": tgt_idx,
                        "missing_tgt_text": tgt_segment["text"],
                        "missing_tgt_start": tgt_segment["start"],
                        "missing_tgt_end": tgt_segment["end"],
                        "missing_tgt_duration": tgt_segment["duration"]
                    })
        
        return enhanced_segments
    
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
        
        # 유사도 통계 계산
        text_similarities = [seg["text_similarity"] for seg in aligned_segments]
        temporal_similarities = [seg["temporal_similarity"] for seg in aligned_segments]
        overall_similarities = [seg["overall_similarity"] for seg in aligned_segments]
        
        # 매핑 타입 카운트
        one_to_one_count = sum(1 for seg in aligned_segments if seg.get("mapping_type") == "one_to_one")
        n_to_m_count = sum(1 for seg in aligned_segments if seg.get("mapping_type") == "n_to_m")
        fallback_count = sum(1 for seg in aligned_segments if seg.get("is_fallback", False))
        
        # 누락 세그먼트 카운트
        missing_count = 0
        for seg in aligned_segments:
            if "missing_segments" in seg:
                missing_count += len(seg["missing_segments"])
        
        # 요약 통계 저장
        stats = {
            "alignment_count": len(aligned_segments),
            "text_similarity": {
                "mean": float(np.mean(text_similarities)),
                "min": float(np.min(text_similarities)),
                "max": float(np.max(text_similarities))
            },
            "temporal_similarity": {
                "mean": float(np.mean(temporal_similarities)),
                "min": float(np.min(temporal_similarities)),
                "max": float(np.max(temporal_similarities))
            },
            "overall_similarity": {
                "mean": float(np.mean(overall_similarities)),
                "min": float(np.min(overall_similarities)),
                "max": float(np.max(overall_similarities))
            },
            "mapping_types": {
                "one_to_one": one_to_one_count,
                "n_to_m": n_to_m_count
            },
            "fallback_count": fallback_count,
            "missing_segments_count": missing_count
        }
        
        # 통계 저장
        stats_path = output_dir / "alignment_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"세그먼트 정렬 결과가 {alignment_path}에 저장되었습니다.")
        logger.info(f"정렬 통계가 {stats_path}에 저장되었습니다.")
        logger.info(f"매핑 타입: 1:1 = {one_to_one_count}, N:M = {n_to_m_count}, 대체 매칭 = {fallback_count}, 누락 세그먼트 = {missing_count}")
    
    def _visualize_alignment_with_missing(
        self, 
        aligned_segments: List[Dict[str, Any]],
        tgt_segments: List[Dict[str, Any]],
        original_tgt_segments: List[Dict[str, Any]],
        output_dir: Path
    ) -> None:
        """
        누락된 세그먼트를 포함한 세그먼트 정렬 결과 시각화
        
        Args:
            aligned_segments: 정렬된 세그먼트 목록
            tgt_segments: 전처리된 타겟 세그먼트 목록
            original_tgt_segments: 원본 타겟 세그먼트 목록
               output_dir: 결과 저장 디렉토리
       """
        plt.figure(figsize=(16, 8))
       
       # 시간축 최대값 계산
        max_time = 0
        for segment in aligned_segments:
           max_time = max(max_time, segment["src_end"], segment["tgt_end"])
       
       # 누락된 세그먼트 확인
        for segment in aligned_segments:
           if "missing_segments" in segment:
               for missing in segment["missing_segments"]:
                   max_time = max(max_time, missing["missing_tgt_end"])
       
       # 여백 추가
        max_time += 1
       
       # 세그먼트 그리기
        for i, segment in enumerate(aligned_segments):
           # 원본 세그먼트 (위쪽)
           src_y = 0.8
           src_start = segment["src_start"]
           src_duration = segment["src_duration"]
           
           # 대체 매칭인 경우 다른 색상 사용
           is_fallback = segment.get("is_fallback", False)
           src_color = 'lightblue' if not is_fallback else 'orange'
           alpha = 0.6 if not is_fallback else 0.5
           
           plt.barh(
               src_y, src_duration, left=src_start, height=0.2,
               color=src_color, alpha=alpha, 
               label='Source' if i == 0 else None
           )
           
           # 세그먼트 번호 표시
           plt.text(
               src_start + src_duration / 2, src_y, f"{i+1}",
               ha='center', va='center', color='black', fontweight='bold'
           )
           
           # 원본 텍스트 표시
           max_text_len = 40
           src_text = segment["src_text"]
           src_text = src_text[:max_text_len] + "..." if len(src_text) > max_text_len else src_text
           
           plt.text(
               src_start + src_duration / 2, src_y + 0.15,
               src_text, ha='center', va='bottom', fontsize=8,
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
           )
           
           # 합성 세그먼트 (아래쪽)
           tgt_y = 0.4
           tgt_start = segment["tgt_start"]
           tgt_duration = segment["tgt_duration"]
           
           # N:M 매핑 또는 대체 매칭인 경우 다른 색상/패턴 사용
           is_n_to_m = segment.get("mapping_type") == "n_to_m"
           
           tgt_color = 'lightgreen'
           if is_fallback:
               tgt_color = 'orange'
           elif is_n_to_m:
               tgt_color = 'lightgreen'
           
           # 패턴 설정
           hatch = '//' if is_n_to_m and not is_fallback else None
           
           plt.barh(
               tgt_y, tgt_duration, left=tgt_start, height=0.2,
               color=tgt_color, alpha=alpha, hatch=hatch,
               label='Target (1:1)' if i == 0 and not is_n_to_m and not is_fallback else None
           )
           
           # N:M 매핑인 경우 첫 번째 발견 시에만 레이블 추가
           if is_n_to_m and not is_fallback and all(seg.get("mapping_type") != "n_to_m" or j >= i for j, seg in enumerate(aligned_segments[:i])):
               plt.barh(
                   0, 0, left=0, height=0,  # 더미 바 (레이블만 위한 것)
                   color=tgt_color, alpha=alpha, hatch='//,',
                   label='Target (N:M)'
               )
           
           # 대체 매칭인 경우 첫 번째 발견 시에만 레이블 추가
           if is_fallback and all(not seg.get("is_fallback", False) or j >= i for j, seg in enumerate(aligned_segments[:i])):
               plt.barh(
                   0, 0, left=0, height=0,  # 더미 바 (레이블만 위한 것)
                   color='orange', alpha=alpha,
                   label='Fallback Match'
               )
           
           # 세그먼트 번호 표시
           plt.text(
               tgt_start + tgt_duration / 2, tgt_y, f"{i+1}",
               ha='center', va='center', color='black', fontweight='bold'
           )
           
           # 타겟 텍스트 표시
           tgt_text = segment["tgt_text"]
           tgt_text = tgt_text[:max_text_len] + "..." if len(tgt_text) > max_text_len else tgt_text
           
           plt.text(
               tgt_start + tgt_duration / 2, tgt_y - 0.15,
               tgt_text, ha='center', va='top', fontsize=8,
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
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
               0.6,
               f"{similarity:.2f}",
               ha='center', va='center', color=color, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3')
           )
           
           # 누락된 세그먼트 표시
           if "missing_segments" in segment:
               for j, missing in enumerate(segment["missing_segments"]):
                   # 누락된 세그먼트 (빨간색)
                   missing_y = 0.2  # 타겟보다 아래에 표시
                   missing_start = missing["missing_tgt_start"]
                   missing_duration = missing["missing_tgt_duration"]
                   
                   plt.barh(
                       missing_y, missing_duration, left=missing_start, height=0.15,
                       color='red', alpha=0.6, hatch='xx',
                       label='Missing Target' if j == 0 and i == 0 else None
                   )
                   
                   # 누락된 텍스트 표시
                   missing_text = missing["missing_tgt_text"]
                   missing_text = missing_text[:max_text_len] + "..." if len(missing_text) > max_text_len else missing_text
                   
                   plt.text(
                       missing_start + missing_duration / 2, missing_y - 0.1,
                       f"MISSING: {missing_text}", ha='center', va='top', fontsize=8,
                       color='red', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
                   )
                   
                   # 가장 가까운 타겟 세그먼트와 연결선 (점선)
                   plt.plot(
                       [tgt_start + tgt_duration/2, missing_start + missing_duration/2],
                       [tgt_y - 0.1, missing_y + 0.1],
                       'r:', alpha=0.5
                   )
       
       # 축 설정
        plt.xlim(0, max_time)
        plt.ylim(0, 1)
        plt.xlabel('시간 (초)')
        plt.yticks([0.2, 0.4, 0.8], ['누락', '합성', '원본'])
        plt.title('세그먼트 정렬 시각화 (누락된 세그먼트 포함)')
        plt.grid(axis='x', alpha=0.3)
       
       # 범례 추가
        import matplotlib.patches as mpatches
        
        plt.legend(loc='upper right')
        
       # 저장
        plt.tight_layout()
        chart_path = output_dir / "segment_alignment_with_missing.png"
        plt.savefig(chart_path)
        plt.close()
        
        # 상세 텍스트 정보 저장
        text_info_path = output_dir / "alignment_text.txt"
        with open(text_info_path, 'w', encoding='utf-8') as f:
           f.write("세그먼트 정렬 상세 정보\n")
           f.write("=" * 50 + "\n\n")
           
           for i, segment in enumerate(aligned_segments):
               f.write(f"### 세그먼트 {i+1} ###\n")
               f.write(f"원본: {segment['src_text']}\n")
               f.write(f"합성: {segment['tgt_text']}\n")
               f.write(f"텍스트 유사도: {segment['text_similarity']:.4f}\n")
               f.write(f"시간 유사도: {segment['temporal_similarity']:.4f}\n")
               f.write(f"발화 속도 유사도: {segment.get('speaking_rate_similarity', 0):.4f}\n")
               f.write(f"종합 유사도: {segment['overall_similarity']:.4f}\n")
               f.write(f"매핑 타입: {segment.get('mapping_type', 'one_to_one')}\n")
               f.write(f"대체 매칭: {segment.get('is_fallback', False)}\n")
               
               if "missing_segments" in segment:
                   f.write("\n-- 누락된 세그먼트 --\n")
                   for missing in segment["missing_segments"]:
                       f.write(f"  텍스트: {missing['missing_tgt_text']}\n")
                       f.write(f"  시간: {missing['missing_tgt_start']:.2f}s - {missing['missing_tgt_end']:.2f}s ({missing['missing_tgt_duration']:.2f}s)\n")
               
               f.write("\n")
       
        logger.info(f"누락된 세그먼트를 포함한 정렬 시각화가 {chart_path}에 저장되었습니다.")
        logger.info(f"상세 텍스트 정보가 {text_info_path}에 저장되었습니다.")