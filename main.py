# main.py
"""
음성 유사도 평가 시스템 메인 모듈
"""
import os
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# 설정 파일 로드
from config import INPUT_DIR, OUTPUT_DIR

# 핵심 모듈 로드
from core.processor import TextGridProcessor, AudioProcessor
from core.analyzer import ProsodyAnalyzer
from core.aligner import SegmentAligner
from core.evaluator import SimilarityEvaluator

# 유틸리티 모듈 로드
from utils.audio_utils import load_audio, plot_audio_comparison
from utils.text_utils import read_text_file, extract_text_from_textgrid
from utils.visualizer import visualize_scores, visualize_radar_chart, create_summary_report

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('prosodic_similarity.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

class ProsodySimilarityAnalyzer:
    """
    원본 음성과 합성 음성의 프로소디 유사도를 분석하는 통합 클래스
    """
    
    def __init__(self, 
                 embedding_model: str = "laser", 
                 similarity_threshold: float = 0.3,
                 enable_n_to_m_mapping: bool = True,  # N:M 매핑 활성화 추가
                 min_segment_duration: float = 0.5):  # 최소 세그먼트 길이 추가
        """
        프로소디 유사도 분석기 초기화
        
        Args:
            embedding_model: 사용할 임베딩 모델 ('laser', 'sbert')
            similarity_threshold: 의미 유사도 임계값
            enable_n_to_m_mapping: N:M 매핑 활성화 여부
            min_segment_duration: 최소 세그먼트 길이 (초)
        """
        # 각 컴포넌트 초기화
        self.textgrid_processor = TextGridProcessor()
        self.audio_processor = AudioProcessor()
        self.segment_aligner = SegmentAligner(
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            enable_n_to_m_mapping=enable_n_to_m_mapping, 
            min_segment_duration=min_segment_duration
        )
        self.prosody_analyzer = ProsodyAnalyzer()
        self.evaluator = SimilarityEvaluator()
        
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.enable_n_to_m_mapping = enable_n_to_m_mapping
        self.min_segment_duration = min_segment_duration
        
        logger.info(f"프로소디 유사도 분석기 초기화 완료 (임베딩 모델: {embedding_model}, 임계값: {similarity_threshold})")
        logger.info(f"N:M 매핑: {'활성화' if enable_n_to_m_mapping else '비활성화'}, 최소 세그먼트 길이: {min_segment_duration}초")

    def analyze(
        self,
        src_audio_path: str,
        src_textgrid_path: str,
        tgt_audio_path: str,
        tgt_textgrid_path: Optional[str] = None,
        src_lang: str = "ko",  # 소스 언어 코드 (ISO 639-1)
        tgt_lang: str = "en",  # 타겟 언어 코드 (ISO 639-1)
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        원본 음성과 합성 음성의 프로소디 유사도 분석 실행
        
        Args:
            src_audio_path: 원본 오디오 파일 경로
            src_textgrid_path: 원본 TextGrid 파일 경로
            tgt_audio_path: 합성 오디오 파일 경로
            tgt_textgrid_path: 합성 TextGrid 파일 경로 (없으면 원본 정보 사용)
            src_lang: 소스 언어 코드 (ISO 639-1)
            tgt_lang: 타겟 언어 코드 (ISO 639-1)
            output_dir: 결과 저장 디렉토리
            
        Returns:
            분석 결과 및 유사도 점수를 포함한 딕셔너리
        """
        # 출력 디렉토리 설정
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path(OUTPUT_DIR)
        
        output_path.mkdir(exist_ok=True, parents=True)
        
        logger.info("유사도 분석 시작")
        logger.info(f"원본 오디오: {src_audio_path} (언어: {src_lang})")
        logger.info(f"원본 TextGrid: {src_textgrid_path}")
        logger.info(f"합성 오디오: {tgt_audio_path} (언어: {tgt_lang})")
        logger.info(f"합성 TextGrid: {tgt_textgrid_path}")
        logger.info(f"출력 디렉토리: {output_path}")
        
        # 1. TextGrid 처리
        logger.info("1. TextGrid 파일 처리 중...")
        src_segments = self.textgrid_processor.process_textgrid(src_textgrid_path)
        
        # 합성 TextGrid가 있으면 처리, 없으면 원본 사용
        if tgt_textgrid_path and os.path.exists(tgt_textgrid_path):
            tgt_segments = self.textgrid_processor.process_textgrid(tgt_textgrid_path)
        else:
            logger.warning("합성 TextGrid가 없습니다. 원본 세그먼트 정보를 사용합니다.")
            tgt_segments = src_segments.copy()
        
        # 2. 오디오 파일 로드
        logger.info("2. 오디오 파일 로드 중...")
        src_audio, src_sr = load_audio(src_audio_path)
        tgt_audio, tgt_sr = load_audio(tgt_audio_path)
        
        # 오디오 비교 시각화
        logger.info("오디오 비교 시각화 생성 중...")
        plot_audio_comparison(
            src_audio, src_sr, tgt_audio, tgt_sr,
            title=f"원본({src_lang}) vs 합성({tgt_lang}) 오디오 비교",
            file_path=str(output_path / "audio_comparison.png")
        )
        
        # 3. 세그먼트 정렬
        logger.info("3. 세그먼트 정렬 중...")
        aligned_segments = self.segment_aligner.align_segments(
            src_segments, tgt_segments,
            src_lang=src_lang, tgt_lang=tgt_lang,
            output_dir=output_path / "alignment"
        )
        
        # 4. 프로소디 분석
        logger.info("4. 프로소디 유사도 분석 중...")
        prosody_scores = self.prosody_analyzer.analyze(
            src_audio, src_sr, src_segments,
            tgt_audio, tgt_sr, tgt_segments,
            output_dir=output_path / "prosody"
        )
        
        # 5. 종합 평가
        logger.info("5. 종합 평가 중...")
        evaluation_results = self.evaluator.evaluate(
            prosody_scores, aligned_segments,
            output_dir=output_path / "evaluation"
        )
        
        # 언어 정보 추가
        evaluation_results["language_info"] = {
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }
        
        # 6. 요약 보고서 및 시각화 생성
        logger.info("6. 요약 보고서 및 시각화 생성 중...")
        
        # 유사도 점수 시각화
        visualize_scores(
            evaluation_results,
            title=f"음성 유사도 평가 결과 ({src_lang} -> {tgt_lang})",
            file_path=str(output_path / "similarity_scores.png")
        )
        
        # 레이더 차트 시각화
        visualize_radar_chart(
            evaluation_results,
            title=f"음성 유사도 레이더 차트 ({src_lang} -> {tgt_lang})",
            file_path=str(output_path / "similarity_radar.png")
        )
        
        # 요약 보고서 생성
        create_summary_report(
            evaluation_results,
            file_path=str(output_path / "summary_report.txt")
        )
        
        logger.info("유사도 분석 완료")
        logger.info(f"최종 점수: {evaluation_results.get('final_score', 0):.4f}, 등급: {evaluation_results.get('grade', 'N/A')}")
        
        return evaluation_results
    
    def analyze_batch(
        self,
        src_dir: str,
        tgt_dir: str,
        src_lang: str = "ko",  # 소스 언어 코드 (ISO 639-1)
        tgt_lang: str = "en",  # 타겟 언어 코드 (ISO 639-1)
        output_dir: Optional[str] = None,
        pattern: str = "*.wav"
    ) -> List[Dict[str, Any]]:
        """
        여러 음성 파일에 대한 일괄 분석 실행
        
        Args:
            src_dir: 원본 오디오 파일 디렉토리
            tgt_dir: 합성 오디오 파일 디렉토리
            src_lang: 소스 언어 코드 (ISO 639-1)
            tgt_lang: 타겟 언어 코드 (ISO 639-1)
            output_dir: 결과 저장 디렉토리
            pattern: 오디오 파일 검색 패턴
            
        Returns:
            각 파일 쌍의 분석 결과 목록
        """
        # 출력 디렉토리 설정
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path(OUTPUT_DIR) / "batch"
        
        output_path.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"일괄 분석 시작: {src_dir}({src_lang}) -> {tgt_dir}({tgt_lang})")
        
        # 원본 디렉토리에서 오디오 파일 검색
        src_files = list(Path(src_dir).glob(pattern))
        
        if not src_files:
            logger.warning(f"'{pattern}' 패턴과 일치하는 원본 파일이 없습니다.")
            return []
        
        results = []
        
        # 각 원본 파일에 대해 처리
        for src_audio_path in src_files:
            # 파일 이름에서 기본 이름 추출
            base_name = src_audio_path.stem
            
            # 원본 TextGrid 파일 경로
            src_textgrid_path = src_audio_path.with_suffix('.TextGrid')
            if not src_textgrid_path.exists():
                logger.warning(f"'{src_textgrid_path}' 파일이 없습니다. 이 파일을 건너뜁니다.")
                continue
            
            # 대응하는 합성 오디오 파일 경로
            tgt_audio_path = Path(tgt_dir) / src_audio_path.name
            if not tgt_audio_path.exists():
                logger.warning(f"'{tgt_audio_path}' 파일이 없습니다. 이 파일을 건너뜁니다.")
                continue
            
            # 합성 TextGrid 파일 경로
            tgt_textgrid_path = tgt_audio_path.with_suffix('.TextGrid')
            
            # 개별 파일 분석
            logger.info(f"파일 분석 중: {base_name}")
            try:
                result = self.analyze(
                    str(src_audio_path),
                    str(src_textgrid_path),
                    str(tgt_audio_path),
                    str(tgt_textgrid_path) if tgt_textgrid_path.exists() else None,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    output_dir=str(output_path / base_name)
                )
                
                # 결과에 파일 정보 추가
                result["file_info"] = {
                    "base_name": base_name,
                    "src_audio": str(src_audio_path),
                    "tgt_audio": str(tgt_audio_path)
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"'{base_name}' 파일 분석 중 오류 발생: {e}")
        
        # 일괄 분석 결과 요약
        if results:
            self._summarize_batch_results(results, output_path, src_lang, tgt_lang)
        
        logger.info(f"일괄 분석 완료: {len(results)}개 파일 처리됨")
        return results
    
    def _summarize_batch_results(
        self, 
        results: List[Dict[str, Any]], 
        output_dir: Path,
        src_lang: str,
        tgt_lang: str
    ) -> None:
        """
        일괄 분석 결과 요약
        
        Args:
            results: 분석 결과 목록
            output_dir: 결과 저장 디렉토리
            src_lang: 소스 언어 코드
            tgt_lang: 타겟 언어 코드
        """
        if not results:
            return
        
        # 결과 요약 파일 생성
        summary_path = output_dir / "batch_summary.txt"
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"=== 일괄 분석 결과 요약 ({src_lang} -> {tgt_lang}) ===\n\n")
                
                # 전체 평균 점수 계산
                avg_final_score = sum(r.get('final_score', 0) for r in results) / len(results)
                avg_prosody_score = sum(r.get('overall', 0) for r in results) / len(results)
                avg_alignment_score = sum(r.get('alignment_score', 0) for r in results) / len(results)
                
                f.write(f"분석된 파일 수: {len(results)}\n")
                f.write(f"평균 최종 점수: {avg_final_score:.4f}\n")
                f.write(f"평균 프로소디 점수: {avg_prosody_score:.4f}\n")
                f.write(f"평균 정렬 점수: {avg_alignment_score:.4f}\n\n")
                
                # 임베딩 모델 및 임계값 정보
                f.write(f"사용된 임베딩 모델: {self.embedding_model}\n")
                f.write(f"의미 유사도 임계값: {self.similarity_threshold}\n\n")
                
                f.write("== 개별 파일 점수 ==\n")
                
                # 점수별로 정렬
                sorted_results = sorted(results, key=lambda r: r.get('final_score', 0), reverse=True)
                
                for i, result in enumerate(sorted_results, 1):
                    base_name = result.get('file_info', {}).get('base_name', f"파일 {i}")
                    final_score = result.get('final_score', 0)
                    alignment_score = result.get('alignment_score', 0)
                    grade = result.get('grade', 'N/A')
                    
                    f.write(f"{i}. {base_name}: {final_score:.4f} (정렬: {alignment_score:.4f}, 등급: {grade})\n")
                
                # 개선 제안사항
                f.write("\n== 일반적인 개선 제안사항 ==\n")
                if avg_final_score < 0.7:
                    f.write("1. 의미 기반 정렬(semantic alignment) 개선 필요: 더 정확한 다국어 임베딩 모델 사용 고려\n")
                    f.write("2. 세그먼트 정확도 향상: 더 정확한 TextGrid 생성 및 정렬 필요\n")
                    f.write(f"3. {src_lang}에서 {tgt_lang}으로의 번역 품질 향상 필요\n")
                    
                    if avg_alignment_score < 0.6:
                        f.write("4. 시간적 정렬 강화: 원본과 합성 세그먼트의 시작/종료 시간을 더 정확히 일치시켜야 함\n")
            
            logger.info(f"일괄 분석 요약이 {summary_path}에 저장되었습니다.")
            
            # 결과를 JSON으로도 저장
            import json
            json_summary_path = output_dir / "batch_summary.json"
            
            summary_data = {
                "analysis_info": {
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "embedding_model": self.embedding_model,
                    "similarity_threshold": self.similarity_threshold,
                    "file_count": len(results)
                },
                "average_scores": {
                    "final_score": avg_final_score,
                    "prosody_score": avg_prosody_score,
                    "alignment_score": avg_alignment_score
                },
                "file_results": [
                    {
                        "file_name": r.get('file_info', {}).get('base_name', f"file_{i}"),
                        "final_score": r.get('final_score', 0),
                        "alignment_score": r.get('alignment_score', 0),
                        "prosody_score": r.get('overall', 0),
                        "grade": r.get('grade', 'N/A')
                    } for i, r in enumerate(sorted_results)
                ]
            }

            with open(json_summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"일괄 분석 JSON 요약이 {json_summary_path}에 저장되었습니다.")
            
        except Exception as e:
            logger.error(f"일괄 분석 요약 생성 중 오류 발생: {e}")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="원본 음성과 합성 음성 간의 프로소디 유사도 분석")
    
    # 단일 파일 분석 모드
    parser.add_argument("--src-audio", help="원본 오디오 파일 경로")
    parser.add_argument("--src-textgrid", help="원본 TextGrid 파일 경로")
    parser.add_argument("--tgt-audio", help="합성 오디오 파일 경로")
    parser.add_argument("--tgt-textgrid", help="합성 TextGrid 파일 경로 (선택적)")
    parser.add_argument("--src-lang", default="ko", help="원본 언어 코드 (ISO 639-1, 기본값: ko)")
    parser.add_argument("--tgt-lang", default="en", help="타겟 언어 코드 (ISO 639-1, 기본값: en)")
    
    # 일괄 분석 모드
    parser.add_argument("--batch", action="store_true", help="일괄 분석 모드 활성화")
    parser.add_argument("--src-dir", help="원본 오디오 파일 디렉토리")
    parser.add_argument("--tgt-dir", help="합성 오디오 파일 디렉토리")
    parser.add_argument("--pattern", default="*.wav", help="오디오 파일 검색 패턴 (기본: *.wav)")
    
    # 임베딩 모델 선택
    parser.add_argument("--embedding-model", choices=["laser", "sbert", "none"], default="laser",
                       help="의미 유사도 계산에 사용할 임베딩 모델 (기본: laser)")
    parser.add_argument("--similarity-threshold", type=float, default=0.3,
                       help="의미 유사도 임계값 (0.0 ~ 1.0, 기본: 0.3)")
    
    # 공통 옵션
    parser.add_argument("--output-dir", help="결과 저장 디렉토리 (기본: ./output)")
    
    args = parser.parse_args()
    
    # 분석기 초기화
    analyzer = ProsodySimilarityAnalyzer(
        embedding_model=args.embedding_model,
        similarity_threshold=args.similarity_threshold
    )
    
    # 실행 모드 결정
    if args.batch:
        # 일괄 분석 모드
        if not args.src_dir or not args.tgt_dir:
            parser.error("일괄 분석 모드에서는 --src-dir과 --tgt-dir이 필요합니다.")
        
        # 일괄 분석 실행
        results = analyzer.analyze_batch(
            args.src_dir,
            args.tgt_dir,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            output_dir=args.output_dir,
            pattern=args.pattern
        )
        
        # 분석 완료 메시지
        if results:
            print(f"\n=== 일괄 분석 완료: {len(results)}개 파일 처리됨 ===")
            print(f"평균 최종 점수: {sum(r.get('final_score', 0) for r in results) / len(results):.4f}")
            print(f"언어 방향: {args.src_lang} -> {args.tgt_lang}")
        
    else:
        # 단일 파일 분석 모드
        if not args.src_audio or not args.src_textgrid or not args.tgt_audio:
            parser.error("단일 파일 분석 모드에서는 --src-audio, --src-textgrid, --tgt-audio가 필요합니다.")
        
        # 단일 파일 분석 실행
        result = analyzer.analyze(
            args.src_audio,
            args.src_textgrid,
            args.tgt_audio,
            args.tgt_textgrid,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            output_dir=args.output_dir
        )
        
        # 분석 완료 메시지
        print("\n=== 유사도 분석 완료 ===")
        print(f"최종 점수: {result.get('final_score', 0):.4f}")
        print(f"등급: {result.get('grade', 'N/A')}")
        print(f"언어 방향: {args.src_lang} -> {args.tgt_lang}")
        print(f"결과가 '{args.output_dir or OUTPUT_DIR}'에 저장되었습니다.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}", exc_info=True)
        print(f"오류가 발생했습니다: {e}")