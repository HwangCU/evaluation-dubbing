"""
자동 더빙 시스템 메인 실행 모듈

이 모듈은 자동 더빙 파이프라인의 전체 흐름을 관리하고 실행합니다.
문장 매핑, 프로소딕 얼라인먼트, 평가 및 시각화 등의 전체 과정을 조정합니다.
"""

import os
import argparse
import logging
import json
from typing import List, Dict, Tuple, Optional, Union

from models.sentence import Sentence, SentenceMapping
from preprocessing.textgrid_processor import TextGridProcessor
from preprocessing.text_processor import TextProcessor
from alignment.embedder import SentenceEmbedder
from alignment.semantic_matcher import SemanticMatcher
from alignment.prosodic_aligner import ProsodicAligner, DynamicProgrammingAligner
from evaluation.metrics import DubbingEvaluator
from evaluation.visualizer import DubbingVisualizer
from synthesis.tts_controller import TTSController

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("autodubbing.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("AutoDubbing")


class AutoDubbingPipeline:
    """
    자동 더빙 파이프라인 통합 클래스
    
    이 클래스는 자동 더빙의 전체 흐름을 관리하고 실행합니다.
    """
    
    def __init__(self, config: Dict = None):
        """
        자동 더빙 파이프라인 초기화
        
        Args:
            config: 설정값 사전 (기본값 사용 시 None)
        """
        self.config = config or {}
        
        # 구성 요소 초기화
        self.embedder = SentenceEmbedder(
            model_name=self.config.get('embedding_model', 'paraphrase-multilingual-MiniLM-L12-v2')
        )
        
        self.matcher = SemanticMatcher(
            embedder=self.embedder,
            similarity_threshold=self.config.get('similarity_threshold', 0.5),
            enforce_sequential=self.config.get('enforce_sequential', True)
        )
        
        self.aligner = ProsodicAligner(
            max_relaxation=self.config.get('max_relaxation', 0.25),
            max_speaking_rate=self.config.get('max_speaking_rate', 2.0)
        )
        
        self.dp_aligner = DynamicProgrammingAligner(
            max_relaxation=self.config.get('max_relaxation', 0.25),
            max_speaking_rate=self.config.get('max_speaking_rate', 2.0)
        )
        
        self.evaluator = DubbingEvaluator()
        
        # 출력 디렉토리 생성
        self.output_dir = self.config.get('output_dir', 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 시각화 디렉토리 생성
        self.viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)
        
        logger.info("자동 더빙 파이프라인 초기화 완료")
    
    def process_textgrids(
        self,
        source_textgrid_path: str,
        target_textgrid_path: str,
        source_lang: str = 'en',
        target_lang: str = 'ko',
        min_pause: float = 0.3
    ) -> Tuple[List[SentenceMapping], List[SentenceMapping]]:
        """
        TextGrid 파일에서 문장을 추출하고 매핑 및 얼라인먼트 수행
        
        Args:
            source_textgrid_path: 소스 TextGrid 파일 경로
            target_textgrid_path: 타겟 TextGrid 파일 경로
            source_lang: 소스 언어 코드
            target_lang: 타겟 언어 코드
            min_pause: 문장 구분을 위한 최소 휴지 시간(초)
            
        Returns:
            (원본 매핑 리스트, 얼라인된 매핑 리스트) 튜플
        """
        logger.info(f"TextGrid 처리 시작: {source_textgrid_path}, {target_textgrid_path}")
        
        # TextGrid에서 문장 추출
        source_sentences = TextGridProcessor.extract_sentences_from_textgrid(
            source_textgrid_path, "words", source_lang, min_pause)
        
        target_sentences = TextGridProcessor.extract_sentences_from_textgrid(
            target_textgrid_path, "words", target_lang, min_pause)
        
        if not source_sentences or not target_sentences:
            logger.error("TextGrid에서 문장 추출 실패")
            return [], []
        
        logger.info(f"소스 문장 {len(source_sentences)}개, 타겟 문장 {len(target_sentences)}개 추출 완료")
        
        # 문장 매핑
        original_mappings = self.matcher.match_sentences(source_sentences, target_sentences)
        
        if not original_mappings:
            logger.warning("문장 매핑 실패, 폴백 전략 시도")
            original_mappings = self.matcher.match_with_fallback(source_sentences, target_sentences)
        
        logger.info(f"문장 매핑 {len(original_mappings)}개 생성 완료")
        
        # 프로소딕 얼라인먼트
        if self.config.get('use_dp_aligner', False):
            # 동적 프로그래밍 얼라이너 사용
            aligned_mappings = self.dp_aligner.align_mappings(original_mappings)
            logger.info("동적 프로그래밍 얼라이너로 프로소딕 얼라인먼트 완료")
        else:
            # 기본 얼라이너 사용
            aligned_mappings = self.aligner.align_for_dubbing(original_mappings)
            logger.info("기본 얼라이너로 프로소딕 얼라인먼트 완료")
        
        return original_mappings, aligned_mappings
    
    def process_scripts(
        self,
        source_text: str,
        target_text: str,
        source_lang: str = 'en',
        target_lang: str = 'ko',
        total_duration: float = None
    ) -> Tuple[List[SentenceMapping], List[SentenceMapping]]:
        """
        텍스트 스크립트를 처리하여 매핑 및 얼라인먼트 수행
        (시뮬레이션용, 시간 정보가 없는 경우)
        
        Args:
            source_text: 소스 텍스트
            target_text: 타겟 텍스트
            source_lang: 소스 언어 코드
            target_lang: 타겟 언어 코드
            total_duration: 전체 오디오 길이 (초, 시뮬레이션용)
            
        Returns:
            (원본 매핑 리스트, 얼라인된 매핑 리스트) 튜플
        """
        logger.info("텍스트 스크립트 처리 시작")
        
        # 텍스트에서 문장 추출
        source_sentences = TextProcessor.create_sentences_from_text(source_text, source_lang)
        target_sentences = TextProcessor.create_sentences_from_text(target_text, target_lang)
        
        if not source_sentences or not target_sentences:
            logger.error("텍스트에서 문장 추출 실패")
            return [], []
        
        logger.info(f"소스 문장 {len(source_sentences)}개, 타겟 문장 {len(target_sentences)}개 추출 완료")
        
        # 시간 정보 추가 (시뮬레이션)
        if total_duration:
            source_sentences = TextProcessor.add_timing_to_sentences_proportionally(
                source_sentences, total_duration)
            
            # 타겟 문장도 비례적으로 시간 추가 (소스와 동일 시간)
            target_sentences = TextProcessor.add_timing_to_sentences_proportionally(
                target_sentences, total_duration)
        
        # 문장 매핑
        original_mappings = self.matcher.match_sentences(source_sentences, target_sentences)
        
        if not original_mappings:
            logger.warning("문장 매핑 실패, 폴백 전략 시도")
            original_mappings = self.matcher.match_with_fallback(source_sentences, target_sentences)
        
        logger.info(f"문장 매핑 {len(original_mappings)}개 생성 완료")
        
        # 시간 정보가 없는 경우 얼라인먼트 건너뜀
        if total_duration:
            # 프로소딕 얼라인먼트
            aligned_mappings = self.aligner.align_for_dubbing(original_mappings)
            logger.info("프로소딕 얼라인먼트 완료")
        else:
            logger.warning("시간 정보가 없어 프로소딕 얼라인먼트를 건너뜁니다.")
            aligned_mappings = original_mappings
        
        return original_mappings, aligned_mappings
    
    def evaluate_and_visualize(
        self,
        original_mappings: List[SentenceMapping],
        aligned_mappings: List[SentenceMapping],
        output_prefix: str = "result",
        asr_word_error_rates: Dict[int, Tuple[float, float]] = None
    ) -> Dict[str, float]:
        """
        더빙 결과 평가 및 시각화
        
        Args:
            original_mappings: 원본 매핑 리스트
            aligned_mappings: 얼라인된 매핑 리스트
            output_prefix: 출력 파일 이름 접두어
            asr_word_error_rates: ASR 단어 오류율 (인텔리지빌리티 계산용)
            
        Returns:
            평가 지표 사전
        """
        if not original_mappings or not aligned_mappings:
            logger.error("평가를 위한 매핑이 없습니다.")
            return {}
        
        # 평가
        original_metrics = DubbingEvaluator.evaluate_all_metrics(
            original_mappings, asr_word_error_rates)
        
        aligned_metrics = DubbingEvaluator.evaluate_all_metrics(
            aligned_mappings, asr_word_error_rates)
        
        # 평가 결과 저장
        with open(os.path.join(self.output_dir, f"{output_prefix}_metrics.json"), 'w', encoding='utf-8') as f:
            json.dump({
                'original': original_metrics,
                'aligned': aligned_metrics
            }, f, indent=2)
        
        # 시각화
        # 1. 원본 매핑 시각화
        DubbingVisualizer.visualize_sentence_mappings(
            original_mappings,
            os.path.join(self.viz_dir, f"{output_prefix}_original_mappings.png"),
            title="Original Sentence Mappings"
        )
        
        # 2. 얼라인된 매핑 시각화
        DubbingVisualizer.visualize_sentence_mappings(
            aligned_mappings,
            os.path.join(self.viz_dir, f"{output_prefix}_aligned_mappings.png"),
            title="Aligned Sentence Mappings"
        )
        
        # 3. 원본 메트릭 시각화
        DubbingVisualizer.visualize_metrics(
            original_metrics,
            os.path.join(self.viz_dir, f"{output_prefix}_original_metrics.png"),
            title="Original Dubbing Metrics"
        )
        
        # 4. 얼라인된 메트릭 시각화
        DubbingVisualizer.visualize_metrics(
            aligned_metrics,
            os.path.join(self.viz_dir, f"{output_prefix}_aligned_metrics.png"),
            title="Aligned Dubbing Metrics"
        )
        
        # 5. 발화 속도 시각화
        DubbingVisualizer.visualize_speaking_rates(
            aligned_mappings,
            os.path.join(self.viz_dir, f"{output_prefix}_speaking_rates.png"),
            title="Speaking Rate Analysis"
        )
        
        # 6. 비교 분석 시각화
        DubbingVisualizer.visualize_comparative_analysis(
            original_mappings,
            aligned_mappings,
            os.path.join(self.viz_dir, f"{output_prefix}_comparative_analysis.png"),
            title="Prosodic Alignment Comparative Analysis"
        )
        
        logger.info(f"평가 및 시각화 완료 (출력 디렉토리: {self.output_dir}, {self.viz_dir})")
        
        return aligned_metrics
    
    def export_mappings(
        self,
        original_mappings: List[SentenceMapping],
        aligned_mappings: List[SentenceMapping],
        output_path: str
    ) -> None:
        """
        매핑 결과를 JSON 파일로 내보내기
        
        Args:
            original_mappings: 원본 매핑 리스트
            aligned_mappings: 얼라인된 매핑 리스트
            output_path: 저장할 JSON 파일 경로
        """
        result = {
            "original": [],
            "aligned": []
        }
        
        # 원본 매핑 처리
        for mapping in original_mappings:
            result["original"].append({
                "source": {
                    "text": mapping.source.text,
                    "lang": mapping.source.lang,
                    "start_time": mapping.source.start_time,
                    "end_time": mapping.source.end_time,
                    "index": mapping.source.index
                },
                "target": {
                    "text": mapping.target.text,
                    "lang": mapping.target.lang,
                    "start_time": mapping.target.start_time,
                    "end_time": mapping.target.end_time,
                    "index": mapping.target.index
                },
                "similarity": mapping.similarity,
                "isochrony_score": mapping.get_isochrony_score()
            })
        
        # 얼라인된 매핑 처리
        for mapping in aligned_mappings:
            result["aligned"].append({
                "source": {
                    "text": mapping.source.text,
                    "lang": mapping.source.lang,
                    "start_time": mapping.source.start_time,
                    "end_time": mapping.source.end_time,
                    "index": mapping.source.index
                },
                "target": {
                    "text": mapping.target.text,
                    "lang": mapping.target.lang,
                    "start_time": mapping.target.start_time,
                    "end_time": mapping.target.end_time,
                    "index": mapping.target.index
                },
                "similarity": mapping.similarity,
                "start_relaxation": mapping.start_relaxation,
                "end_relaxation": mapping.end_relaxation,
                "time_dilation": mapping.time_dilation,
                "isochrony_score": mapping.get_isochrony_score(),
                "effective_duration": mapping.get_effective_target_duration()
            })
        
        # JSON 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"매핑 결과를 {output_path}에 저장했습니다.")

    def synthesize_aligned_audio(
        self,
        aligned_mappings: List[SentenceMapping],
        output_prefix: str = "result"
        ) -> Tuple[str, Dict[str, float]]:
            
        # """
        # 얼라인된 매핑 정보를 기반으로 음성 합성
        
        # Args:
        #     aligned_mappings: 얼라인된 SentenceMapping 객체 리스트
        #     output_prefix: 출력 파일 이름 접두어
            
        # Returns:
        #     (최종 오디오 파일 경로, 품질 평가 지표 사전) 튜플
        # """

        # 오디오 출력 디렉토리 생성
        audio_dir = os.path.join(self.output_dir, 'audio')
        os.makedirs(audio_dir, exist_ok=True)
        
        # 세그먼트 디렉토리 생성
        segments_dir = os.path.join(audio_dir, f"{output_prefix}_segments")
        os.makedirs(segments_dir, exist_ok=True)
        
        # TTS 컨트롤러 초기화
        tts_controller = TTSController(
            tts_engine=self.config.get('tts_engine', 'gtts'),
            voice_id=self.config.get('voice_id', None)
        )
        
        # 개별 세그먼트 합성
        segment_files = tts_controller.synthesize_aligned_speech(
            aligned_mappings,
            segments_dir,
            prefix=output_prefix
        )
        
        if not segment_files:
            logger.error("합성할 세그먼트가 없습니다.")
            return None, {}
        
        # 최종 오디오 파일 경로
        final_audio_path = os.path.join(audio_dir, f"{output_prefix}_final.mp3")
        
        # 세그먼트 연결
        success = tts_controller.concatenate_audio_segments(
            segment_files,
            final_audio_path
        )
        
        if not success:
            logger.error("최종 오디오 생성 실패")
            return None, {}
        
        # 합성 품질 평가 (필요한 경우)
        # 원본 오디오가 있을 경우 비교 평가 가능
        quality_metrics = {}
        
        logger.info(f"최종 합성 오디오 생성 완료: {final_audio_path}")
        
        return final_audio_path, quality_metrics
    
def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="자동 더빙 시스템")
    
    # 입력 모드 설정
    parser.add_argument('--mode', type=str, choices=['textgrid', 'text'], default='textgrid',
                       help='입력 모드 (textgrid: TextGrid 파일, text: 텍스트 스크립트)')
    
    # TextGrid 모드 인자
    parser.add_argument('--source-textgrid', type=str, help='소스 TextGrid 파일 경로')
    parser.add_argument('--target-textgrid', type=str, help='타겟 TextGrid 파일 경로')
    
    # 텍스트 모드 인자
    parser.add_argument('--source-text', type=str, help='소스 텍스트 파일 경로')
    parser.add_argument('--target-text', type=str, help='타겟 텍스트 파일 경로')
    parser.add_argument('--total-duration', type=float, help='전체 오디오 길이 (초, 시뮬레이션용)')
    
    # 공통 인자
    parser.add_argument('--source-lang', type=str, default='en', help='소스 언어 코드')
    parser.add_argument('--target-lang', type=str, default='ko', help='타겟 언어 코드')
    parser.add_argument('--min-pause', type=float, default=0.3, help='문장 구분을 위한 최소 휴지 시간(초)')
    parser.add_argument('--output-prefix', type=str, default='result', help='출력 파일 이름 접두어')
    parser.add_argument('--output-dir', type=str, default='outputs', help='출력 디렉토리')
    parser.add_argument('--embedding-model', type=str, default='paraphrase-multilingual-MiniLM-L12-v2',
                       help='사용할 임베딩 모델 이름')
    parser.add_argument('--similarity-threshold', type=float, default=0.5,
                       help='문장 매핑 최소 유사도 임계값 (0.0~1.0)')
    parser.add_argument('--use-dp-aligner', action='store_true',
                       help='동적 프로그래밍 얼라이너 사용 여부')
    
    args = parser.parse_args()
    
    # 설정 사전 생성
    config = {
        'output_dir': args.output_dir,
        'embedding_model': args.embedding_model,
        'similarity_threshold': args.similarity_threshold,
        'use_dp_aligner': args.use_dp_aligner
    }
    
    # 자동 더빙 파이프라인 초기화
    pipeline = AutoDubbingPipeline(config)
    
    # 모드에 따라 처리
    if args.mode == 'textgrid':
        if not args.source_textgrid or not args.target_textgrid:
            parser.error("TextGrid 모드에서는 --source-textgrid와 --target-textgrid가 필요합니다.")
        
        # TextGrid 처리
        original_mappings, aligned_mappings = pipeline.process_textgrids(
            args.source_textgrid,
            args.target_textgrid,
            args.source_lang,
            args.target_lang,
            args.min_pause
        )
    else:  # text 모드
        if not args.source_text or not args.target_text:
            parser.error("텍스트 모드에서는 --source-text와 --target-text가 필요합니다.")
        
        # 텍스트 파일 읽기
        with open(args.source_text, 'r', encoding='utf-8') as f:
            source_text = f.read()
        
        with open(args.target_text, 'r', encoding='utf-8') as f:
            target_text = f.read()
        
        # 텍스트 처리
        original_mappings, aligned_mappings = pipeline.process_scripts(
            source_text,
            target_text,
            args.source_lang,
            args.target_lang,
            args.total_duration
        )
    
    # 매핑이 생성되었는지 확인
    if not original_mappings or not aligned_mappings:
        logger.error("매핑 생성 실패")
        return
    
    # 평가 및 시각화
    metrics = pipeline.evaluate_and_visualize(
        original_mappings,
        aligned_mappings,
        args.output_prefix
    )
    
    # 매핑 결과 내보내기
    pipeline.export_mappings(
        original_mappings,
        aligned_mappings,
        os.path.join(args.output_dir, f"{args.output_prefix}_mappings.json")
    )
    
    # 최종 결과 출력
    logger.info("==== 자동 더빙 처리 완료 ====")
    logger.info(f"문장 매핑 수: {len(aligned_mappings)}")
    logger.info(f"이소크로니 점수: {metrics.get('isochrony', 0.0):.3f}")
    logger.info(f"스무스니스 점수: {metrics.get('smoothness', 0.0):.3f}")
    logger.info(f"플루언시 점수: {metrics.get('fluency', 0.0):.3f}")
    logger.info(f"종합 점수: {metrics.get('overall', 0.0):.3f}")
    logger.info(f"출력 디렉토리: {args.output_dir}")
    logger.info("==========================")


if __name__ == "__main__":
    main()