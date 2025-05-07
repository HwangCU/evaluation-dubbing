# analyze_single.py
from pathlib import Path
from main import ProsodySimilarityAnalyzer

# 분석기 초기화
analyzer = ProsodySimilarityAnalyzer()

# 단일 파일 분석 실행
result = analyzer.analyze(
    src_audio_path="data/input/윤장목소리1.wav",
    src_textgrid_path="data/input/윤장목소리1.TextGrid",
    tgt_audio_path="data/input/mine_en.wav",
    tgt_textgrid_path="data/input/mine_en.TextGrid",
    output_dir="data/output/example"
)

# 결과 출력
print("\n=== 유사도 분석 완료 ===")
print(f"최종 점수: {result.get('final_score', 0):.4f}")
print(f"등급: {result.get('grade', 'N/A')}")
print(f"주요 특성 점수:")
for key, value in result.items():
    if isinstance(value, float) and key not in ('final_score', 'overall', 'grade'):
        print(f"- {key}: {value:.4f}")