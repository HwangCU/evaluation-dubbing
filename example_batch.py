# analyze_batch.py
from pathlib import Path
from main import ProsodySimilarityAnalyzer

# 분석기 초기화
analyzer = ProsodySimilarityAnalyzer()

# 일괄 분석 실행
results = analyzer.analyze_batch(
    src_dir="data/input/originals",
    tgt_dir="data/input/synthesized",
    output_dir="data/output/batch",
    pattern="*.wav"
)

# 결과 출력
print(f"\n=== 일괄 분석 완료: {len(results)}개 파일 처리됨 ===")
if results:
    avg_score = sum(r.get('final_score', 0) for r in results) / len(results)
    print(f"평균 최종 점수: {avg_score:.4f}")
    
    # 최고/최저 점수 파일 찾기
    best = max(results, key=lambda r: r.get('final_score', 0))
    worst = min(results, key=lambda r: r.get('final_score', 0))
    
    print(f"\n최고 점수 파일: {best['file_info']['base_name']} - {best.get('final_score', 0):.4f} (등급: {best.get('grade', 'N/A')})")
    print(f"최저 점수 파일: {worst['file_info']['base_name']} - {worst.get('final_score', 0):.4f} (등급: {worst.get('grade', 'N/A')})")