# example_batch.py
from pathlib import Path
from main import ProsodySimilarityAnalyzer
from config import EVAL_CONFIG, SEMANTIC_CONFIG
import json
import os

def main():
    # =============================================
    # 설정값 직접 지정 (여기서 원하는 대로 수정)
    # =============================================
    
    # 디렉토리 설정
    src_dir = "C:/Users/SSAFY/Desktop/work/onion_PJT/data/p225_en"                # 원본 오디오 파일 디렉토리
    tgt_dir = "C:/Users/SSAFY/Desktop/work/onion_PJT/data/p225_kr"                # 합성 오디오 파일 디렉토리
    src_textgrid_dir = "C:/Users/SSAFY/Desktop/work/onion_PJT/S12P31S307/src/AI/Evaluate/data/input/p225_en/text_grid" # 원본 TextGrid 파일 디렉토리
    tgt_textgrid_dir = "C:/Users/SSAFY/Desktop/work/onion_PJT/S12P31S307/src/AI/Evaluate/data/input/p225_kr/text_grid" # 타겟 TextGrid 파일 디렉토리
    output_dir = "C:/Users/SSAFY/Desktop/work/onion_PJT/S12P31S307/src/AI/Evaluate/data/output"  # 결과 저장 디렉토리
    
    # 언어 설정
    src_lang = "en"  # 원본 언어 코드
    tgt_lang = "kr"  # 타겟 언어 코드
    
    # SSML 생성 여부
    generate_ssml = True
    
    # config.py에서 설정 가져오기
    default_model = SEMANTIC_CONFIG.get("default_model", "laser")
    similarity_threshold = SEMANTIC_CONFIG.get("similarity_threshold", 0.3)
    
    # =============================================
    # 실행 코드 (일반적으로 수정 불필요)
    # =============================================
    
    # 언어별 발화 속도 계수
    language_coeffs = EVAL_CONFIG.get("language_speed_coefficients", {})
    print(f"언어별 발화 속도 계수: {language_coeffs}")
    
    print(f"일괄 분석 시작: {src_lang} -> {tgt_lang}")
    print(f"원본 오디오 디렉토리: {src_dir}")
    print(f"타겟 오디오 디렉토리: {tgt_dir}")
    print(f"원본 TextGrid 디렉토리: {src_textgrid_dir}")
    print(f"타겟 TextGrid 디렉토리: {tgt_textgrid_dir}")
    print(f"임베딩 모델: {default_model}, 유사도 임계값: {similarity_threshold}")
    
    # 분석기 초기화
    analyzer = ProsodySimilarityAnalyzer(
        embedding_model=default_model,
        similarity_threshold=similarity_threshold,
        generate_ssml=generate_ssml
    )
    
    # 결과 저장 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 일괄 분석 실행
    results = analyzer.analyze_batch(
        src_dir=src_dir,
        tgt_dir=tgt_dir,
        src_textgrid_dir=src_textgrid_dir,
        tgt_textgrid_dir=tgt_textgrid_dir,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        output_dir=output_dir,
    )
    
    # 분석 결과가 없는 경우
    if not results:
        print("분석 결과가 없습니다. 파일 경로와 패턴을 확인하세요.")
        return
    
    # 분석 결과 요약 출력
    print(f"/n=== 일괄 분석 완료: {len(results)}개 파일 처리됨 ===")
    
    avg_final_score = sum(r.get('final_score', 0) for r in results) / len(results)
    avg_overall_score = sum(r.get('overall', 0) for r in results) / len(results)
    avg_alignment_score = sum(r.get('alignment_score', 0) for r in results) / len(results)
    
    print(f"평균 최종 점수: {avg_final_score:.4f}")
    print(f"평균 프로소디 점수: {avg_overall_score:.4f}")
    print(f"평균 정렬 점수: {avg_alignment_score:.4f}")
    
    # 개별 파일 결과 요약
    print("/n개별 파일 점수 요약:")
    sorted_results = sorted(results, key=lambda r: r.get('final_score', 0), reverse=True)
    
    for i, result in enumerate(sorted_results, 1):
        base_name = result.get('file_info', {}).get('base_name', f"파일 {i}")
        final_score = result.get('final_score', 0)
        overall_score = result.get('overall', 0)
        alignment_score = result.get('alignment_score', 0)
        grade = result.get('grade', 'N/A')
        
        print(f"{i}. {base_name}: 최종={final_score:.4f}, 프로소디={overall_score:.4f}, 정렬={alignment_score:.4f}, 등급={grade}")
    
    # 언어별 속도 계수 정보 출력
    src_coef = language_coeffs.get(src_lang, 1.0)
    tgt_coef = language_coeffs.get(tgt_lang, 1.0)
    speed_ratio = tgt_coef / src_coef
    
    print(f"\n언어 정보:")
    print(f"- 소스 언어({src_lang}) 속도 계수: {src_coef}")
    print(f"- 타겟 언어({tgt_lang}) 속도 계수: {tgt_coef}")
    print(f"- 언어간 속도 비율: {speed_ratio:.2f}")
    print(f"  → 즉, {tgt_lang}는 {src_lang}보다 약 {abs(speed_ratio-1)*100:.1f}% {'빠름' if speed_ratio > 1 else '느림'}")
    
    # SSML 정보 출력
    if generate_ssml:
        combined_ssml_path = os.path.join(output_dir, "combined_tts_output.ssml")
        if os.path.exists(combined_ssml_path):
            print(f"\n병합된 SSML 파일이 생성되었습니다: {combined_ssml_path}")
            
            # SSML 내용 미리보기
            try:
                with open(combined_ssml_path, 'r', encoding='utf-8') as f:
                    ssml_content = f.read()
                
                preview_length = 200
                print(f"\nSSML 미리보기 (처음 {preview_length}자):")
                print("-" * 50)
                print(ssml_content[:preview_length] + "..." if len(ssml_content) > preview_length else ssml_content)
                print("-" * 50)
            except Exception as e:
                print(f"SSML 파일 읽기 중 오류 발생: {e}")
    
    # 요약 파일 안내
    print(f"\n상세 요약 정보:")
    print(f"- 텍스트 요약: {os.path.join(output_dir, 'batch_summary.txt')}")
    print(f"- JSON 요약: {os.path.join(output_dir, 'batch_summary.json')}")
    
    # 처리된 파일 목록 반환 (필요한 경우 활용)
    return results

if __name__ == "__main__":
    main()