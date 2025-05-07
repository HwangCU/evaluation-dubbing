# example_single.py
from pathlib import Path
from main import ProsodySimilarityAnalyzer
from config import EVAL_CONFIG, SEMANTIC_CONFIG

def main():
    # config.py에서 설정 가져오기
    default_model = SEMANTIC_CONFIG.get("default_model", "laser")
    similarity_threshold = SEMANTIC_CONFIG.get("similarity_threshold", 0.3)
    
    # 언어별 발화 속도 계수 (참고용)
    language_coeffs = EVAL_CONFIG.get("language_speed_coefficients", {})
    print(f"언어별 발화 속도 계수: {language_coeffs}")
    
    # 분석할 언어 쌍 정의
    src_lang = "en"  
    tgt_lang = "kr" 
    
    print(f"분석 시작: {src_lang} -> {tgt_lang}")
    print(f"임베딩 모델: {default_model}, 유사도 임계값: {similarity_threshold}")
    
    # 분석기 초기화 (config.py 설정 활용)
    analyzer = ProsodySimilarityAnalyzer(
        embedding_model=default_model,
        similarity_threshold=similarity_threshold
        # enable_n_to_m_mapping 파라미터 제거 (또는 ProsodySimilarityAnalyzer 클래스 수정 후 활성화)
    )
    
    # 단일 파일 분석 실행
    result = analyzer.analyze(
        src_audio_path="data/input/en/eleven_joker2/eleven_joker_eng2.wav",
        src_textgrid_path="data/input/text_grid/eleven_joker_eng2.TextGrid",
        tgt_audio_path="data/input/kr/eleven_joker/eleven_joker_kor1.wav",
        tgt_textgrid_path="data/input/text_grid/eleven_joker_kor1.TextGrid",
        src_lang=src_lang,  # 소스 언어 코드
        tgt_lang=tgt_lang,  # 타겟 언어 코드
        output_dir="data/output/eleven_joker",  # 결과 저장 디렉토리
    )
    
    # 결과 출력
    print("\n=== 유사도 분석 완료 ===")
    print(f"최종 점수: {result.get('final_score', 0):.4f}")
    print(f"등급: {result.get('grade', 'N/A')}")
    
    # 주요 특성 점수 출력
    print(f"\n주요 특성 점수:")
    for key, value in result.items():
        if isinstance(value, float) and key not in ('final_score', 'overall'):
            print(f"- {key}: {value:.4f}")
    
    # 언어별 속도 계수 활용 예시
    src_coef = language_coeffs.get(src_lang, 1.0)
    tgt_coef = language_coeffs.get(tgt_lang, 1.0)
    speed_ratio = tgt_coef / src_coef
    
    print(f"\n언어 정보:")
    print(f"- 소스 언어({src_lang}) 속도 계수: {src_coef}")
    print(f"- 타겟 언어({tgt_lang}) 속도 계수: {tgt_coef}")
    print(f"- 언어간 속도 비율: {speed_ratio:.2f}")
    print(f"  → 즉, {tgt_lang}는 {src_lang}보다 약 {abs(speed_ratio-1)*100:.1f}% {'빠름' if speed_ratio > 1 else '느림'}")

if __name__ == "__main__":
    main()