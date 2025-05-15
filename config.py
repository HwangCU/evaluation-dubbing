# config.py

# 경로 설정
INPUT_DIR = "data/input"        # 입력 파일 디렉토리
OUTPUT_DIR = "data/output"      # 출력 파일 디렉토리

# 평가 매개변수 설정
EVAL_CONFIG = {
    "min_silence": 0.3,         # 최소 무음 구간 (초)
    
    # 프로소디 특성 가중치
    "feature_weights": {        
        "pause": 0.3,           # 휴지(일시정지) 유사도 가중치
        "pitch": 0.1,           # 음높이 유사도 가중치
        "energy": 0.2,          # 에너지 유사도 가중치
        "rhythm": 0.1,          # 리듬 유사도 가중치
        "vowel": 0.3            # 모음 길이 유사도 가중치
    },
    
    # 정렬 특성 가중치
    "alignment_weights": {
        "text_similarity": 0.1,       # 텍스트 유사도 가중치
        "temporal_similarity": 0.45,   # 시간적 유사도 가중치
        "speaking_rate_similarity": 0.45  # 발화 속도 유사도 가중치
    },
    
    # 최종 점수 가중치
    "final_score_weights": {
        "alignment": 0.8,        # 정렬 점수 가중치
        "prosody": 0.2           # 프로소디 점수 가중치
    },
    
    # 시간적 유사도 계산을 위한 가중치 및 매개변수
    "temporal_similarity_params": {
        "start_weight": 0.6,      # 시작 시간 가중치 (α)
        "end_weight": 0.4,        # 종료 시간 가중치 (β)
        "relaxation_ratio": 0.5,  # 허용 가능한 relaxation 비율
        "penalty_factor": 0.7     # relaxation 범위 초과 시 패널티 계수
    },

    # 언어별 발화 속도 계수 (영어 기준)
    "language_speed_coefficients": {
        "en": 1.0,    # 영어 (기준)
        "ko": 0.85,   # 한국어 (영어보다 약간 느림)
        "ja": 0.9,    # 일본어
        "zh": 0.8,    # 중국어
        "es": 1.1,    # 스페인어 (영어보다 약간 빠름)
        "fr": 1.05,   # 프랑스어
        "de": 0.95,   # 독일어
        "it": 1.15    # 이탈리아어 (영어보다 빠름)
    }
}

# 오디오 처리 설정
AUDIO_CONFIG = {
    "sample_rate": 22050,       # 샘플링 레이트
    "frame_length": 1024,       # 프레임 길이
    "hop_length": 256,          # 홉 길이
    "n_mels": 128               # 멜 스펙트로그램 주파수 밴드 수
}

# 시각화 설정
VIZ_CONFIG = {
    "plot_width": 12,           # 플롯 너비
    "plot_height": 8,           # 플롯 높이
    "dpi": 100,                 # 해상도
    "colors": {                 # 색상 설정
        "source": "#1f77b4",    # 원본 음성 색상
        "target": "#ff7f0e"     # 합성 음성 색상
    }
}

# 의미 유사도 계산 설정
SEMANTIC_CONFIG = {
    "default_model": "laser",     # 기본 임베딩 모델
    "similarity_threshold": 0.3,  # 의미 유사도 임계값
    "fallback_mode": "time"       # 매칭 실패 시 대체 모드 (시간 기반)
}