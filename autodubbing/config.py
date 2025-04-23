"""
자동 더빙 시스템의 설정 값을 관리하는 모듈

이 모듈은 전체 시스템에서 사용되는 설정값들을 중앙에서 관리합니다.
"""

# 다국어 임베딩 모델 설정
EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'  # 기본 모델
EMBEDDING_MODEL_OPTIONS = {
    'laser': 'LASER',  # Facebook LASER 모델
    'labse': 'LaBSE',  # Google's Language-agnostic BERT Sentence Embedding
    'minilm': 'paraphrase-multilingual-MiniLM-L12-v2',  # HuggingFace의 가벼운 다국어 모델
    'distiluse': 'distiluse-base-multilingual-cased-v1'  # 효율적인 다국어 모델
}

# 문장 추출 설정
MIN_PAUSE_THRESHOLD = 0.3  # 문장 구분을 위한 최소 휴지 시간(초)
MIN_PHRASE_LENGTH = 2  # 최소 문구 단어 수

# 매핑 및 얼라인먼트 설정
SIMILARITY_THRESHOLD = 0.5  # 문장 매핑 최소 유사도 임계값 (0.0~1.0)
ENFORCE_SEQUENTIAL = True   # 순차적 매핑 강제 여부
ALIGNMENT_MAX_RELAXATION = 0.25  # 최대 시간 경계 완화 비율 (원본 시간의 %)

# 온스크린/오프스크린 설정
ON_SCREEN_ISOCHRONY_WEIGHT = 0.8  # 온스크린 더빙의 isochrony 가중치
OFF_SCREEN_ISOCHRONY_WEIGHT = 0.4  # 오프스크린 더빙의 isochrony 가중치
SPEAKING_RATE_MAX = 2.0  # 최대 허용 발화 속도 비율

# 평가 지표 설정
SMOOTHNESS_THRESHOLD = 0.2  # Smoothness 계산을 위한 발화 속도 변화 임계값
FLUENCY_THRESHOLD = 0.3     # Fluency 계산을 위한 임계값

# 로깅 설정
LOGGING_LEVEL = 'INFO'  # 로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)