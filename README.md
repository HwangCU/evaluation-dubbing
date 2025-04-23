
## 파일구조
automatic_dubbing/
│
├── main.py                    # 메인 파이프라인 및 진입점
├── embedder.py                # 다국어 문장 임베딩 모듈
├── matcher.py                 # 의미 유사도 기반 문장 매칭 모듈
├── aligner.py                 # 프로소딕 정렬 알고리즘 구현
├── tts.py                     # 음성 합성 엔진 인터페이스
├── renderer.py                # 최종 오디오 렌더링 및 처리
├── evaluator.py               # 더빙 품질 평가 모듈
├── utils.py                   # TextGrid 처리 및 유틸리티 기능
│
├── output/                    # 출력 파일 저장 디렉토리
│   ├── audio/                 # 합성된 오디오 파일
│   ├── dubbed_audio.wav       # 최종 더빙된 오디오
│   ├── results.json           # 정렬 및 매칭 결과
│   └── evaluation.json        # 평가 결과
│
└── requirements.txt           # 필요한 패키지 목록

## 시스템 사용 법
시스템 사용 방법
pythonfrom pathlib import Path
from dubbing_system.main import AutomaticDubbingPipeline

# 파이프라인 초기화
pipeline = AutomaticDubbingPipeline(
    src_lang="ko",
    tgt_lang="en",
    embedding_model="LASER",
    use_relaxation=True
)

# 더빙 처리
results = pipeline.process(
    src_audio_path="path/to/source_audio.wav",
    src_textgrid_path="path/to/source.TextGrid",
    tgt_text=["This is an example.", "Of target language sentences."],
    output_dir=Path("output"),
    on_screen_segments=[True, False]  # 첫 번째 세그먼트는 on-screen, 두 번째는 off-screen
)

# 평가 결과 출력
print(f"Evaluation results: {results['evaluation']}")
평가 결과 예시
json{
  "original": {
    "isochrony": 0.404,
    "smoothness": 0.190,
    "fluency": 0.599,
    "intelligibility": 0.826,
    "overall": 0.485
  },
  "aligned": {
    "isochrony": 0.785,
    "smoothness": 0.673,
    "fluency": 0.812,
    "intelligibility": 0.801,
    "overall": 0.768
  }
}
