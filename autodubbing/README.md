## 사용 방법
1. TextGrid 파일을 사용한 더빙
소스와 타겟 언어의 TextGrid 파일이 있을 경우 (단어별 타임스탬프 정보 포함):
```bash
python main.py --mode textgrid \
  --source-textgrid path/to/source.TextGrid \
  --target-textgrid path/to/target.TextGrid \
  --source-lang en \
  --target-lang fr \
  --min-pause 0.3 \
  --output-prefix my_dubbing \
  --output-dir outputs
  ```

2. 텍스트 파일을 사용한 더빙 (시뮬레이션)
TextGrid 파일이 없고 텍스트만 있는 경우 시뮬레이션 모드를 사용합니다:
```bash
python main.py --mode text \
  --source-text path/to/source.txt \
  --target-text path/to/target.txt \
  --source-lang en \
  --target-lang fr \
  --total-duration 60.0 \
  --output-prefix my_text_dubbing \
  --output-dir outputs
```

3. 고급 옵션
더 세밀한 제어를 위한 추가 옵션:
```bash
python main.py --mode textgrid \
  --source-textgrid path/to/source.TextGrid \
  --target-textgrid path/to/target.TextGrid \
  --source-lang en \
  --target-lang fr \
  --embedding-model paraphrase-multilingual-MiniLM-L12-v2 \
  --similarity-threshold 0.6 \
  --use-dp-aligner \
  --tts-engine gtts \
  --voice-id female \
  --output-prefix advanced_dubbing
  ```

## 결과 확인
실행 후 다음 결과물이 생성됩니다:

매핑 정보: outputs/mappings/[prefix]_mappings.json
평가 지표: outputs/metrics/[prefix]_metrics.json
시각화 이미지: outputs/visualizations/ 디렉토리 내 여러 png 파일
합성 오디오:

개별 세그먼트: outputs/audio/[prefix]_segments/
최종 오디오: outputs/audio/[prefix]_final.mp3

### 평가 지표 해석
시스템이 생성하는 주요 평가 지표

Isochrony: 원본과 더빙된 콘텐츠 간의 시간적 일치도 (0.0~1.0)
Smoothness: 발화 속도 변화의 부드러움 (0.0~1.0)
Fluency: 발화의 자연스러움 (0.0~1.0)
Intelligibility: 음성의 명료함 (0.0~1.0)
Overall: 종합 점수 (가중 평균)

## 파일 구조
```
autodubbing/
├── config.py                      # 설정 파일
├── main.py                        # 메인 실행 모듈
│
├── models/                        # 데이터 모델 모듈
│   ├── __init__.py
│   └── sentence.py                # 문장 및 매핑 데이터 클래스
│
├── preprocessing/                 # 전처리 모듈
│   ├── __init__.py
│   ├── textgrid_processor.py      # TextGrid 파일 처리
│   └── text_processor.py          # 텍스트 전처리
│
├── alignment/                     # 얼라인먼트 모듈
│   ├── __init__.py
│   ├── embedder.py                # 문장 임베딩
│   ├── semantic_matcher.py        # 의미 기반 문장 매칭
│   └── prosodic_aligner.py        # 프로소딕 얼라인먼트
│
├── evaluation/                    # 평가 모듈
│   ├── __init__.py
│   ├── metrics.py                 # 평가 지표 계산
│   └── visualizer.py              # 결과 시각화
│
├── synthesis/                     # 합성 모듈 (새로 추가)
│   ├── __init__.py
│   └── tts_controller.py          # TTS 제어 및 합성음성 생성
│
├── outputs/                       # 출력 디렉토리
│   ├── mappings/                  # 매핑 결과 저장
│   ├── metrics/                   # 평가 지표 결과 저장
│   ├── visualizations/            # 시각화 결과 저장
│   └── audio/                     # 합성 오디오 파일 저장
│
└── requirements.txt               # 필요 라이브러리리
```