# 자동 더빙 프로소딕 유사도 평가 시스템

이 프로젝트는 원본 음성과 합성 음성 간의 프로소딕(운율) 유사도를 분석하고 평가하는 시스템입니다. 특히 Automatic Dubbing(자동 더빙) 시스템에서 생성된 음성의 품질을 정량적으로 측정하여 개선 방향을 제시합니다.

## 주요 특징

- 다국어 음성 간 프로소딕 유사도 평가 (한국어-영어 등 다양한 언어 쌍 지원)
- 다양한 프로소딕 요소 분석 (휴지, 음높이, 에너지, 리듬, 모음 길이 등)
- 텍스트와 음성의 세그먼트 정렬 및 유사도 계산
- 다국어 문장 임베딩 모델(LASER, SBERT)을 활용한 의미적 유사도 평가
- 평가 결과 시각화 및 SSML 변환 기능
- 자세한 분석 리포트 생성

## 요구사항

- Python 3.10 이상
- Montreal Forced Aligner (TextGrid 생성용)
- 필수 라이브러리:
  - librosa
  - numpy
  - matplotlib
  - scipy
  - textgrid
  - laserembeddings
  - sentence_transformers
  - fastdtw

## 설치 방법

1. 저장소 클론
```bash
git clone
cd 
```

2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

3. 다국어 임베딩 모델 설치
```bash
# LASER 모델 설치
pip install laserembeddings
python -m laserembeddings download-models

# Sentence-BERT 설치
pip install sentence-transformers
```

4. Montreal Forced Aligner 설치 (TextGrid 생성용)
```bash
conda config --add channels conda-forge
conda install montreal-forced-aligner
```

## 사용 방법

### 1. TextGrid 파일 생성하기

먼저 원본 및 합성 오디오에 대한 TextGrid 파일을 생성해야 합니다:

```bash
# 영어 모델 다운로드
mfa model download dictionary english_mfa
mfa model download acoustic english_mfa

# 정렬 수행
mfa align CORPUS_DIRECTORY DICTIONARY_PATH ACOUSTIC_MODEL_PATH OUTPUT_DIRECTORY
```

더 자세한 내용은 `./MFA/README` 와 [Montreal Forced Aligner 문서](https://montreal-forced-aligner.readthedocs.io/)를 참조하세요.

### 2. 단일 파일 분석

커맨드 라인에서 단일 음성 파일 쌍을 분석하려면:

```bash
python main.py --src-audio source.wav --src-textgrid source.TextGrid \
               --tgt-audio target.wav --tgt-textgrid target.TextGrid \
               --src-lang en --tgt-lang ko \
               --generate-ssml --output-dir ./output
```

또는 Python 스크립트에서:

```python
from main import ProsodySimilarityAnalyzer

analyzer = ProsodySimilarityAnalyzer(
    embedding_model="laser",
    similarity_threshold=0.3,
    generate_ssml=True
)

result = analyzer.analyze(
    src_audio_path="data/input/en/source.wav",
    src_textgrid_path="data/input/text_grid/source.TextGrid",
    tgt_audio_path="data/input/kr/target.wav",
    tgt_textgrid_path="data/input/text_grid/target.TextGrid",
    src_lang="en",
    tgt_lang="kr",
    output_dir="data/output/result"
)

# 결과 출력
print(f"최종 점수: {result.get('final_score', 0):.4f}")
print(f"등급: {result.get('grade', 'N/A')}")
```

### 3. 일괄 처리 분석

여러 파일을 일괄적으로 분석하려면:

```bash
python main.py --batch \
               --src-dir "./data/input/en" \
               --tgt-dir "./data/input/kr" \
               --src-textgrid-dir "./data/input/en/text_grid" \
               --tgt-textgrid-dir "./data/input/kr/text_grid" \
               --src-lang "en" --tgt-lang "kr" \
               --generate-ssml --output-dir "./data/output"
```

예제 스크립트 활용:
```bash
# 단일 파일 분석 예제
python example_single.py

# 일괄 분석 예제
python example_batch.py
```

## 분석 방법론

### 1. 프로소딕 요소 분석

다음과 같은 프로소딕 요소들의 유사도를 측정합니다:

#### 1.1. 휴지(Pause) 패턴 유사도
- **무음 구간 검출**: RMS 에너지를 사용해 무음 구간을 식별합니다.
- **개수 유사도** 및 **위치 유사도**를 종합하여 최종 점수를 계산합니다.

#### 1.2. 음높이(Pitch) 패턴 유사도
- **피치 추출 및 정규화**: 서로 다른 화자의 음역대 차이를 보정합니다.
- **상관 계수**를 계산하여 피치 윤곽의 유사도를 측정합니다.

#### 1.3. 에너지(Energy) 패턴 유사도
- **에너지 윤곽 추출 및 정규화**: 절대적 볼륨 차이를 보정합니다.
- **상관 계수**를 계산하여 에너지 패턴의 유사도를 측정합니다.

#### 1.4. 리듬(Rhythm) 패턴 유사도
- **온셋 감지**: 소리의 시작점을 식별하여 리듬 패턴을 추출합니다.
- **비트 개수 유사도** 및 **간격 패턴 유사도**를 종합합니다.

#### 1.5. 모음(Vowel) 길이 패턴 유사도
- **모음 식별 및 길이 측정**: 각 언어별 모음 목록을 사용하여 분석합니다.
- **개수 유사도** 및 **길이 패턴 유사도**를 종합합니다.

### 2. 세그먼트 정렬 및 비교

텍스트와 시간 정보를 활용하여 원본-합성 세그먼트를 최적으로 정렬합니다:

- **텍스트 기반 매칭**: 다국어 임베딩 모델을 사용한 텍스트 유사도 계산
- **시간 기반 매칭**: 세그먼트 간 시간 중첩도 계산
- **N:M 매핑**: 필요한 경우 여러 세그먼트를 묶어 매칭

### 3. 종합 평가

다양한 유사도 점수를 종합하여 최종 평가를 수행합니다:

- **가중 평균 계산**: 각 특성의 중요도에 따라 가중치 적용
- **등급 부여**: A+, A, B, C, D, F 등급으로 분류
- **개선 제안 생성**: 낮은 점수의 특성에 대한 구체적 개선 방안 제시

## 출력 결과

분석 결과는 지정된 출력 디렉토리에 다음과 같은 형식으로 저장됩니다:

- **summary_report.txt**: 전체 평가 요약 보고서
- **similarity_radar.png**: 유사도 특성을 시각화한 레이더 차트
- **segment_alignment_with_missing.png**: 세그먼트 정렬 시각화
- **alignment/segment_alignment.json**: 세그먼트 정렬 결과
- **tts_output.ssml**: 정렬 결과를 기반으로 생성된 SSML 파일
- **audio_comparison.png**: 원본-합성 오디오 파형 비교

## 프로젝트 구조

```
./
│
├── main.py                    # 메인 실행 파일 
├── config.py                  # 설정 파일
│
├── core/                      # 핵심 기능 모듈
│   ├── __init__.py
│   ├── processor.py           # TextGrid 및 오디오 처리 모듈
│   ├── aligner.py             # 프로소딕 정렬 모듈
│   ├── analyzer.py            # 음성 유사도 분석 모듈
│   ├── evaluator.py           # 최종 평가 및 점수 계산 모듈
│   └── alignment_to_ssml.py   # 정렬 결과를 SSML로 변환하는 모듈
│
├── utils/                     # 유틸리티 기능
│   ├── __init__.py
│   ├── audio_utils.py         # 오디오 처리 유틸리티
│   ├── text_utils.py          # 텍스트 처리 유틸리티 
│   └── visualizer.py          # 결과 시각화 모듈
│
├── example_single.py          # 단일 파일 분석 예제
├── example_batch.py           # 일괄 분석 예제
│
└── data/                      # 입출력 데이터 디렉토리
    ├── input/                 # 입력 파일 디렉토리 (원본/합성 음성, TextGrid)
    └── output/                # 출력 파일 디렉토리 (평가 결과, 시각화)
```

## 설정 옵션

`config.py` 파일에서 다양한 설정을 조정할 수 있습니다:

- **feature_weights**: 각 프로소딕 특성의 가중치 (휴지, 음높이, 에너지, 리듬, 모음)
- **alignment_weights**: 정렬 특성의 가중치 (텍스트 유사도, 시간적 유사도, 발화 속도 유사도)
- **final_score_weights**: 최종 점수 계산 가중치 (프로소디 점수, 정렬 점수)
- **language_speed_coefficients**: 언어별 발화 속도 계수
- **similarity_threshold**: 의미 유사도 임계값

## 학술적 배경

이 프로젝트는 다음 연구 논문들에 기반하여 개발되었습니다:

1. Virkar, Y., Federico, M., Enyedi, R., & Barra-Chicote, R. (2021). **"Improvements to Prosodic Alignment for Automatic Dubbing"**. IEEE ICASSP, pp. 7543-7574.
2. Virkar, Y., Federico, M., Enyedi, R., & Barra-Chicote, R. (2022). **"Prosodic Alignment for Off-Screen Automatic Dubbing"**. arXiv:2204.02530.
3. Öktem, A., Farrús, M., & Bonafonte, A. (2019). **"Prosodic Phrase Alignment for Machine Dubbing"**. Interspeech 2019.