## 환경 세팅
**Anaconda를 통한 설치**
```bash
conda config --add channels conda-forge
conda install montreal-forced-aligner
```
**사전 훈련된 Dictionary 및 Acoustic Model 다운로드**
```bash
mfa model download dictionary english_mfa
mfa model download acoustic english_mfa

mfa model download dictionary korean_mfa
mfa model download acoustic korean_mfa
```
## 사용법
코드 실행: run_mfa 함수는 mfa align 명령어를 실행하여 음성 및 텍스트 파일을 정렬하고, process_mfa_results 함수는 TextGrid 파일을 읽고 시각화된 결과를 출력합니다.


**전체 코드 실행 예시:**
``` python
if __name__ == "__main__":
    # 경로 설정
    corpus_directory = "path/to/audio_and_text"  # 음성 및 텍스트 파일이 포함된 디렉토리
    dictionary_path = "path/to/dictionary.dict"  # 사전 파일 경로
    acoustic_model_path = "path/to/acoustic_model.zip"  # 음향 모델 경로
    output_directory = "path/to/output_textgrid"  # MFA 결과 저장 경로

    # MFA 실행: 음성 정렬
    run_mfa(corpus_directory, dictionary_path, acoustic_model_path, output_directory)

    # MFA 결과를 시각화: TextGrid 파일을 읽고 시각화 생성
    process_mfa_results(output_directory, corpus_directory, "path/to/alignment_plots")

```
### 파일 경로 예시

음성 및 텍스트 파일 디렉토리:

`corpus_directory = "../input/en"`

여기에는 audio1.wav, audio1.txt와 같은 파일들이 포함됩니다.

사전 파일 경로:

`dictionary_path = "C:/Users/SSAFY/Documents/MFA/pretrained_models/dictionary/english_mfa.dict"`

음향 모델 경로:

`acoustic_model_path = "C:/Users/SSAFY/Documents/MFA/pretrained_models/acoustic/english_mfa.zip"`

출력 디렉토리 (TextGrid 저장):

`output_directory = "../input/text_grid"`

시각화 결과 저장 디렉토리:

`alignment_plots = "../input/text_grid/alignment_plots"`