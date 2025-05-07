import os
import pathlib
import matplotlib.pyplot as plt
import matplotlib
from textgrid import TextGrid
import subprocess

# IPA 문자(발음 기호) 지원을 위해 DejaVu Sans 폰트 사용
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic']  # 폰트 설정
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# TextGrid 파일에서 음소(phoneme) 정보를 추출하는 함수
def extract_phonemes(textgrid_path):
    tg = TextGrid.fromFile(textgrid_path)  # TextGrid 파일 로드
    # "phones"라는 이름의 tier를 찾음
    phone_tier = [tier for tier in tg.tiers if tier.name.lower() == "phones"][0]

    phonemes = []  # 음소 리스트 초기화
    # 각 음소에 대해 시작시간과 끝시간을 추출
    for interval in phone_tier.intervals:
        label = interval.mark.strip()  # 음소 라벨
        if label == "":  # 음소가 비어 있으면 건너뜀
            continue
        phonemes.append({
            "phoneme": label,
            "start": interval.minTime,  # 시작 시간
            "end": interval.maxTime  # 끝 시간
        })
    return phonemes

# 텍스트 파일에서 전사(Transcript)를 읽어오는 함수
def read_transcript(txt_path):
    if not os.path.exists(txt_path):  # 파일이 없으면 기본 메시지 반환
        return "(No transcript found)"
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read().strip()

# 음소 정렬 결과를 시각화하는 함수
def plot_alignment(phonemes, transcript, output_path, title=None):
    fig, ax = plt.subplots(figsize=(12, 3.5))  # 조금 더 높은 크기의 그래프 생성

    # 음소 바 그리기
    y = 0.5
    for ph in phonemes:
        x_start = ph["start"]
        width = ph["end"] - ph["start"]
        ax.barh(y, width, left=x_start, height=0.3, edgecolor='black', color='skyblue')
        ax.text(x_start + width / 2, y, ph["phoneme"], ha='center', va='center', fontsize=10)

    # 축 설정
    ax.set_xlabel("Time (s)")  # x축: 시간
    ax.set_yticks([])  # y축은 필요 없으므로 비움
    ax.set_xlim(phonemes[0]["start"], phonemes[-1]["end"])  # x축 범위 설정

    # 제목 설정
    fig.suptitle(f"{title}", fontsize=14, y=0.95, ha='left', x=0.01)

    # 전사 내용 추가
    ax.set_title(f"Transcript: {transcript}", fontsize=11, loc='left', pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.88])  # 제목 공간 조정
    plt.savefig(output_path)  # 파일로 저장
    plt.close()

# MFA(Montreal Forced Aligner) 명령어 실행 함수
def run_mfa(corpus_dir, dictionary_path, acoustic_model_path, output_dir):
    """
    주어진 경로에서 MFA를 실행하여 음성 정렬을 수행하는 함수.
    
    Parameters:
    - corpus_dir: 음성 파일과 텍스트 파일이 있는 디렉토리 경로.
    - dictionary_path: 사전(dictionary) 경로.
    - acoustic_model_path: 사전 훈련된 음향 모델 경로 (예: en_us_arpa.tar.gz).
    - output_dir: 결과를 저장할 출력 디렉토리 경로.
    
    Returns:
    - None
    """
    # MFA 명령어 구성
    mfa_command = [
        "mfa", "align", 
        corpus_dir,  # 음성 파일과 텍스트 파일이 포함된 디렉토리
        dictionary_path,  # 사전 파일 경로
        acoustic_model_path,  # 음향 모델 경로
        output_dir  # 결과를 저장할 디렉토리
    ]
    
    # MFA 실행
    try:
        subprocess.run(mfa_command, check=True)
        print("MFA alignment completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

# TextGrid 파일을 읽고 시각화하는 함수
def batch_visualize(tg_path, txt_root, output_dir):
    base = tg_path.stem  # 파일 이름에서 확장자를 제거한 부분
    txt_path = f"{txt_root}/{base}.txt"  # 대응하는 텍스트 파일 경로
    print(f"txt_path: {txt_path}")  # 처리 중인 파일 경로 출력
    output_path = os.path.join(output_dir, f"{base}.png")  # 출력 이미지 경로

    phonemes = extract_phonemes(tg_path)  # TextGrid에서 음소 추출
    transcript = read_transcript(txt_path)  # 전사 텍스트 읽기
    plot_alignment(phonemes, transcript, output_path, title=base)  # 음소 정렬 시각화
    print(f"Saved: {output_path}")

# MFA 결과 디렉토리에서 모든 TextGrid 파일을 처리하고 시각화하는 함수
def process_mfa_results(mfa_output_dir, txt_root, output_dir):
    textgrid_files = list(pathlib.Path(mfa_output_dir).rglob("*.TextGrid"))  # 모든 TextGrid 파일 찾기
    for idx, tg_file in enumerate(textgrid_files):
        print(f"Processing {idx + 1}/{len(textgrid_files)}: {tg_file}")
        os.makedirs(output_dir, exist_ok=True)  # 출력 디렉토리가 없으면 생성
        batch_visualize(tg_file, txt_root, output_dir)  # 시각화 작업

if __name__ == "__main__":
    # MFA 실행을 위한 경로 설정
    corpus_directory = "../data/input/kr"  # 음성 및 텍스트 파일이 포함된 디렉토리
    dictionary_path = "C:/Users/SSAFY/Documents/MFA/pretrained_models/dictionary/korean_mfa.dict"  # 사전 파일 경로
    acoustic_model_path = "C:/Users/SSAFY/Documents/MFA/pretrained_models/acoustic/korean_mfa.zip"  # 음향 모델 경로
    output_directory = "../data/input/text_grid"  # 결과를 저장할 디렉토리 경로

    # MFA 실행: 음성 정렬
    run_mfa(corpus_directory, dictionary_path, acoustic_model_path, output_directory)

    # MFA 결과를 시각화: TextGrid 파일을 읽고 시각화 생성
    process_mfa_results(output_directory, corpus_directory, "../data/input/text_grid/alignment_plots")
