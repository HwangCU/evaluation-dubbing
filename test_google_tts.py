def test_google_tts():
    """Google Cloud TTS가 제대로 동작하는지 테스트합니다."""
    from tts import TextToSpeech
    import os
    
    # Google 자격 증명 확인
    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    print(f"현재 사용 중인 Google 자격 증명 경로: {cred_path}")
    
    # 자격 증명 파일 존재 확인
    if cred_path and os.path.exists(cred_path):
        print(f"자격 증명 파일이 존재합니다. 크기: {os.path.getsize(cred_path)} 바이트")
    else:
        print("경고: 자격 증명 파일이 존재하지 않습니다!")
    
    # TTS 엔진 초기화
    try:
        tts = TextToSpeech(
            lang="ko-KR",
            engine="google",
            voice_id="ko-KR-Standard-A",  # 한국어 여성 음성
            speaking_rate=1.0,
            pitch=0.0,
            volume=1.0
        )
        
        # 간단한 문장 합성 테스트
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        result = tts.synthesize(
            sentences=["안녕하세요. 이것은 Google Cloud TTS 테스트입니다."]
        )
        
        if result and result[0]['audio_path']:
            print(f"성공! 오디오가 생성되었습니다: {result[0]['audio_path']}")
            print(f"오디오 길이: {result[0]['duration']:.2f}초")
            return True
        else:
            print("오류: 오디오 생성에 실패했습니다.")
            return False
    except Exception as e:
        print(f"Google TTS 테스트 중 오류 발생: {e}")
        import traceback
        print(traceback.format_exc())
        return False

# 테스트 실행
if __name__ == "__main__":
    test_google_tts()