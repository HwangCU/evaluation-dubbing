import sys
print(sys.executable)

try:
    import textgrid
    print("textgrid 모듈 로드 성공!")
    print(f"모듈 위치: {textgrid.__file__}")
except ImportError as e:
    print(f"모듈 로드 실패: {e}")