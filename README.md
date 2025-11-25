# 비디오 분석 및 음악 생성 요청 중계 서버 (Video Analysis Intermediate Server)

이 프로젝트는 Next.js 프론트엔드로부터 비디오 파일을 받아 Google Gemini를 통해 분석하고, 그 결과를 n8n 웹훅으로 전달하는 FastAPI 기반의 중간 서버입니다.

## 📁 프로젝트 구조

```
video_analysis_server/
├── server.py            # FastAPI 서버 메인 파일
├── video_Processor.py   # Gemini 비디오 분석 로직
├── media_utils.py       # 비디오 다운샘플링 및 오디오 자막 추출 (FFmpeg, Whisper)
├── requirements.txt     # 의존성 패키지 목록
└── README.md            # 설명서
```

## 🚀 로컬 실행 방법 (Local Development)

### 1. 환경 설정

Python 3.8 이상 및 **FFmpeg**가 설치되어 있어야 합니다.

**FFmpeg 설치:**
- **Mac**: `brew install ffmpeg`
- **Ubuntu**: `sudo apt update && sudo apt install ffmpeg`
- **Windows**: [FFmpeg 다운로드](https://ffmpeg.org/download.html) 후 PATH 설정

```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

보안을 위해 API 키 등은 `.env` 파일에 저장하여 관리합니다.

1.  `.env.example` 파일을 복사하여 `.env` 파일을 생성합니다.
    ```bash
    cp .env.example .env
    ```
2.  `.env` 파일을 열어 실제 값을 입력합니다.
    ```ini
    GEMINI_API_KEY=your_gemini_api_key
    N8N_WEBHOOK_URL=your_n8n_webhook_url
    ```
    *주의: `.env` 파일은 Git에 커밋되지 않도록 `.gitignore`에 포함되어 있습니다.*

### 3. 서버 실행

```bash
# 기본 실행 (포트 8000)
uvicorn server:app --reload
```

## ☁️ 배포 가이드 (Deployment)

### Cloudtype 배포

1.  **GitHub 저장소 연결**: `UMC_mid_PreProcessing` 저장소를 Cloudtype에 연결합니다.
2.  **설정**:
    - **Language**: Python
    - **Version**: 3.9 이상
    - **Start Command**: `uvicorn server:app --host 0.0.0.0 --port 8000`
3.  **환경 변수 (Environment Variables)**:
    - Cloudtype 대시보드에서 `GEMINI_API_KEY`와 `N8N_WEBHOOK_URL`을 설정합니다.

### Docker 배포 (선택사항)

Dockerfile을 생성하여 컨테이너로 배포할 수도 있습니다.

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📡 API 명세

### 비디오 분석 요청

- **URL**: `/api/analyze-video`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Body**:
    - `file`: 비디오 파일 (Binary, max 100MB)

**응답 (Response):**

```json
{
  "message": "Video received and processing started",
  "filename": "video.mp4",
  "status": "processing"
}
```

*참고: 실제 분석 결과는 백그라운드 작업 후 n8n 웹훅으로 전송됩니다.*
