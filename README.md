# 🫁 Lung-Ai Pro: 흉부 X-ray 뼈 억제 및 통합 진단 보조 시스템

Lung-Ai Pro는 흉부 X-ray(CXR) 판독 시 갈비뼈와 쇄골 등 복잡한 뼈 구조물에 가려져 미세 병변(결절, 결핵 등)을 놓치는 문제를 해결하기 위해 개발된 딥러닝 기반 뼈 억제(Bone Suppression) 전처리 및 질병 분류 파이프라인입니다. 

글로벌 오픈소스(ResNet_BS)를 타겟 데이터에 맞춰 512x512 고해상도로 파인튜닝(Fine-Tuning)하였으며, 독자적인 CLAHE 후처리와 100% 알파 블렌딩 기술을 적용하여 진단의 민감도(Recall)를 극대화했습니다.

---

## ✨ 핵심 기능 (Key Features)

* **🚀 512x512 Native 뼈 억제 추론:** 이미지 축소(Downsampling) 없이 512x512 고해상도 원본을 그대로 처리하여 미세 병변의 픽셀 손실을 방지합니다.
* **🧠 정밀한 폐 분할 및 100% 알파 블렌딩:** `PSPNet`을 활용해 폐 영역(ROI)만 정확히 추출한 뒤, 뼈가 억제된 결과물을 폐 내부 영역에만 100% 가중치로 부드럽게 합성하여 판독 AI의 오작동(OOD)을 방지합니다.
* **💡 CLAHE 명암비 최적화:** 뼈 억제 이후 흐려지는 연조직(Soft Tissue)의 명암비를 국소적으로 끌어올려 질병 분류 모델의 탐지력을 향상시킵니다.
* **📊 듀얼 스트림(Dual-Stream) A/B 테스트:** '원본 단독 판독' vs 'Lung-Ai 전처리 통합 판독' 성능을 실시간으로 비교하고, 그 결과를 엑셀 리포트로 자동 추출합니다.
* **🌐 MLOps 통합 웹 서비스:** React(Frontend)와 FastAPI(Backend) 기반으로 DICOM 파일 디코딩 및 실시간 전처리 듀얼 뷰어 렌더링을 지원합니다.

---

## 📂 디렉토리 구조 (Directory Structure)

```text
Lung-Ai-Pro/
├── lung-ai-web/             # React 기반 프론트엔드 코드
├── RajaramanModel.py        # 뼈 억제 AI 모델 아키텍처
├── bone_model_512_final.pth # 학습이 완료된 뼈 억제 파이토치 모델 가중치
├── main.py                  # FastAPI 기반 백엔드 구동 서버
├── train_512.py             # 모델 파인튜닝 학습 스크립트
├── start.bat                # 프론트엔드 및 백엔드 원클릭 실행 스크립트
└── testdrive.ipynb          # 주피터 노트북 테스트 환경
```

---

## 🛠️ 설치 및 실행 방법 (Getting Started)

본 프로젝트를 로컬 환경에서 실행하기 위해서는 Python(백엔드)과 Node.js(프론트엔드) 환경이 모두 필요합니다.

### 1. 사전 준비 (Prerequisites)
* **Python 3.9+** 이상
* **Node.js 18+** 이상

### 2. 저장소 복제 및 의존성 패키지 설치
터미널(또는 명령 프롬프트)을 열고 아래 명령어들을 순서대로 실행하여 프로젝트를 세팅합니다.

```bash
# 1. 저장소 클론 (복제)
git clone [https://github.com/본인깃허브아이디/Lung-Ai-Pro.git](https://github.com/본인깃허브아이디/Lung-Ai-Pro.git)
cd Lung-Ai-Pro

# 2. 파이썬(Backend) 라이브러리 설치
pip install torch torchvision fastapi uvicorn opencv-python pydicom python-multipart pandas

# 3. 프론트엔드(Frontend) 패키지 설치
cd lung-ai-web
npm install
cd ..
```

### 3. 프로젝트 실행 (Run)

**방법 A: 원클릭 자동 실행 (추천)**
프로젝트 폴더 내에 있는 `start.bat` 파일을 더블클릭하면 프론트엔드와 백엔드 서버가 동시에 실행됩니다.

**방법 B: 수동 실행**
각각의 터미널을 열어 프론트엔드와 백엔드를 따로 실행합니다.

* **Backend (FastAPI 서버 구동)**
```bash
  uvicorn main:app --reload --host 127.0.0.1 --port 8000
  ```
* **Frontend (React 개발 서버 구동)**
```bash
  cd lung-ai-web
  npm run dev
  ```

### 4. 서비스 접속
서버 실행이 완료되면, 웹 브라우저를 열고 `http://localhost:5173` (Vite 기본 포트)로 접속하여 Lung-Ai Pro 시스템을 이용하실 수 있습니다.
