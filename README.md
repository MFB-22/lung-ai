# 🫁 Lung-Ai Pro: SOTA CXR Bone Suppression System

> **AI 기반 흉부 X-ray(CXR) 폐 영역 분할 및 뼈 음영 억제 전처리 플랫폼** > 흉부 X-ray 판독 시 갈비뼈 및 쇄골 음영이 폐 내부 병변을 가리는 문제를 해결하기 위해, 듀얼 AI 네트워크를 활용하여 뼈 음영을 깔끔하게 억제하고 병변 가시성을 극대화하는 고속 전처리 시스템입니다.

<br/>

## 🌟 Key Features
- **🏥 의료 데이터(DICOM) 완벽 지원:** 일반 이미지(PNG/JPG)는 물론, 브라우저가 해독하지 못하는 `.dcm` 파일을 FastAPI 백엔드가 0.1초 만에 썸네일로 구워 프론트엔드로 실시간 스트리밍합니다.
- **📂 폴더 단위 일괄 업로드:** 수백 장의 환자 데이터를 한 번에 업로드하고 전처리 대기열을 관리할 수 있는 직관적인 React 대시보드를 제공합니다.
- **⚡ 듀얼 AI 파이프라인 (Dual-Stream):** `PSPNet`을 활용한 해부학적 폐 영역 분할과 자체 가중치(`lumora_bone_model.pth`)를 장착한 `Lightweight U-Net` 기반의 뼈 음영 억제 추론이 동시에 이루어집니다.
- **🎨 자연스러운 이미지 블렌딩:** 가우시안 블러 기반의 알파 채널 합성과 통계적 휘도 분포 일치화(`match_mean_std`) 로직을 통해 이질감 없는 고품질 전처리 결과를 보장합니다.

<br/>

## 🛠 Tech Stack
- **Frontend:** React.js, Vite, Tailwind CSS
- **Backend:** Python, FastAPI, Uvicorn
- **AI & Vision:** PyTorch, TorchXRayVision, OpenCV (cv2)

<br/>

## ⚙️ AI Core Processing Pipeline
본 시스템의 전처리 버튼을 클릭하는 순간, 백엔드 서버에서 아래의 5단계 파이프라인이 즉각적으로 구동됩니다.

1. **Image Decoding & Normalization:** DICOM 메타데이터 파싱 및 0~255 범위 8비트 정규화
2. **Tensor Transformation:** 모델 입력 규격에 맞춘 512x512 해상도 및 PyTorch 텐서 변환
3. **Lung Segmentation:** `PSPNet` 모델 구동을 통한 좌/우 폐 영역 마스킹 및 `cv2.findContours`를 통한 윤곽선 정제
4. **Bone Suppression:** 마스킹된 폐 영역 내부의 뼈 패턴을 `Lightweight U-Net` 모델로 추론 및 억제
5. **Alpha Blending & Match Mean/Std:** 뼈가 제거된 결과와 원본 간의 가우시안 소프트 블렌딩 및 휘도 보정 수행, GPU 가비지 컬렉션(RAM 정리) 후 PNG 스트리밍

<br/>

## 📊 System Architecture

```mermaid
graph TD
    %% 스타일 정의
    style InOut fill:#FFF3E6,stroke:#F5A623,stroke-width:2px;
    style Router fill:#E6F2FF,stroke:#4A90E2,stroke-width:2px;
    style CoreFunc fill:#ECECF0,stroke:#A0A0A0,stroke-width:2px;
    style AI fill:#E6FFE6,stroke:#22C55E,stroke-width:2px;

    Upload([upload_file: DCM / PNG / JPG]) --> Router{FastAPI Router}
    
    %% Preview Pipeline
    Router -->|POST /api/preview| Preview[pydicom.dcmread]
    Preview --> PreviewNorm[cv2.normalize]
    PreviewNorm --> PreviewEncode[cv2.imencode .png]
    PreviewEncode --> Out1([Return Preview PNG])
    
    %% Preprocessing Pipeline
    Router -->|POST /api/preprocess| Main[run_lung_ai]
    Main --> Decode[Image Decoding & Normalization]
    Decode --> Tensor[Tensor Transformation: 512x512]
    Tensor --> SegModel[seg_model: PSPNet]:::AI
    SegModel --> Mask[Mask Refinement: m_clean]
    Mask -->|cv2.findContours & dilate| BoneModel[bone_model: Lightweight U-Net]:::AI
    BoneModel -->|lumora_bone_model.pth| Blend[Alpha Blending]
    Blend -->|cv2.GaussianBlur & addWeighted| Match[match_mean_std]
    Match --> Cleanup[Resource Cleanup: cuda.empty_cache]
    Cleanup --> Encode[cv2.imencode .png]
    Encode --> Out2([Return Processed PNG])

    class Upload,Out1,Out2 InOut;
    class Router Router;
    class Main,Decode,Tensor,Mask,Blend,Match,Cleanup,Encode,Preview,PreviewNorm,PreviewEncode CoreFunc;