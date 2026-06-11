# 🫁 Lung-Ai Pro: 흉부 X-ray 뼈 억제 및 통합 진단 보조 시스템

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)

**Lung-Ai Pro**는 흉부 X-ray(CXR) 판독 시 갈비뼈와 쇄골 등 복잡한 뼈 구조물에 가려져 미세 병변(결절, 결핵 등)을 놓치는 문제를 해결하기 위해 개발된 **딥러닝 기반 뼈 억제(Bone Suppression) 전처리 및 질병 분류 파이프라인**입니다. 

글로벌 오픈소스(ResNet_BS)를 타겟 데이터에 맞춰 512x512 고해상도로 파인튜닝(Fine-Tuning)하였으며, 독자적인 CLAHE 후처리와 100% 알파 블렌딩 기술을 적용하여 진단의 민감도(Recall)를 극대화했습니다.

---

## ✨ 핵심 기능 (Key Features)

* **🚀 512x512 Native 뼈 억제 추론 (Fine-tuned ResNet_BS):** * 이미지 축소(Downsampling) 없이 512x512 고해상도 원본을 그대로 처리하여 미세 병변의 픽셀 손실을 방지합니다. (JSRT & BSE-JSRT 데이터셋 기반 Transfer Learning 적용)
* **🧠 정밀한 폐 분할 및 100% 알파 블렌딩:** * `PSPNet`을 활용해 폐 영역(ROI)만 정확히 추출한 뒤, 뼈가 억제된 결과물을 폐 내부 영역에만 **100% 가중치**로 부드럽게 합성하여 OOD(Out-of-Distribution) 에러를 방지합니다.
* **💡 CLAHE 명암비 최적화:** * 뼈 억제 이후 흐려지는 연조직(Soft Tissue)의 명암비를 국소적으로 끌어올려 질병 분류 모델(`DenseNet121`)의 탐지력을 향상시킵니다.
* **📊 듀얼 스트림(Dual-Stream) A/B 테스트 검증:** * '원본 단독 판독' vs 'Lung-Ai 전처리 통합 판독' 성능을 실시간으로 비교하고, 그 결과를 엑셀(Excel) 리포트로 자동 추출합니다.
* **🌐 MLOps 통합 웹 서비스:** * React(Frontend)와 FastAPI(Backend) 기반으로 DICOM 파일 디코딩 및 실시간 전처리 듀얼 뷰어 렌더링을 지원합니다.

---

## 🏗️ 시스템 아키텍처 (System Architecture)

```mermaid
graph TD
    classDef startEnd fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef router fill:#e6f2ff,stroke:#4A90E2,stroke-width:2px;
    classDef process fill:#ffffff,stroke:#4A90E2,stroke-width:2px;
    classDef updated fill:#fff3e6,stroke:#F5A623,stroke-width:2px;

    Start([Upload: DCM / PNG / JPG]):::startEnd --> Router{FastAPI Router}:::router
    
    %% Preview
    Router -->|POST /api/preview| Prev1[pydicom.dcmread]:::process
    Prev1 --> Prev2[cv2.imencode .png]:::process
    Prev2 --> PrevEnd([Return Preview PNG]):::startEnd

    %% Preprocess
    Router -->|POST /api/preprocess| Prep1[run_lung_ai]:::process
    Prep1 --> Prep2[Tensor Transformation: 512x512]:::process
    Prep2 --> Prep3[seg_model: PSPNet]:::process
    Prep3 --> Prep4[Mask Refinement: m_clean]:::process
    
    Prep4 -->|cv2.findContours & dilate| Prep5[bone_model: ResNet_BS 512x512]:::updated
    Prep5 -->|lumora_bone_model_512_finetuned.pth| Prep6[CLAHE Contrast Optimization]:::updated
    Prep6 -->|cv2.createCLAHE| Prep7[100% Alpha Blending]:::updated
    Prep7 -->|cv2.GaussianBlur| Prep8[Resource Cleanup: cuda.empty_cache]:::process
    
    Prep8 --> Prep9[cv2.imencode .png]:::process
    Prep9 --> PrepEnd([Return Processed PNG]):::startEnd
