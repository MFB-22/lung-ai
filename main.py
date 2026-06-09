from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchxrayvision as xrv
import pydicom
import io
import os
import gc

# ==========================================
# 1. 커스텀 뼈 제거 AI 아키텍처 (U-Net)
# ==========================================
class LightweightUNet(nn.Module):
    def __init__(self):
        super(LightweightUNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(16))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.final = nn.Sequential(nn.Conv2d(16, 1, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(self.up2(b))
        d1 = self.dec1(self.up1(d2))
        return self.final(d1)

def match_mean_std(source, template):
    s_mean, s_std = source.mean(), source.std()
    t_mean, t_std = template.mean(), template.std()
    matched = (source - s_mean) * (t_std / (s_std + 1e-8)) + t_mean
    return np.clip(matched, 0, 255).astype(np.uint8)

# ==========================================
# 2. FastAPI 앱 및 AI 모델 초기화
# ==========================================
app = FastAPI(title="Lung-Ai Preprocessing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 [Lung-Ai Server] 모델을 {device}에 로드하는 중...")

seg_model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
bone_model = LightweightUNet().to(device)
bone_model_path = "lumora_bone_model.pth"

if os.path.exists(bone_model_path):
    bone_model.load_state_dict(torch.load(bone_model_path, map_location=device))
    print("✅ [Lung-Ai Server] 뼈 제거 AI 장착 완료!")
else:
    print("⚠️ [주의] lumora_bone_model.pth 파일이 없습니다.")
bone_model.eval()

# ==========================================
# 3. 실제 API 엔드포인트
# ==========================================
import traceback

@app.post("/api/preprocess")
async def run_lung_ai(file: UploadFile = File(...)):
    try:
        print(f"\n--- 🚀 [새로운 요청 수신] 파일명: {file.filename} ---")
        contents = await file.read()
        filename = file.filename.lower()
        
        # [STEP 1] 디코딩
        print("[1/5] 파일 디코딩 및 변환 중...")
        if filename.endswith('.dcm'):
            dicom_data = pydicom.dcmread(io.BytesIO(contents))
            pixel_array = dicom_data.pixel_array
            raw_img = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if len(raw_img.shape) == 3:
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        else:
            nparr = np.frombuffer(contents, np.uint8)
            raw_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if raw_img is None:
            raise ValueError("cv2.imdecode 또는 DICOM 변환 결과가 None입니다.")

        # [STEP 2] 텐서 변환
        print("[2/5] 512x512 리사이즈 및 PyTorch 텐서 변환 중...")
        img_512 = cv2.resize(raw_img, (512, 512))
        img_512_norm = xrv.datasets.normalize(img_512, 255)
        img_tensor = torch.from_numpy(img_512_norm).float().unsqueeze(0).unsqueeze(0).to(device)

        # [STEP 3] AI 모델 추론
        print(f"[3/5] AI 모델 연산 진행 중 (현재 메모리: {device})...")
        with torch.no_grad():
            preds_seg = seg_model(img_tensor)
            left_lung = torch.sigmoid(preds_seg[:, 4, :, :]) > 0.7
            right_lung = torch.sigmoid(preds_seg[:, 5, :, :]) > 0.7
            mask_np = ((left_lung | right_lung).float().cpu().numpy()[0] * 255).astype(np.uint8)

            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            m_clean = np.zeros_like(mask_np)
            if len(contours) > 0:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                cv2.drawContours(m_clean, contours[:2], -1, 255, thickness=cv2.FILLED)
                m_clean = cv2.dilate(m_clean, np.ones((15, 15), np.uint8), iterations=1)

            seg_img_512 = cv2.bitwise_and(img_512, img_512, mask=m_clean)
            img_224 = cv2.resize(seg_img_512, (224, 224))
            bone_input = torch.from_numpy(img_224).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

            bone_out_224 = (bone_model(bone_input).cpu().numpy()[0, 0, :, :] * 255).astype(np.uint8)

        # [STEP 4] 후처리 및 병합
        print("[4/5] 후처리 및 이미지 블렌딩 중...")
        bone_out_512 = cv2.resize(bone_out_224, (512, 512))
        blended = cv2.addWeighted(bone_out_512, 0.3, img_512, 0.7, 0)
        
        alpha = cv2.GaussianBlur(m_clean, (31, 31), 0).astype(float) / 255.0
        final_img = (alpha * blended + (1 - alpha) * img_512).astype(np.uint8)
        final_processed_img = match_mean_std(final_img, img_512)

        # [STEP 5] 메모리 정리 및 전송
        print("[5/5] 처리 완료! 프론트엔드로 PNG 변환 전송...")
        del img_tensor, preds_seg, bone_input
        torch.cuda.empty_cache()
        gc.collect()

        _, encoded_img = cv2.imencode('.png', final_processed_img)
        return Response(content=encoded_img.tobytes(), media_type="image/png")

    except Exception as e:
        # 💡 여기가 핵심입니다! 서버가 죽은 이유를 강제로 터미널에 출력합니다.
        print(f"\n❌ [치명적 에러 발생] {filename} 처리 중 실패했습니다.")
        print("=" * 50)
        traceback.print_exc()
        print("=" * 50)
        return Response(status_code=500, content=f"Server Error: {str(e)}")

    del img_tensor, preds_seg, bone_input
    torch.cuda.empty_cache()
    gc.collect()

    _, encoded_img = cv2.imencode('.png', final_processed_img)
    return Response(content=encoded_img.tobytes(), media_type="image/png")
# --- (기존 코드 유지) ---

# ==========================================
# 4. DICOM 미리보기 전용 API (새로 추가)
# ==========================================
@app.post("/api/preview")
async def get_dicom_preview(file: UploadFile = File(...)):
    # 오직 DICOM 파일만 PNG 썸네일로 변환해서 돌려줍니다.
    if not file.filename.lower().endswith('.dcm'):
        return Response(status_code=400, content="Only DICOM files need preview.")
    
    contents = await file.read()
    try:
        dicom_data = pydicom.dcmread(io.BytesIO(contents))
        pixel_array = dicom_data.pixel_array
        # 0~255 PNG용 픽셀로 변환
        raw_img = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if len(raw_img.shape) == 3:
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            
        _, encoded_img = cv2.imencode('.png', raw_img)
        return Response(content=encoded_img.tobytes(), media_type="image/png")
        
    except Exception as e:
        return Response(status_code=500, content=f"Preview Error: {str(e)}")