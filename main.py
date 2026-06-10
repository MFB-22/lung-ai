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
import traceback

# ==========================================
# 1. 오픈소스 뼈 억제 AI 아키텍처 (ResNet Generator)
# (이 부분은 다운로드한 오픈소스의 모델 클래스 코드로 완벽히 동일하게 맞춰야 합니다)
# 아래는 가장 범용적으로 쓰이는 9-block ResNet 기반 모델의 표준 예시입니다.
# ==========================================
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class OpenSourceBoneSuppressionNet(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_blocks=9):
        super(OpenSourceBoneSuppressionNet, self).__init__()
        
        # 1. 초기 컨볼루션
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]
        
        # 2. 다운샘플링 (인코더)
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]
            
        # 3. ResNet 블록 (병목 구간)
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]
            
        # 4. 업샘플링 (디코더)
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
            
        # 5. 최종 출력
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh() # 뼈 억제 오픈소스는 주로 -1 ~ 1 사이로 출력합니다.
        ]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# ==========================================
# 2. FastAPI 앱 및 AI 모델 로드
# ==========================================
app = FastAPI(title="Lung-Ai Preprocessing API (OpenSource BS)")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 [Lung-Ai Server] 모델을 {device}에 로드하는 중...")

# (선택) 폐 영역 분할용 모델
seg_model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()

# 💡 최신 오픈소스 모델 로드
bone_model = OpenSourceBoneSuppressionNet(input_nc=1, output_nc=1).to(device)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 💡 다운로드한 오픈소스 가중치 파일 이름
bone_model_path = os.path.join(BASE_DIR, "resnet_bs_model.pth") 

if os.path.exists(bone_model_path):
    try:
        bone_model.load_state_dict(torch.load(bone_model_path, map_location=device))
        print("✅ [Lung-Ai Server] 글로벌 오픈소스 뼈 제거 AI 장착 완료!")
    except Exception as e:
        print(f"⚠️ [에러] 가중치 구조가 맞지 않습니다. 다운로드한 오픈소스의 원본 파이썬 클래스 구조로 1번 항목을 교체해주세요.\n{e}")
else:
    print(f"⚠️ [경고] {bone_model_path} 파일이 없습니다. GitHub에서 가중치를 다운받아 넣어주세요!")

bone_model.eval()

# ==========================================
# 3. 통합 전처리 엔드포인트
# ==========================================
@app.post("/api/preprocess")
async def run_lung_ai(file: UploadFile = File(...)):
    filename = file.filename if file.filename else "Unknown"
    
    try:
        print(f"\n--- 🚀 [새로운 요청 수신] 파일명: {filename} ---")
        contents = await file.read()
        
        # [STEP 1] 디코딩
        if filename.lower().endswith('.dcm'):
            dicom_data = pydicom.dcmread(io.BytesIO(contents))
            pixel_array = dicom_data.pixel_array
            raw_img = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if len(raw_img.shape) == 3:
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        else:
            nparr = np.frombuffer(contents, np.uint8)
            raw_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # [STEP 2] 모델 입력용 텐서 변환
        img_512 = cv2.resize(raw_img, (512, 512))
        
        # 💡 Tanh() 출력을 가지는 모델을 위한 -1 ~ 1 정규화
        img_normalized = (img_512.astype(np.float32) / 127.5) - 1.0 
        bone_input = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).to(device)

        # [STEP 3] 오픈소스 AI 모델 추론
        print("[3/5] 오픈소스 AI 모델 연산 진행 중...")
        with torch.no_grad():
            # 뼈 억제 수행
            bone_out_tensor = bone_model(bone_input)
            
            # -1 ~ 1 결과를 다시 0 ~ 255 이미지로 복원
            bone_out_np = bone_out_tensor.cpu().numpy()[0, 0, :, :]
            bone_out_np = ((bone_out_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        # [STEP 4] 메모리 정리 및 전송
        print("[4/5] 처리 완료! 프론트엔드로 PNG 변환 전송...")
        del bone_input, bone_out_tensor
        torch.cuda.empty_cache()
        gc.collect()

        _, encoded_img = cv2.imencode('.png', bone_out_np)
        return Response(content=encoded_img.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"\n❌ [에러 발생] {filename} 처리 실패\n")
        traceback.print_exc()
        return Response(status_code=500, content=f"Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)