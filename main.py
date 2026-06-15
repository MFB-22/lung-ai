from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import cv2
import numpy as np
import torch
import pydicom
import io
import os
import gc
import traceback
from RajaramanModel import ResNet_BS

# ==========================================
# 🚨 [가장 중요] 깃허브 모델 아키텍처 불러오기
# ==========================================
# RajaramanModel.py 파일을 열어보시면 `class 클래스이름(nn.Module):` 형태로 적혀있을 것입니다.
# 그 클래스 이름을 아래에 정확히 적어주세요. (예: ResNet_BS, ResnetGenerator 등)
try:
    from RajaramanModel import ResNet  # 👈 'ResNet' 부분을 실제 클래스 이름으로 바꿔주세요!
except ImportError:
    print("⚠️ RajaramanModel.py 파일이 없거나 클래스 이름을 찾을 수 없습니다.")


# ==========================================
# 1. FastAPI 앱 및 파이토치 모델 초기화
# ==========================================
app = FastAPI(title="Lung-Ai Preprocessing API (PyTorch OpenSource)")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 [Lung-Ai Server] 모델을 {device}에 로드하는 중...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 💡 다운받으신 .tar 가중치 파일
bone_model_path = os.path.join(BASE_DIR, "trained_network_nonEqualised.tar")


try:
    # 1. 모델 뼈대 생성 (클래스 이름과 필수 이미지 사이즈 파라미터 추가!)
    bone_model = ResNet_BS(input_array_shape=(1, 256, 256)).to(device)
    
    # 2. .tar 가중치 보따리 읽어오기
    checkpoint = torch.load(bone_model_path, map_location=device)
    
    # 3. 보따리 구조에 맞게 가중치 덮어씌우기 (에러 방지용 다중 조건문)
    if 'state_dict' in checkpoint:
        bone_model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        bone_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        bone_model.load_state_dict(checkpoint) # 가중치만 덩그러니 있을 경우
        
    bone_model.eval()
    print("✅ [Lung-Ai Server] 글로벌 파이토치 뼈 제거 AI 장착 완벽 성공!")
    
except Exception as e:
    print(f"⚠️ [치명적 에러] 가중치 로드 실패! RajaramanModel.py 내부의 클래스명을 확인해주세요.\n{e}")
    bone_model = None

# ==========================================
# 2. 통합 전처리 AI 엔드포인트
# ==========================================
@app.post("/api/preprocess")
async def run_lung_ai(file: UploadFile = File(...)):
    filename = file.filename if file.filename else "Unknown"
    
    try:
        print(f"\n--- 🚀 [새로운 요청 수신] 파일명: {filename} ---")
        contents = await file.read()
        
        # [STEP 1] 디코딩 및 흉부 X-ray(CXR) 여부 엄격 검증
        dicom_data = pydicom.dcmread(io.BytesIO(contents))
        
        # 🚨 [핵심 추가] DICOM 메타데이터 추출
        # 1. Modality: 촬영 장비 종류 (CR: 컴퓨터 방사선, DX: 디지털 방사선 등)
        modality = getattr(dicom_data, 'Modality', '').upper()
        
        # 2. BodyPartExamined: 촬영 부위 (CHEST, KNEE, HEAD 등)
        body_part = getattr(dicom_data, 'BodyPartExamined', '').upper()
        
        print(f"👉 [데이터 확인] 장비: {modality}, 부위: {body_part}")
        
        # 🛑 검증 1: X-ray 촬영(CR, DX, XR)이 아닌 경우 (예: CT, MR, US 등 거부)
        if modality not in ['CR', 'DX', 'XR']:
            print(f"⚠️ [처리 중단] X-ray 영상이 아닙니다. (입력된 장비: {modality})")
            return Response(status_code=400, content=f"Only X-ray images are supported. Detected Modality: {modality}")
            
        # 🛑 검증 2: 촬영 부위가 '가슴(CHEST)'이 아닌 경우 (단, 태그가 아예 누락된 빈 값인 경우는 통과시킴)
        if body_part and 'CHEST' not in body_part and 'LUNG' not in body_part:
            print(f"⚠️ [처리 중단] 흉부(Chest) 영상이 아닙니다. (입력된 부위: {body_part})")
            return Response(status_code=400, content=f"Only Chest X-rays are supported. Detected Body Part: {body_part}")

        # 검증을 통과한 순수 흉부 X-ray만 픽셀 데이터 추출 진행
        pixel_array = dicom_data.pixel_array
        raw_img = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if len(raw_img.shape) == 3:
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        # [STEP 2] 텐서 변환 (모델 입력용)
        # 해당 오픈소스가 통상적으로 요구하는 256x256 크기로 리사이즈
        img_resized = cv2.resize(raw_img, (256, 256))
        
        # 0 ~ 1 사이로 정규화 후 PyTorch 텐서로 변환 [Batch, Channel, H, W]
        img_normalized = img_resized.astype(np.float32) / 255.0
        bone_input = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).to(device)

        # [STEP 3] AI 모델 추론 (진짜 뼈 지우기)
        print("[3/5] 파이토치 뼈 억제 AI 연산 진행 중...")
        with torch.no_grad():
            if bone_model is not None:
                bone_out_tensor = bone_model(bone_input)
                
                # 텐서를 넘파이 배열로 변환
                bone_out_np = bone_out_tensor.cpu().numpy()[0, 0, :, :]
                
                # 결과값이 -1~1 (Tanh) 인지, 0~1 (Sigmoid) 인지 자동 대응하여 0~255로 복원
                if bone_out_np.min() < 0:
                    bone_out_np = ((bone_out_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                else:
                    bone_out_np = (bone_out_np * 255.0).clip(0, 255).astype(np.uint8)
            else:
                raise ValueError("AI 모델이 정상적으로 로드되지 않았습니다.")

       # [STEP 4] 후처리 및 이미지 전송
        print("[4/5] 처리 완료! 고화질 복원 및 명암 조절(CLAHE) 진행...")
        
        # 1. 프론트엔드 출력을 위해 다시 512x512 해상도로 깨끗하게 키워줍니다.
        final_processed_img = cv2.resize(bone_out_np, (512, 512), interpolation=cv2.INTER_CUBIC)

        # 💡 2. [핵심 추가] 기존에 사용했던 CLAHE 명암비 극대화 로직 적용
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        final_processed_img = clahe.apply(final_processed_img)

        del bone_input, bone_out_tensor
        torch.cuda.empty_cache()
        gc.collect()

        _, encoded_img = cv2.imencode('.png', final_processed_img)
        return Response(content=encoded_img.tobytes(), media_type="image/png")
    
    except Exception as e:
        print(f"\n❌ [에러 발생] {filename} 처리 실패\n")
        traceback.print_exc()
        return Response(status_code=500, content=f"Server Error: {str(e)}")

# ==========================================
# 3. DICOM 미리보기 전용 API
# ==========================================
@app.post("/api/preview")
async def get_dicom_preview(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.dcm'):
        return Response(status_code=400, content="Only DICOM files need preview.")
    
    contents = await file.read()
    try:
        dicom_data = pydicom.dcmread(io.BytesIO(contents))
        pixel_array = dicom_data.pixel_array
        raw_img = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if len(raw_img.shape) == 3:
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            
        _, encoded_img = cv2.imencode('.png', raw_img)
        return Response(content=encoded_img.tobytes(), media_type="image/png")
        
    except Exception as e:
        return Response(status_code=500, content=f"Preview Error: {str(e)}")


# ==========================================
# 4. 재생 버튼(▶) 클릭 시 서버 자동 실행
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)