import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import glob
from RajaramanModel import ResNet_BS

# 1. 파인튜닝용 512x512 데이터셋 클래스
class BoneSuppressionDataset512(Dataset):
    def __init__(self, source_dir, target_dir):
        # source_dir: 원본 X-ray 폴더 / target_dir: 뼈 지워진 정답 폴더
        self.source_paths = sorted(glob.glob(os.path.join(source_dir, "*.png")))
        self.target_paths = sorted(glob.glob(os.path.join(target_dir, "*.png")))

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, idx):
        # 512x512 고정 리사이즈 및 정규화
        src_img = cv2.imread(self.source_paths[idx], cv2.IMREAD_GRAYSCALE)
        tgt_img = cv2.imread(self.target_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        src_img = cv2.resize(src_img, (512, 512)).astype('float32') / 255.0
        tgt_img = cv2.resize(tgt_img, (512, 512)).astype('float32') / 255.0
        
        return torch.tensor(src_img).unsqueeze(0), torch.tensor(tgt_img).unsqueeze(0)

# 2. 파인튜닝 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 💡 모델을 512 해상도로 생성
model = ResNet_BS(input_array_shape=(1, 512, 512)).to(device)

# 💡 현재 파이썬 파일이 있는 폴더 위치를 자동으로 찾아서 절대 경로 생성
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tar_path = os.path.join(BASE_DIR, "trained_network_nonEqualised.tar")

# 💡 절대 경로로 가중치 불러오기
checkpoint = torch.load(tar_path, map_location=device)

# 💡 발견한 보따리 이름('model_state_dict')으로 정확하게 가중치만 쏙 빼서 넣기
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
elif 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

# 이미 학습된 모델이므로 학습률(Learning Rate)은 매우 작게 설정! (미세 조정)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss() # 픽셀 차이를 계산하는 손실 함수

# 3. 학습 루프
def train_model():
    # TODO: 폴더 경로 지정 필요
    dataset = BoneSuppressionDataset512("원본_폴더_경로", "정답_폴더_경로")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model.train()
    epochs = 10 # 파인튜닝이므로 10~20번만 돌아도 충분합니다.
    
    print("🚀 512x512 고화질 파인튜닝 시작...")
    for epoch in range(epochs):
        epoch_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss/len(dataloader):.4f}")

    # 학습된 512 전용 가중치 저장!
    save_path = os.path.join(BASE_DIR, "lumora_bone_model_512_finetuned.pth")
    torch.save(model.state_dict(), save_path)
    print("✅ 파인튜닝 완료 및 저장 성공!")

if __name__ == "__main__":
    train_model()