import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm

# Mixed Precision 수정
from torch.amp import autocast, GradScaler   # ← 여기 수정!

# ==========================================
# 1. 환경 설정
# ==========================================
DATA_DIR = r"C:\Users\whdeh\Documents\대학교 3학년\융합설계 및 프로젝트기본\PadChest_GR_progression_prior_studies"
CACHE_DIR = "processed_cache"
BATCH_SIZE = 10
EPOCHS = 30
LEARNING_RATE = 0.001
BASE_CH = 24

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[Lung-Ai Training] 🔥 Device: {str(DEVICE).upper()}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ==========================================
# 2. Dataset (동일)
# ==========================================
class PadChestBoneDataset(Dataset):
    def __init__(self, folder_path, cache_dir=CACHE_DIR):
        self.file_list = glob.glob(os.path.join(folder_path, "*.png"))
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        if len(self.file_list) == 0:
            raise FileNotFoundError(f"❌ '{folder_path}'에 PNG 파일이 없습니다!")

        print(f"[Dataset] 총 {len(self.file_list)}개 이미지 Preprocessing 시작...")
        for idx, file_path in enumerate(self.file_list):
            cache_path = os.path.join(cache_dir, f"{idx:06d}.npz")
            if not os.path.exists(cache_path):
                self._preprocess_and_save(file_path, cache_path)
        print(f"✅ Preprocessing 완료!")

    def _preprocess_and_save(self, file_path, cache_path):
        img_array = np.fromfile(file_path, np.uint8)
        raw_img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        raw_img = cv2.resize(raw_img, (512, 512))

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(raw_img)

        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_tophat)

        _, bone_mask = cv2.threshold(tophat, 20, 255, cv2.THRESH_BINARY)
        bone_mask_dilated = cv2.dilate(bone_mask, np.ones((5, 5), np.uint8), iterations=2)

        pseudo_target = cv2.inpaint(enhanced, bone_mask_dilated, 5, cv2.INPAINT_TELEA)
        final_target = cv2.GaussianBlur(pseudo_target, (3, 3), 0)

        np.savez_compressed(cache_path, input=raw_img.astype(np.uint8), target=final_target.astype(np.uint8))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        cache_path = os.path.join(self.cache_dir, f"{idx:06d}.npz")
        data = np.load(cache_path)
        x = torch.from_numpy(data['input']).float().unsqueeze(0) / 255.0
        y = torch.from_numpy(data['target']).float().unsqueeze(0) / 255.0
        return x, y


# U-Net, Loss 등은 동일 (생략 없이 전체 넣으려면 이전 코드 복사)
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, base_ch=BASE_CH):
        super().__init__()
        self.enc1 = ConvBlock(1, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = ConvBlock(base_ch*4, base_ch*8)

        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch*8, base_ch*4)

        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch*4, base_ch*2)

        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch*2, base_ch)

        self.final = nn.Conv2d(base_ch, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.sigmoid(self.final(d1))

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.75):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def ssim_loss(self, pred, target):
        mu_p = pred.mean()
        mu_t = target.mean()
        sig_p = pred.var()
        sig_t = target.var()
        sig_pt = ((pred - mu_p) * (target - mu_t)).mean()
        C1, C2 = 0.01**2, 0.03**2
        ssim = (2*mu_p*mu_t + C1) * (2*sig_pt + C2) / \
               ((mu_p**2 + mu_t**2 + C1) * (sig_p + sig_t + C2))
        return 1 - ssim

    def forward(self, pred, target):
        return self.alpha * self.mse(pred, target) + (1 - self.alpha) * self.ssim_loss(pred, target)


# ==========================================
# 학습 준비
# ==========================================
dataset = PadChestBoneDataset(DATA_DIR)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True, persistent_workers=True, prefetch_factor=2)

model = UNet().to(DEVICE)

if hasattr(torch, 'compile'):
    try:
        model = torch.compile(model)
        print("✅ torch.compile 적용됨")
    except Exception as e:
        print("torch.compile 실패:", e)

criterion = CombinedLoss(alpha=0.75)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# GradScaler 수정
scaler = GradScaler()   # torch.amp에서 불러왔으므로 경고 사라짐

print("\n🚀 학습 시작! (진행바 + 경고 제거 버전)")

# ==========================================
# 학습 루프
# ==========================================
loss_history = []
best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=True)
    
    for inputs, targets in progress_bar:
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type=DEVICE.type):   # ← device_type 추가
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        progress_bar.set_postfix(loss=f"{loss.item():.5f}")

    epoch_loss = running_loss / len(dataset)
    loss_history.append(epoch_loss)
    scheduler.step(epoch_loss)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1:>3}/{EPOCHS}] Average Loss: {epoch_loss:.6f}  (lr={current_lr:.6f})")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "lumora_bone_model_best.pth")

    torch.cuda.empty_cache()
    gc.collect()

# 저장 및 그래프
torch.save(model.state_dict(), "lumora_bone_model_final.pth")
print(f"\n🎉 학습 완료! Best Loss: {best_loss:.6f}")

plt.figure(figsize=(8, 5))
plt.plot(loss_history, label='Train Loss', color='royalblue')
plt.title('Lung-Ai Bone Suppression Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.savefig("loss_curve.png", dpi=200)
plt.show()