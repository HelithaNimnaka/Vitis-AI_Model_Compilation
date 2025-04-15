import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from pytorch_nndct.apis import torch_quantizer
from xfeat import XFeat
import subprocess

# === CONFIGURATION ===
HPATCHES_DIR = '/workspace/HPatches'
OUTPUT_DIR = '/workspace/output'
MODEL_PATH = '/workspace/xfeat.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1
CALIB_ITER = 30
INPUT_SHAPE = (1, 1, 480, 640)
ARCH_JSON_PATH = '/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json'  # <-- update if needed

# Ensure output dirs exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
compiled_output_dir = os.path.join(OUTPUT_DIR, "compiled")
os.makedirs(compiled_output_dir, exist_ok=True)

# === HPatches Dataset ===
class HPatchesDataset(Dataset):
    def __init__(self, root_dir, image_size=(480, 640), max_samples=100):
        self.samples = []
        self.image_size = image_size
        for root, _, files in os.walk(root_dir):
            for fname in sorted(files):
                if fname.endswith(('.ppm', '.png', '.jpg')):
                    self.samples.append(os.path.join(root, fname))
                    if len(self.samples) >= max_samples:
                        break
            if len(self.samples) >= max_samples:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = cv2.imread(self.samples[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.image_size)
        img = img.astype('float32') / 255.0
        return torch.tensor(img).unsqueeze(0)  # [1, H, W]

# === Load model ===
print("Loading XFeat model...")
model = XFeat(weights=MODEL_PATH).net.to(DEVICE).eval()

# === Prepare calibration data ===
print("Loading HPatches dataset...")
calib_dataset = HPatchesDataset(HPATCHES_DIR)
calib_loader = DataLoader(calib_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Calibration function ===
def calibrate(model, loader, num_iter):
    model.eval()
    with torch.no_grad():
        for i, img in enumerate(loader):
            if i >= num_iter:
                break
            model(img.to(DEVICE))
            print(f"[Calibration] Step {i+1}/{num_iter}", end='\r')
    print("\nCalibration done.")

# === Calibration Phase ===
input_tensor = torch.randn(INPUT_SHAPE).to(DEVICE)
quantizer = torch_quantizer("calib", model, input_args=input_tensor)
quant_model = quantizer.quant_model
calibrate(quant_model, calib_loader, CALIB_ITER)
quantizer.export_quant_config()

# === Test Mode (for xmodel export) ===
quantizer = torch_quantizer("test", model, input_args=input_tensor)
quant_model = quantizer.quant_model
quant_model(input_tensor)  # dummy forward

# === Export .xmodel and PyTorch checkpoint ===
quantizer.export_xmodel(output_dir=OUTPUT_DIR)
torch.save(quant_model.state_dict(), os.path.join(OUTPUT_DIR, "xfeat_quantized.pth"))
print(f"Exported xmodel and checkpoint to: {OUTPUT_DIR}")

# === Vitis AI Compilation ===
xmodel_path = os.path.join(OUTPUT_DIR, "XFeatModel_int.xmodel")  # adjust if named differently
compile_cmd = [
    "vai_c_xir",
    "-x", xmodel_path,
    "-a", ARCH_JSON_PATH,
    "-o", compiled_output_dir,
    "-n", "compiled_xfeat"
]

try:
    print("Starting Vitis AI compilation...")
    subprocess.run(compile_cmd, check=True)
    print(f"Compilation complete. Output saved to: {compiled_output_dir}")
except subprocess.CalledProcessError as e:
    print(f"Compilation failed: {e}")
