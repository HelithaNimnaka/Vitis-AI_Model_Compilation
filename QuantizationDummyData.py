import os
import torch
from pytorch_nndct.apis import torch_quantizer
from xfeat import XFeat
from torch.utils.data import DataLoader
import numpy as np

# Paths
MODEL_PATH = '/workspace/xfeat.pt'  # Already loaded inside XFeat
OUTPUT_DIR = '/workspace/output'

# Parameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1
CALIB_ITER = 20
INPUT_SHAPE = (1, 1, 480, 640)  # grayscale input

print("Loading XFeat model...")
model = XFeat(weights=MODEL_PATH).net.to(DEVICE).eval()  # Direct access to the core XFeatModel
print("Model loaded.")

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = np.random.rand(480, 640).astype(np.float32)  # Grayscale
        return torch.tensor(image).unsqueeze(0)  # Shape: [1, H, W]

print("Creating dummy dataset...")
calib_dataset = DummyDataset()
calib_loader = DataLoader(calib_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("Starting calibration...")

def calibrate(quant_model, loader, num_iter):
    print("Starting calibration...")
    quant_model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= num_iter:
                break
            input_tensor = data.to(DEVICE)  # shape: [B, 1, H, W]
            quant_model(input_tensor)
    print("Calibration done.")

print("Initializing quantization...")
input_tensor = torch.randn(INPUT_SHAPE).to(DEVICE)

quantizer = torch_quantizer(
    quant_mode="calib",
    module=model,
    input_args=input_tensor
)
quant_model = quantizer.quant_model
print("Quantization model created.")

calibrate(quant_model, calib_loader, CALIB_ITER)
print("Starting quantization...")

quantizer.export_quant_config()

print("Switching to test mode for export...")
quantizer = torch_quantizer(
    quant_mode="test",
    module=model,
    input_args=input_tensor
)
quant_model = quantizer.quant_model

# Run dummy inference to generate quantized ops
quant_model(input_tensor)

print("Model compiled successfully.")
print("Exporting quantized model...")
quantizer.export_xmodel(output_dir=OUTPUT_DIR)
print("Exported quantized model to:", OUTPUT_DIR)
print("Saving quantized model state dict...")
torch.save(quant_model.state_dict(), os.path.join(OUTPUT_DIR, "quantized_xfeat.pth"))


import subprocess
print("Compiling model...")
compiled_output_dir = os.path.join(OUTPUT_DIR, "compiled")
os.makedirs(compiled_output_dir, exist_ok=True)

xmodel_path = os.path.join(OUTPUT_DIR, "XFeatModel_int.xmodel")  # adjust if needed
arch_path = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"  # update as needed

compile_cmd = [
    "vai_c_xir",
    "-x", xmodel_path,
    "-a", arch_path,
    "-o", compiled_output_dir,
    "-n", "compiled_xfeat"
]

subprocess.run(compile_cmd, check=True)
print("Model compiled successfully.")