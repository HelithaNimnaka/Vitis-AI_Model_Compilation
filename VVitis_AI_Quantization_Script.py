import argparse
import os
import torch
from pytorch_nndct.apis import torch_quantizer
from datasets.patches_dataset import PatchesDataset
from models.SuperPointNet_gauss2 import SuperPointNet_gauss2
import yaml
from export import export_descriptor

# Configuration Paths

MODEL_PATH = '/workspace/Deep_Learning_Helitha/Quantization/pytorch-superpoint/logs/superpoint_kitti_heat2_0/checkpoints/superPointNet_50000_checkpoint.pth.tar'
OUTPUT_DIR = "/workspace/Deep_Learning_Helitha/Quantization/pytorch-superpoint/quantized"
DATA_DIR = "/workspace/Deep_Learning_Helitha/"
CONFIG_FILE = "/workspace/Deep_Learning_Helitha/Quantization/pytorch-superpoint/configs/magicpoint_repeatability_heatmap.yaml"

# Hyperparameters
BATCH_SIZE = 20
CALIB_ITER = 29
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Model
print("Loading SuperPoint model...")
model = SuperPointNet_gauss2().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully.")

# Load Configuration
print(f"Loading configuration from {CONFIG_FILE}...")
with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

# Ensure 'path' key is in the dataset configuration
if 'path' not in config['data']:
    config['data']['path'] = DATA_DIR
if 'pretrained' not in config['model']:
    config['model']['pretrained'] = MODEL_PATH
print("Configuration loaded successfully.")

# Prepare Calibration Dataset
print("Preparing calibration dataset...")
calibration_dataset = PatchesDataset(**config['data'])
calibration_loader = torch.utils.data.DataLoader(
    calibration_dataset, batch_size=BATCH_SIZE, shuffle=False
)
print("Calibration dataset prepared.")

# Calibration Function
def calibrate(quant_model, data_loader, num_iter):
    print("Starting calibration...")
    quant_model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= num_iter:
                break
            inputs = data["image"].to(DEVICE)
            # Convert NHWC to NCHW
            inputs = inputs.permute(0, 3, 1, 2)
            # Convert RGB to grayscale
            if inputs.shape[1] == 3:
                inputs = torch.mean(inputs, dim=1, keepdim=True)
            quant_model(inputs)
    print("Calibration completed.")

# Quantize Model
print("Initializing quantization...")
input_tensor = torch.randn(1, 1, 480, 640).to(DEVICE)
quantizer = torch_quantizer(
    quant_mode="calib",
    module=model,
    input_args=input_tensor
)
quant_model = quantizer.quant_model
print("Quantization initialized.")

# Perform Calibration
calibrate(quant_model, calibration_loader, CALIB_ITER)

# Export Quantized Model for Testing
print("Exporting quantized model configuration...")
quantizer.export_quant_config()

# Prepare Test Dataset
print("Preparing test dataset...")
test_config = config['data']
test_dataset = PatchesDataset(**test_config)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False  # Ensure batch size = 1 for xmodel export
)
print("Test dataset prepared.")

# Switch to Testing Mode
print("Switching to test mode...")
quantizer = torch_quantizer(
    quant_mode="test",
    module=model,
    input_args=input_tensor
)
quant_model = quantizer.quant_model

# Run Forward Pass Before Export
print("Performing forward pass for quantized model...")
quant_model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        if i >= 1:  # Run only one batch for forward pass
            break
        inputs = data["image"].to(DEVICE)
        # Convert NHWC to NCHW
        inputs = inputs.permute(0, 3, 1, 2)
        # Convert RGB to grayscale
        if inputs.shape[1] == 3:
            inputs = torch.mean(inputs, dim=1, keepdim=True)
        quant_model(inputs)
print("Forward pass completed.")

# Export Descriptor for Testing
print("Running export_descriptor to generate keypoint predictions...")
args = argparse.Namespace(
    config=CONFIG_FILE,
    exper_name="export_test",
    correspondence=False,
    eval=True,
    debug=False,
    command="export_descriptor"  # Add the missing command attribute
)
export_descriptor(config, OUTPUT_DIR, args)

# Export .xmodel and Compile to Target
print(f"Exporting .xmodel to {OUTPUT_DIR}...")
quantizer.export_xmodel(deploy_check=False, output_dir=OUTPUT_DIR)
print(f"Quantized .xmodel exported to: {OUTPUT_DIR}")

# Save Quantized PyTorch Model
quantized_model_path = os.path.join(OUTPUT_DIR, "quantized_model.pth")
torch.save(quant_model.state_dict(), quantized_model_path)
print(f"Quantized PyTorch model saved at: {quantized_model_path}")

# Compile the exported .xmodel
import subprocess
compiled_output_dir = os.path.join(OUTPUT_DIR, "compiled")
os.makedirs(compiled_output_dir, exist_ok=True)
xmodel_path = os.path.join(OUTPUT_DIR, "SuperPointNet_gauss2_int.xmodel")
arch_json_path = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"
compile_command = [
    "vai_c_xir",
    "-x", xmodel_path,
    "-a", arch_json_path,
    "-o", compiled_output_dir,
    "-n", "compiled_by_H_"
]

try:
    print("Starting compilation...")
    subprocess.run(compile_command, check=True)
    print(f"Compilation completed. Compiled model saved to: {compiled_output_dir}")
except subprocess.CalledProcessError as e:
    print(f"Error during compilation: {e}")
