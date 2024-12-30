# Vitis AI Model Compilation and Deployment Guide

This repository provides detailed instructions for downloading, extracting, compiling, and deploying AI models using **Vitis AI** within a Docker environment.

---

## Table of Contents
1. [Overview](#overview)
2. [Environment Setup](#environment-setup)
3. [Model Download and Extraction](#model-download-and-extraction)
4. [Model Compilation](#model-compilation)
5. [Finding the `arch.json` File](#finding-the-archjson-file)
6. [Example Workflow](#example-workflow)
7. [Notes](#notes)

---

## Overview

**Vitis AI** is a development stack optimized for Xilinx hardware platforms. It enables developers to accelerate AI inference tasks efficiently using specialized Deep Learning Processing Units (DPUs). Supported platforms include:
- **KV260 Vision Starter Kit**
- **ZCU102 Evaluation Kit**
- **ZCU104 Evaluation Kit**

Key tools used:
- **Vitis AI Docker**: Provides a pre-configured environment.
- **`vai_c_tensorflow`**: Compiler for TensorFlow models.

---

## Environment Setup

1. **Run the Vitis AI Docker Image**:
   Start the Vitis AI Docker container:
   ```bash
   docker run -it xilinx/vitis-ai-pytorch-gpu
Clone the Vitis AI Repository: Clone the official repository to access the model zoo and related tools:

bash
Copy code
git clone https://github.com/Xilinx/Vitis-AI
Navigate to the Model Zoo:

bash
Copy code
cd Vitis-AI/model_zoo
Model Download and Extraction
Run the Downloader Script: Use downloader.py to list and download available models:

bash
Copy code
python3 downloader.py
List All Models: Input all to display all available models:

text
Copy code
input: all
Choose a Model: Input the corresponding number from the list. For example:

text
Copy code
input num: 1
This downloads the selected model as a .zip file (e.g., tf_superpoint_3.5.zip).

Extract the Model: Unzip the downloaded file:

bash
Copy code
unzip tf_superpoint_3.5.zip
Locate the quantized evaluation model:

text
Copy code
tf_superpoint_3.5/quantized/quantize_eval_model.pb
Model Compilation
Locate arch.json: Architecture files required for compilation are preloaded in the Docker container. Navigate to the directory /opt/vitis_ai/compiler/arch:

bash
Copy code
cd /opt/vitis_ai/compiler/arch
List the available architecture directories:

bash
Copy code
ls
Navigate to the directory for your target platform. For example, for the KV260 platform:

bash
Copy code
cd DPUCZDX8G/KV260
List the contents to locate the arch.json file:

bash
Copy code
ls
Example path:

text
Copy code
/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
Compile the Model: Use the vai_c_tensorflow command to compile the quantized model. Replace placeholders with actual paths:

bash
Copy code
vai_c_tensorflow -f /PATH/TO/quantize_eval_model.pb \
-a /PATH/TO/arch.json \
-o /OUTPUTPATH \
-n netname
Example for tf_superpoint_3.5 on KV260:

bash
Copy code
vai_c_tensorflow -f Vitis-AI/model_zoo/tf_superpoint_3.5/quantized/quantize_eval_model.pb \
-a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
-o Vitis-AI/model_zoo/tf_superpoint_3.5/Output \
-n superpoint_net
Verify Compilation: Check the output directory for the compiled .xmodel file:

bash
Copy code
ls Vitis-AI/model_zoo/tf_superpoint_3.5/Output
Finding the arch.json File
Architecture files are preloaded in the Docker container. Use the following commands to locate them:

Navigate to the architecture directory:

bash
Copy code
cd /opt/vitis_ai/compiler/arch
List available DPUs:

bash
Copy code
ls
For KV260:

bash
Copy code
cd DPUCZDX8G/KV260
ls
The arch.json file for KV260 will be present in this directory.

Example Workflow
Download and Extract Model:

bash
Copy code
cd Vitis-AI/model_zoo
python3 downloader.py
unzip tf_superpoint_3.5.zip
Locate arch.json:

bash
Copy code
cd /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260
Compile the Model:

bash
Copy code
vai_c_tensorflow -f Vitis-AI/model_zoo/tf_superpoint_3.5/quantized/quantize_eval_model.pb \
-a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
-o Vitis-AI/model_zoo/tf_superpoint_3.5/Output \
-n superpoint_net
Notes
Supported Frameworks:

tf: TensorFlow 1.x
tf2: TensorFlow 2.x
cf: Caffe
dk: Darknet
pt: PyTorch
all: Lists all available models.
Ensure the .xmodel file is present in the output directory after compilation.

Use the correct arch.json file based on your target platform:

KV260: /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
ZCU102: /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json
ZCU104: /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
Additional Resources
Vitis AI Documentation
Vitis AI Model Zoo
Xilinx KV260 Platform
python
Copy code

This version is a complete, standalone Markdown document with all necessary information consolidated.
