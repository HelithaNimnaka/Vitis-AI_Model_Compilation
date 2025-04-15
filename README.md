We have created a [PTQ and compilation script](PTQ_Quantization_Script_Superpoint.py) for the pre-trained SuperPoint model available [here](https://github.com/eric-yyjau/pytorch-superpoint/tree/master) for the AMD Xilinx Kria KR260 Robotics Starter Kit, and the target DPU used is DPUCZDX8G.

---

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
8. [Additional Resources](#additional-resources)

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
1. **Docker Pull Command**:  
   Pull the official Vitis AI Docker image. For more details, visit the [Docker Hub page](https://hub.docker.com/r/xilinx/vitis-ai).
   ```bash
   docker pull xilinx/vitis-ai
   ```
   We recommend using a GPU-enabled Docker. Installation instructions are provided [xilinx.github.io/Vitis-AI](https://xilinx.github.io/Vitis-AI/3.0/html/docs/install/install.html).
   
2. **Run the Vitis AI Docker Image**:
   Start the Vitis AI Docker container:
   ```bash
   docker run -it --rm -v "$(pwd)":/workspace -w /workspace --gpus all xilinx/vitis-ai-pytorch-gpu:3.5.0.001-1eed93cde bash
   ```

3. **Clone the Vitis AI Repository**:
   Clone the official repository to access the model zoo and related tools:
   ```bash
   git clone https://github.com/Xilinx/Vitis-AI
   ```
   For a specified version:
   ```bash
   git clone --branch v3.0 --single-branch https://github.com/Xilinx/Vitis-AI.git
   ```
   

5. **Navigate to the Model Zoo**:
   ```bash
   cd Vitis-AI/model_zoo
   ```

---

## Model Download and Extraction

1. **Run the Downloader Script**:
   Use `downloader.py` to list and download available models:
   ```bash
   python3 downloader.py
   ```

2. **List All Models**:
   Input `all` to display all available models:
   ```text
   input: all
   ```

3. **Choose a Model**:
   Input the corresponding number from the list. For example:
   ```text
   input num: 48
   ```
4. **Choose Model Type**:
   After selecting the model, choose the model type by inputting the corresponding number. For example:
   ```text
   0:all
   1:GPU
   2:vek280
   ```

   This downloads the selected model as a `.zip` file (e.g., `tf_superpoint_3.5.zip`).

5. **Extract the Model**:
   Unzip the downloaded file:
   ```bash
   unzip tf_superpoint_3.5.zip
   ```

   Locate the quantized evaluation model:
   ```text
   tf_superpoint_3.5/quantized/quantize_eval_model.pb
   ```

---

## Model Compilation

1. **Locate `arch.json`**:
   Architecture files required for compilation are preloaded in the Docker container. Navigate to the directory `/opt/vitis_ai/compiler/arch`:

   ```bash
   cd /opt/vitis_ai/compiler/arch
   ```

   List the available architecture directories:

   ```bash
   ls
   ```

   Navigate to the directory for your target platform. For example, for the KV260 platform:

   ```bash
   cd DPUCZDX8G/KV260
   ```

   List the contents to locate the `arch.json` file:

   ```bash
   ls
   ```

   Example path:
   ```text
   /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
   ```

2. **Compile the Model**:
   Use the `vai_c_tensorflow` command to compile the quantized model. Replace placeholders with actual paths:
   ```bash
   vai_c_tensorflow -f /PATH/TO/quantize_eval_model.pb    -a /PATH/TO/arch.json    -o /OUTPUTPATH    -n netname
   ```

   Example for `tf_superpoint_3.5` on KV260:
   ```bash
   vai_c_tensorflow -f Vitis-AI/model_zoo/tf_superpoint_3.5/quantized/quantize_eval_model.pb    -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json    -o Vitis-AI/model_zoo/tf_superpoint_3.5/Output    -n superpoint_net
   ```

3. **Verify Compilation**:
   Check the output directory for the compiled `.xmodel` file:
   ```bash
   ls Vitis-AI/model_zoo/tf_superpoint_3.5/Output
   ```

---

## Finding the `arch.json` File

Architecture files are preloaded in the Docker container. Use the following commands to locate them:

1. Navigate to the architecture directory:
   ```bash
   cd /opt/vitis_ai/compiler/arch
   ```

2. List available DPUs:
   ```bash
   ls
   ```

3. For KV260:
   ```bash
   cd DPUCZDX8G/KV260
   ls
   ```

   The `arch.json` file for KV260 will be present in this directory.

---

## Example Workflow

1. **Download and Extract Model**:
   ```bash
   cd Vitis-AI/model_zoo
   python3 downloader.py
   unzip tf_superpoint_3.5.zip
   ```

2. **Locate `arch.json`**:
   ```bash
   cd /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260
   ```

3. **Compile the Model**:
   ```bash
   vai_c_tensorflow -f Vitis-AI/model_zoo/tf_superpoint_3.5/quantized/quantize_eval_model.pb    -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json    -o Vitis-AI/model_zoo/tf_superpoint_3.5/Output    -n superpoint_net
   ```

---

## Notes

1. **Supported Frameworks**:
   - `tf`: TensorFlow 1.x
   - `tf2`: TensorFlow 2.x
   - `cf`: Caffe
   - `dk`: Darknet
   - `pt`: PyTorch
   - `all`: Lists all available models.

2. Ensure the `.xmodel` file is present in the output directory after compilation.

3. Use the correct `arch.json` file based on your target platform:
   - **KV260**: `/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json`
   - **ZCU102**: `/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json`
   - **ZCU104**: `/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json`

---

## Additional Resources

- [Vitis AI Documentation](https://github.com/Xilinx/Vitis-AI)
- [Vitis AI Model Zoo](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo)
- [Xilinx KV260 Platform](https://www.xilinx.com/products/som/kria/kv260-vision-starter-kit.html)
