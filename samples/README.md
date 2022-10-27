# CV-CUDA Samples

## Description

Sample applications to show how use some of CV-CUDA's functionalities.

## Pre-requisites

- Recommended linux distros:
    - Ubuntu >= 20.04 (tested with 20.04 and 22.04)
    - WSL2 with Ubuntu >= 20.04 (tested with 20.04)
- CUDA driver >= 11.7
- TensorRT
- torch
- torchvision

## Setup to compile the sample from source.

1. Install CUDA and TensorRT Or use the TensorRT docker.

   ```
   docker run -it --gpus=all -v <local mount path>:<docker mount path> nvcr.io/nvidia/tensorrt:22.09-py3
   ```

2. Install the packages

   ```
   dpkg -i nvcv-lib-0.1.0-cuda11-x86_64-linux.deb
   dpkg -i nvcv-dev-0.1.0-cuda11-x86_64-linux.deb
   dpkg -i cvcuda-samples-0.1.0-cuda11-x86_64-linux.deb
   ```

3. Copy the samples folder to the target directory

   ```
   cp -rf /opt/nvidia/cvcuda0/samples ~/
   cd ~/samples
   ```

4. Install the dependencies

   ```
   chmod a+x scripts/*.sh
   chmod a+x scripts/*.py
   ./scripts/install_dependencies.sh
   ```

5. Build the sample

   ```
   ./scripts/build_samples.sh
   ```
6. Run the sample. This script serializes generates a onnx files from the pytorch resnet model and converts to TensorRT engine which is loaded by the sample application.

   Note: The first run serializes the model which would make the inference slow.
   The maximum image dimensions needs to be set in the Main.cpp file if testing with different images.

   ```
   ./scripts/run_samples.sh
   ```

## License

Nvidia Software Evaluation License

## Attributions

- Weimaraner.jpg image is obtained from [wikimedia](https://commons.wikimedia.org/wiki/File:Baegle_dwa.jpg) under Creative Commons Attribution-Share Alike 3.0 Unported license.
- tabby_tiger_cat.jpg is obtained from [maxpixel.net](https://www.maxpixel.net/Cute-Kitten-Cat-Tabby-Animals-Outdoors-Pets-1506960) under Creative Commons Zero - CC0 license).
