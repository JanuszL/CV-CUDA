# CV-CUDA

## Description

CV-CUDA is a library of CUDA-accelerated computer vision and image processing algorithms.
CV-CUDA originated as a collaborative effort between NVIDIA and Bytedance.

## Badges
TBD

## Pre-requisites

- Recommended linux distros:
    - Ubuntu >= 20.04 (tested with 20.04 and 22.04)
    - WSL2 with Ubuntu >= 20.04 (tested with 20.04)
- docker, tested with v20.10.12 - https://www.docker.com/
- nvidia-docker2 - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- CUDA driver >= 11.7

## Build instructions

Docker containers are used to have a controlled environment with everything that is needed
to successfully build CV-CUDA.

1. Clone the repository

   ```
   git clone ssh://git@gitlab-master.nvidia.com:12051/cv/cvcuda.git
   cd cvcuda
   ```

2. Initialize the cloned repository. It installs git pre-commit hooks.
   In case of errors, follow the instructions shown to install some dependent packages.
   If it hangs, try to run it when disconnected from the VPN.  It is only needed once.

   ```
   ./init_repo.sh
   ```

3. Login to docker on gitlab (needed only once).
   The current user name may be different than the docker username.
   The current user must be in the docker group to be able to run docker without sudo.
   If you don't have already one personal access token, please go to
   [CV-CUDA gitlab project page](https://gitlab-master.nvidia.com/cv/cvcuda) and consult
   [Personal Access Tokens](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html)
   for instructions on how to create them.

   ```
   docker login gitlab-master.nvidia.com:5005 -u <your_username> -p <your_personal_access_token>
   ```

4. Start the docker environment for development.

   ```
   docker/env_devel_linux.sh
   ```

   From now on you're inside docker. The local cloned repository is mapped to `$HOME/cvcuda` inside the
   container. The container starts in $HOME.

5. Build CV-CUDA

   ```
   cd ~/cvcuda
   ci/build.sh
   ```

   This will compile a x86 release build of CV-CUDA inside `build-rel` directory.
   The library is in build-rel/lib, docs in build-rel/docs and executables (tests, etc...) in build-rel/bin.

   The script accepts some parameters to control the creation of the build tree:

   ```
   ci/build.sh [release|debug] [output build tree path]
   ```

   By default it builds for release.

   If output build tree path isn't specified, it'll be `build-rel` for release builds, and build-deb for debug.

6. Build documentation

   `ci/build_docs.sh [build folder]

   Example:
   `ci/build_docs.sh build

7. Run tests

   The tests are in `<buildtree>/bin`. They can be executed from within the docker container. You can run the script
   below to run all tests at once. Here's an example when build tree is created in `build-rel`

   ```
   build-rel/bin/run_tests.sh
   ```

8. Package installers

   From a succesfully built project, installers can be generated using cpack:

   ```
   cd build-rel
   cpack .
   ```

   This will generate in the build directory both Debian installers and tarballs (\*.tar.xz), needed for integration in other distros.

   For a fine-grained choice of what installers to generated, the full syntax is:

   ```
   cmake . -G [DEB|TXZ]
   ```

   - DEB for Debian packages
   - TXZ for \*.tar.xz tarballs.

## Notes
- To do a local lint check in all files, run:

   ```
   pre-commit run -a
   ```

## License

Nvidia Software Evaluation License
