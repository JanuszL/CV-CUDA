# CV-CUDA

## Description

CV-CUDA is a library of CUDA-accelerated computer vision and image processing algorithms.

## Badges
TBD

## Pre-requisites

- Linux box, Ubuntu >= 20.04 recommended.
- docker - https://www.docker.com/
- nvidia-docker2 - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- CUDA driver

## Build instructions

Docker containers are used to have a controlled environment with everything that is needed
to successfully build CV-CUDA.

1. Clone the repository

   `git clone ssh://git@gitlab-master.nvidia.com:12051/cv/cvcuda.git`

   `cd cvcuda`

2. Initialize the cloned repository. It installs git pre-commit hooks.

   `./init_repo.sh`

2. Start the docker environment for development

   `docker/env_devel_linux.sh`

   From now on you're inside docker. The local cloned repository is mapped to `/cvcuda` inside the
   container. The container starts in this directory.

3. Build CV-CUDA

   `ci/build.sh`

   This will compile a x86 release build of CV-CUDA inside `build-rel` directory.
   The library is in build-rel/lib and executables (tests, etc...) in build-rel/bin.

   The script accepts some parameters to control the creation of the build tree:

   `ci/build.sh [release|debug] [output build tree path]`

   By default it builds for release.

   If output build tree path isn't specified, it'll be `build-rel` for release builds, and build-deb for debug.

4. Run tests

   The tests are in `<buildtree>/bin`. They can be executed from within the docker container.

5. Package installers

   From a succesfully built project, installers can be generated using cpack:

   `cd build-rel`

   `cpack .`

   This will generate in the build directory both Debian installers and tarballs (\*.tar.xz), needed for integration in other distros.

   For a fine-grained choice of what installers to generated, the full syntax is:

   `cmake . -G [DEB|TXZ]`

   - DEB for Debian packages
   - TXZ for \*.tar.xz tarballs.

## Notes
- To do a local lint check in all files, run:

  `pre-commit run -a`

## License

Currently CV-CUDA is for internal use only, within NVIDIA.
