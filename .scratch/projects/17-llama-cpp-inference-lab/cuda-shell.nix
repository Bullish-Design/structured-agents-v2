# Standalone CUDA build shell for the llama.cpp cuda-3060 build.
# NON-INVASIVE: does not touch repo-root devenv.nix. Enter with:
#   NIXPKGS_ALLOW_UNFREE=1 nix-shell --impure cuda-shell.nix
# Provides: gcc, cmake, ninja, and the CUDA toolkit (nvcc + cudart + libs)
# for an RTX 3060 (sm_86) build.
let
  nixpkgs = builtins.getFlake "nixpkgs";
  pkgs = import nixpkgs.outPath {
    system = "x86_64-linux";
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
  };
  cuda = pkgs.cudaPackages;
in
pkgs.mkShell {
  name = "llamacpp-cuda-3060";
  nativeBuildInputs = [
    pkgs.cmake
    pkgs.ninja
    pkgs.ccache
    pkgs.gcc
    cuda.cuda_nvcc
  ];
  buildInputs = [
    cuda.cuda_cudart
    cuda.cuda_cccl
    cuda.libcublas
    cuda.cuda_nvrtc
  ];
  shellHook = ''
    export CCACHE_DIR="''${CCACHE_DIR:-$HOME/.cache/llamacpp-ccache}"
    mkdir -p "$CCACHE_DIR"
    export CUDAToolkit_ROOT=${cuda.cuda_nvcc}
    echo "nvcc: $(command -v nvcc)"
    echo "cmake: $(command -v cmake)"
    echo "gcc: $(command -v gcc)"
  '';
}
