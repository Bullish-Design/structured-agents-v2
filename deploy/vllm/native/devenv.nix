{ pkgs, ... }:

let
  cudaToolkit = pkgs.symlinkJoin {
    name = "structured-agents-cuda-13.0";
    paths = [
      pkgs.cudaPackages_13_0.cuda_cudart
      pkgs.cudaPackages_13_0.cuda_nvcc
    ];
  };
in
{
  packages = [ pkgs.curl cudaToolkit pkgs.ffmpeg pkgs.git pkgs.jq pkgs.python312 pkgs.stdenv.cc pkgs.uv pkgs.zlib ];

  # Match the systemd unit's runtime linker environment. PyTorch needs
  # libstdc++, while CUDA's host driver library is exposed by NixOS here.
  enterShell = ''
    export CUDA_HOME=${cudaToolkit}
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc pkgs.ffmpeg pkgs.zlib ]}:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
    export VLLM_USE_FLASHINFER_SAMPLER=0
  '';
}
