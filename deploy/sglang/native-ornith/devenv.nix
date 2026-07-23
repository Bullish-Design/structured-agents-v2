{ pkgs, ... }:

let
  cudaToolkit = pkgs.symlinkJoin {
    name = "structured-agents-sglang-ornith-cuda-13.0";
    paths = [
      pkgs.cudaPackages_13_0.cuda_cudart
      pkgs.cudaPackages_13_0.cuda_nvcc
    ];
  };
in
{
  packages = [ pkgs.curl cudaToolkit pkgs.git pkgs.jq pkgs.python312 pkgs.stdenv.cc pkgs.uv pkgs.zlib ];

  # Match the NixOS service's CUDA/runtime linker environment. This devenv is
  # standalone: its uv project and .venv live beside this file, never under
  # deploy/vllm/native.
  enterShell = ''
    export CUDA_HOME=${cudaToolkit}
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc pkgs.zlib ]}:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
    # Load the opt-in GGUF config-resolution adapter from this isolated project.
    export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"
  '';
}
