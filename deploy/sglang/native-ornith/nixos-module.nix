{ config, lib, pkgs, ... }:

let
  cfg = config.services.structuredAgentsSglangOrnith;
  cudaToolkit = pkgs.symlinkJoin {
    name = "structured-agents-sglang-ornith-cuda-13.0";
    paths = [
      pkgs.cudaPackages_13_0.cuda_cudart
      pkgs.cudaPackages_13_0.cuda_nvcc
    ];
  };
in
{
  options.services.structuredAgentsSglangOrnith = {
    enable = lib.mkEnableOption "the isolated SGLang Ornith-1.0-9B GGUF compatibility spike";
    repositoryPath = lib.mkOption { type = lib.types.str; default = "/home/andrew/Documents/Projects/structured-agents-v2"; };
    user = lib.mkOption { type = lib.types.str; default = "andrew"; };
    group = lib.mkOption { type = lib.types.str; default = "users"; };
    sglangVersion = lib.mkOption { type = lib.types.str; default = "0.5.14"; description = "SGLang package/version label, pinned in deploy/sglang/native-ornith/pyproject.toml."; };
    modelPath = lib.mkOption { type = lib.types.str; default = "/home/andrew/.cache/structured-agents/models/Ornith-1.0-9B-UD-Q4_K_XL.gguf"; };
    tokenizerPath = lib.mkOption { type = lib.types.str; default = "/home/andrew/.cache/structured-agents/sglang-ornith-tokenizer"; };
    servedModelName = lib.mkOption { type = lib.types.str; default = "base"; };
    gpu = lib.mkOption { type = lib.types.str; default = "1"; };
    port = lib.mkOption { type = lib.types.port; default = 8003; };
    contextLength = lib.mkOption { type = lib.types.int; default = 16384; };
    memFractionStatic = lib.mkOption { type = lib.types.str; default = "0.80"; };
    apiKey = lib.mkOption { type = lib.types.str; default = ""; };
    cachePath = lib.mkOption { type = lib.types.str; default = "/var/lib/structured-agents-sglang-ornith/cache"; description = "Dedicated SGLang/Hugging Face cache; never vLLM's or the gemma4 spike's cache."; };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.structured-agents-sglang-ornith = {
      description = "Structured Agents isolated SGLang Ornith-1.0-9B GGUF spike";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      path = [ pkgs.bash pkgs.coreutils pkgs.devenv pkgs.gnugrep pkgs.python312 pkgs.stdenv.cc pkgs.uv cudaToolkit pkgs.zlib ];
      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        WorkingDirectory = "${cfg.repositoryPath}/deploy/sglang/native-ornith";
        Environment = [
          "MODEL_PATH=${cfg.modelPath}"
          "TOKENIZER_PATH=${cfg.tokenizerPath}"
          "SGLANG_VERSION=${cfg.sglangVersion}"
          "SERVED_MODEL_NAME=${cfg.servedModelName}"
          "CUDA_VISIBLE_DEVICES=${cfg.gpu}"
          "PORT=${toString cfg.port}"
          "CONTEXT_LENGTH=${toString cfg.contextLength}"
          "MEM_FRACTION_STATIC=${cfg.memFractionStatic}"
          "MAX_RUNNING_REQUESTS=1"
          "CPU_OFFLOAD_GB=0"
          "SGLANG_CACHE_DIR=${cfg.cachePath}"
          "API_KEY=${cfg.apiKey}"
          "CUDA_HOME=${cudaToolkit}"
          "LD_LIBRARY_PATH=${lib.makeLibraryPath [ pkgs.stdenv.cc.cc pkgs.zlib ]}:/run/opengl-driver/lib"
          "TRITON_LIBCUDA_PATH=/run/opengl-driver/lib"
          "NIXPKGS_ALLOW_UNFREE=1"
        ];
        ExecStart = "${pkgs.bash}/bin/bash ${cfg.repositoryPath}/deploy/sglang/native-ornith/run.sh";
        Restart = "no";
        TimeoutStartSec = "20min";
        TimeoutStopSec = "2min";
        StateDirectory = "structured-agents-sglang-ornith";
        UMask = "0077";
      };
    };
  };
}
