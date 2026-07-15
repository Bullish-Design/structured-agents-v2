{ config, lib, pkgs, ... }:

let
  cfg = config.services.structuredAgentsSglang;
  cudaToolkit = pkgs.symlinkJoin {
    name = "structured-agents-sglang-cuda-13.0";
    paths = [
      pkgs.cudaPackages_13_0.cuda_cudart
      pkgs.cudaPackages_13_0.cuda_nvcc
    ];
  };
in
{
  options.services.structuredAgentsSglang = {
    enable = lib.mkEnableOption "the isolated SGLang Gemma 4 GGUF compatibility spike";
    repositoryPath = lib.mkOption { type = lib.types.str; default = "/home/andrew/Documents/Projects/structured-agents-v2"; };
    user = lib.mkOption { type = lib.types.str; default = "andrew"; };
    group = lib.mkOption { type = lib.types.str; default = "users"; };
    sglangVersion = lib.mkOption { type = lib.types.str; default = "0.5.14"; description = "SGLang package/version label, pinned in deploy/sglang/native/pyproject.toml."; };
    modelPath = lib.mkOption { type = lib.types.str; default = "/var/lib/structured-agents-vllm/hf/hub/models--unsloth--gemma-4-12B-it-qat-GGUF/blobs/cc9ff072e0a8203429ed854e6662c17a6c2bc1e5dca5b475dd4736caaacbc165"; };
    tokenizerPath = lib.mkOption { type = lib.types.str; default = "/var/lib/structured-agents-vllm/hf/gemma4-config-0e2b1058541244490925fbacf8972041435691ac"; };
    draftModelPath = lib.mkOption { type = lib.types.str; default = "/home/andrew/.cache/structured-agents/models/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf"; description = "Recorded only; serve.sh refuses MTP until runtime-proven."; };
    servedModelName = lib.mkOption { type = lib.types.str; default = "base"; };
    gpu = lib.mkOption { type = lib.types.str; default = "0"; };
    port = lib.mkOption { type = lib.types.port; default = 8002; };
    contextLength = lib.mkOption { type = lib.types.int; default = 16384; };
    memFractionStatic = lib.mkOption { type = lib.types.str; default = "0.80"; };
    apiKey = lib.mkOption { type = lib.types.str; default = ""; };
    cachePath = lib.mkOption { type = lib.types.str; default = "/var/lib/structured-agents-sglang/cache"; description = "Dedicated SGLang/Hugging Face cache; never vLLM's cache."; };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.structured-agents-sglang = {
      description = "Structured Agents isolated SGLang Gemma 4 GGUF spike";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      path = [ pkgs.bash pkgs.coreutils pkgs.devenv pkgs.gnugrep pkgs.python312 pkgs.stdenv.cc pkgs.uv cudaToolkit pkgs.zlib ];
      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        WorkingDirectory = "${cfg.repositoryPath}/deploy/sglang/native";
        Environment = [
          "MODEL_PATH=${cfg.modelPath}"
          "TOKENIZER_PATH=${cfg.tokenizerPath}"
          "DRAFT_MODEL_PATH=${cfg.draftModelPath}"
          "SGLANG_VERSION=${cfg.sglangVersion}"
          "SERVED_MODEL_NAME=${cfg.servedModelName}"
          "CUDA_VISIBLE_DEVICES=${cfg.gpu}"
          "PORT=${toString cfg.port}"
          "CONTEXT_LENGTH=${toString cfg.contextLength}"
          "MEM_FRACTION_STATIC=${cfg.memFractionStatic}"
          "MAX_RUNNING_REQUESTS=1"
          "CPU_OFFLOAD_GB=0"
          "ENABLE_MTP=0"
          "SGLANG_CACHE_DIR=${cfg.cachePath}"
          "API_KEY=${cfg.apiKey}"
          "CUDA_HOME=${cudaToolkit}"
          "LD_LIBRARY_PATH=${lib.makeLibraryPath [ pkgs.stdenv.cc.cc pkgs.zlib ]}:/run/opengl-driver/lib"
          "TRITON_LIBCUDA_PATH=/run/opengl-driver/lib"
          "NIXPKGS_ALLOW_UNFREE=1"
        ];
        ExecStart = "${pkgs.bash}/bin/bash ${cfg.repositoryPath}/deploy/sglang/native/run.sh";
        Restart = "no";
        TimeoutStartSec = "20min";
        TimeoutStopSec = "2min";
        StateDirectory = "structured-agents-sglang";
        UMask = "0077";
      };
    };
  };
}
