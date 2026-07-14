{ config, lib, pkgs, ... }:

let
  cfg = config.services.structuredAgentsVllm;
  cudaToolkit = pkgs.symlinkJoin {
    name = "structured-agents-cuda-13.0";
    paths = [
      pkgs.cudaPackages_13_0.cuda_cudart
      pkgs.cudaPackages_13_0.cuda_nvcc
    ];
  };
in
{
  options.services.structuredAgentsVllm = {
    enable = lib.mkEnableOption "the native structured-agents vLLM server";

    repositoryPath = lib.mkOption {
      type = lib.types.str;
      example = "/home/andrew/Documents/Projects/structured-agents-v2";
      description = "Absolute checkout path; it must contain the locked native devenv environment.";
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "andrew";
      description = "Account that owns the checkout, devenv state, and GPU process.";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "users";
      description = "Primary group for the vLLM process.";
    };

    model = lib.mkOption {
      type = lib.types.str;
      default = "unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL";
      description = "Hugging Face GGUF model selector for the Gemma 4 serving profile.";
    };

    modelRevision = lib.mkOption {
      type = lib.types.str;
      default = "f18012b8f690e563b7f872cb764b4cb3de90b14a";
      description = "Immutable Hugging Face commit for the model weights.";
    };

    tokenizer = lib.mkOption {
      type = lib.types.str;
      default = "google/gemma-4-12B-it";
      description = "Hugging Face tokenizer repository used instead of GGUF metadata conversion.";
    };

    tokenizerRevision = lib.mkOption {
      type = lib.types.str;
      default = "0e2b1058541244490925fbacf8972041435691ac";
      description = "Immutable Hugging Face commit for the tokenizer repository.";
    };

    hfConfigPath = lib.mkOption {
      type = lib.types.str;
      default = "google/gemma-4-12B-it";
      description = "Hugging Face configuration repository used by the GGUF loader.";
    };

    hfConfigRevision = lib.mkOption {
      type = lib.types.str;
      default = "0e2b1058541244490925fbacf8972041435691ac";
      description = "Immutable Hugging Face commit for the configuration repository.";
    };

    servedModelName = lib.mkOption {
      type = lib.types.str;
      default = "base";
      description = "OpenAI-compatible model name exposed by vLLM.";
    };

    maxModelLen = lib.mkOption {
      type = lib.types.int;
      default = 16384;
      description = "CUDA-graph all-GPU context limit for the Gemma 4 profile.";
    };

    gpuMemoryUtilization = lib.mkOption {
      type = lib.types.str;
      default = "0.82";
      description = "Fraction of GPU memory vLLM may reserve.";
    };

    gpu = lib.mkOption {
      type = lib.types.str;
      default = "1";
      description = "Dedicated NVIDIA GPU index, exposed to vLLM through CUDA_VISIBLE_DEVICES.";
    };

    hfHome = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/structured-agents-vllm/hf";
      description = "Persistent Hugging Face cache directory.";
    };

    publishViaTailscale = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Publish the loopback-only API through Tailscale Serve HTTPS on port 443.";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.structured-agents-vllm = {
      description = "Structured Agents native vLLM API";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      path = [ pkgs.bash pkgs.coreutils cudaToolkit pkgs.devenv pkgs.stdenv.cc pkgs.uv ];

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        WorkingDirectory = "${cfg.repositoryPath}/deploy/vllm/native";
        Environment = [
          "MODEL=${cfg.model}"
          "MODEL_REVISION=${cfg.modelRevision}"
          "TOKENIZER=${cfg.tokenizer}"
          "TOKENIZER_REVISION=${cfg.tokenizerRevision}"
          "HF_CONFIG_PATH=${cfg.hfConfigPath}"
          "HF_CONFIG_REVISION=${cfg.hfConfigRevision}"
          "SERVED_MODEL_NAME=${cfg.servedModelName}"
          "MAX_MODEL_LEN=${toString cfg.maxModelLen}"
          "GPU_MEMORY_UTILIZATION=${cfg.gpuMemoryUtilization}"
          "CUDA_VISIBLE_DEVICES=${cfg.gpu}"
          "CUDA_HOME=${cudaToolkit}"
          "NIXPKGS_ALLOW_UNFREE=1"
          "HF_HOME=${cfg.hfHome}"
          # PyTorch's native extension is dynamically linked against libstdc++.
          # NixOS exposes the loaded NVIDIA driver's libcuda.so.1 through
          # /run/opengl-driver; a systemd service inherits neither path from an
          # interactive devenv shell.
          "LD_LIBRARY_PATH=${lib.makeLibraryPath [ pkgs.stdenv.cc.cc pkgs.ffmpeg pkgs.zlib ]}:/run/opengl-driver/lib"
          # Triton otherwise invokes the non-existent /sbin/ldconfig on NixOS
          # while locating libcuda during its first compilation.
          "TRITON_LIBCUDA_PATH=/run/opengl-driver/lib"
          # FlashInfer's optional sampler JIT combines its bundled CUDA headers
          # with the pip-provided CUDA 13 compiler, which is incompatible on
          # this NixOS profile. vLLM's native sampler remains fully supported.
          "VLLM_USE_FLASHINFER_SAMPLER=0"
        ];
        ExecStart = "${pkgs.bash}/bin/bash ${cfg.repositoryPath}/deploy/vllm/native/run.sh";
        Restart = "on-failure";
        RestartSec = "10s";
        TimeoutStartSec = "20min";
        TimeoutStopSec = "2min";
        StateDirectory = "structured-agents-vllm";
        UMask = "0077";
      };
    };
    # This changes only Serve's HTTPS :443 rule. Existing TCP Serve rules (such
    # as the zelligate ports) are left untouched.
    systemd.services.structured-agents-vllm-tailscale-serve = {
      description = "Publish Structured Agents vLLM through Tailscale Serve";
      wantedBy = lib.optionals cfg.publishViaTailscale [ "multi-user.target" ];
      requires = [ "structured-agents-vllm.service" "tailscaled.service" ];
      after = [ "network-online.target" "tailscaled.service" "structured-agents-vllm.service" ];
      wants = [ "network-online.target" ];
      path = [ pkgs.tailscale ];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStart = "${pkgs.tailscale}/bin/tailscale serve --bg --https=443 http://127.0.0.1:8000";
        ExecStop = "${pkgs.tailscale}/bin/tailscale serve --https=443 off";
      };
    };
  };
}
