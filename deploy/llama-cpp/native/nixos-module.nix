{ config, lib, pkgs, ... }:

let
  cfg = config.services.structuredAgentsLlamaCpp;
  llamaCpp = pkgs.llama-cpp.override { cudaSupport = true; };
in
{
  options.services.structuredAgentsLlamaCpp = {
    enable = lib.mkEnableOption "the independent llama.cpp comparison server";

    repositoryPath = lib.mkOption {
      type = lib.types.str;
      example = "/home/andrew/Documents/Projects/structured-agents-v2";
      description = "Absolute checkout containing deploy/llama-cpp/native.";
    };
    user = lib.mkOption { type = lib.types.str; default = "andrew"; };
    group = lib.mkOption { type = lib.types.str; default = "users"; };

    # This is the immutable GGUF blob already downloaded by the vLLM profile.
    # The current Hugging Face Xet cache stores it by content hash rather than
    # exposing a snapshots/<revision>/<filename> path.
    modelPath = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/structured-agents-vllm/hf/hub/models--unsloth--gemma-4-12B-it-qat-GGUF/blobs/cc9ff072e0a8203429ed854e6662c17a6c2bc1e5dca5b475dd4736caaacbc165";
      description = "Absolute path to gemma-4-12B-it-qat-UD-Q4_K_XL.gguf.";
    };
    draftModelPath = lib.mkOption {
      type = lib.types.str;
      default = "/home/andrew/.cache/structured-agents/models/mtp-gemma-4-12B-it.gguf";
      description = "Absolute path to Unsloth's Gemma 4 MTP drafter GGUF (fallback profile).";
    };
    q8DraftModelPath = lib.mkOption {
      type = lib.types.str;
      default = "/home/andrew/.cache/structured-agents/models/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf";
      description = "Absolute path to Janvitos' Q8_0 Gemma 4 12B QAT MTP assistant GGUF.";
    };
    draftProfile = lib.mkOption {
      type = lib.types.enum [ "unsloth" "q8" ];
      default = "unsloth";
      description = "MTP drafter to serve; q8 is enabled only after a successful runtime benchmark.";
    };
    specDraftNMax = lib.mkOption {
      type = lib.types.ints.positive;
      default = 4;
      description = "Maximum tokens proposed per Gemma 4 MTP speculative-decoding step.";
    };
    servedModelName = lib.mkOption { type = lib.types.str; default = "base"; };
    port = lib.mkOption { type = lib.types.port; default = 8001; };
    gpu = lib.mkOption {
      type = lib.types.str;
      default = "0";
      description = "Dedicated NVIDIA GPU index; GPU 1 remains reserved for vLLM.";
    };
    contextSize = lib.mkOption { type = lib.types.int; default = 16384; };
    parallelSlots = lib.mkOption { type = lib.types.int; default = 1; };
    cpuThreads = lib.mkOption { type = lib.types.int; default = 8; };
    apiKey = lib.mkOption {
      type = lib.types.str;
      default = "";
      description = "Optional llama.cpp API key; leave empty to use tailnet ACLs only.";
    };
    publishViaTailscale = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Publish the loopback API through Tailscale Serve HTTPS on :8443.";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.structured-agents-llama-cpp = {
      description = "Structured Agents llama.cpp Gemma 4 comparison API";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      path = [ pkgs.bash pkgs.coreutils llamaCpp ];
      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        WorkingDirectory = "${cfg.repositoryPath}/deploy/llama-cpp/native";
        Environment = [
          "MODEL_PATH=${cfg.modelPath}"
          "DRAFT_MODEL_PATH=${if cfg.draftProfile == "q8" then cfg.q8DraftModelPath else cfg.draftModelPath}"
          "SPEC_DRAFT_N_MAX=${toString cfg.specDraftNMax}"
          "SERVED_MODEL_NAME=${cfg.servedModelName}"
          "PORT=${toString cfg.port}"
          "CUDA_VISIBLE_DEVICES=${cfg.gpu}"
          "CTX_SIZE=${toString cfg.contextSize}"
          "PARALLEL_SLOTS=${toString cfg.parallelSlots}"
          "CPU_THREADS=${toString cfg.cpuThreads}"
          "CPU_BATCH_THREADS=${toString cfg.cpuThreads}"
          "API_KEY=${cfg.apiKey}"
        ];
        ExecStart = "${pkgs.bash}/bin/bash ${cfg.repositoryPath}/deploy/llama-cpp/native/serve.sh";
        Restart = "on-failure";
        RestartSec = "10s";
        TimeoutStartSec = "10min";
        TimeoutStopSec = "2min";
        UMask = "0077";
      };
    };

    # Use :8443 so this never replaces vLLM's existing :443 Serve rule.
    systemd.services.structured-agents-llama-cpp-tailscale-serve = {
      description = "Publish Structured Agents llama.cpp through Tailscale Serve";
      wantedBy = lib.optionals cfg.publishViaTailscale [ "multi-user.target" ];
      requires = [ "structured-agents-llama-cpp.service" "tailscaled.service" ];
      after = [ "network-online.target" "tailscaled.service" "structured-agents-llama-cpp.service" ];
      path = [ pkgs.tailscale ];
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStart = "${pkgs.tailscale}/bin/tailscale serve --bg --https=8443 http://127.0.0.1:${toString cfg.port}";
        ExecStop = "${pkgs.tailscale}/bin/tailscale serve --https=8443 off";
      };
    };
  };
}
