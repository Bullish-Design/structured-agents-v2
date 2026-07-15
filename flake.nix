{
  description = "Structured Agents NixOS modules";

  outputs = { self }: {
    nixosModules.structuredAgentsVllm = import ./deploy/vllm/native/nixos-module.nix;
    nixosModules.structuredAgentsLlamaCpp = import ./deploy/llama-cpp/native/nixos-module.nix;
    nixosModules.structuredAgentsSglang = import ./deploy/sglang/native/nixos-module.nix;
  };
}
