{
  description = "Structured Agents NixOS modules";

  outputs = { self }: {
    nixosModules.structuredAgentsVllm = import ./deploy/vllm/native/nixos-module.nix;
  };
}
