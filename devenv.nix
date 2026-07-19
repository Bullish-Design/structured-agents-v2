{ pkgs, ... }:

{
  packages = [
    pkgs.git
    pkgs.uv
  ];

  languages.python = {
    enable = true;
    version = "3.13";
    venv.enable = true;
    uv.enable = true;
  };

  enterShell = ''
    git --version
  '';
}
