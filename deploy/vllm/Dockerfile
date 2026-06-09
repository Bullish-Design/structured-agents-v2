# Drop-in vLLM OpenAI-compatible server to replace the llama.cpp `remora-server`.
#
# The official image ships the vLLM OpenAI API server + CUDA runtime. We layer a
# thin entrypoint that builds `vllm serve` args from env vars so the container is
# configured the same way regardless of base model / LoRA set / vLLM version.
#
# Models (base weights + LoRA adapters) are bind-mounted at /models, never baked in.
ARG VLLM_TAG=latest
FROM vllm/vllm-openai:${VLLM_TAG}

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV MODELS_DIR=/models
EXPOSE 8000

# Override the base image's entrypoint; we call `vllm serve` ourselves.
ENTRYPOINT ["/entrypoint.sh"]
