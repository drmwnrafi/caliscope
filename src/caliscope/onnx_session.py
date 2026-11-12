"""Shared ONNX Runtime session factory with automatic GPU provider selection.

Supports NVIDIA, AMD, Intel, Apple, and other accelerator backends
when their ONNX Runtime execution providers are installed.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

GPU_PROVIDER_PRIORITY = [
    "TensorRTExecutionProvider",  # NVIDIA TensorRT
    "CUDAExecutionProvider",      # NVIDIA CUDA
    "ROCMExecutionProvider",      # AMD ROCm
    "MIGraphXExecutionProvider",  # AMD MIGraphX
    "DmlExecutionProvider",       # Intel / AMD / NVIDIA DirectML
    "OpenVINOExecutionProvider",  # Intel CPU / GPU / NPU
    "CoreMLExecutionProvider",    # Apple hardware
    "CANNExecutionProvider",      # Huawei Ascend
    "QNNExecutionProvider",       # Qualcomm
    "WebGPUExecutionProvider",    # WebGPU
]


def create_inference_session(model_path: Path):
    """Create an ONNX Runtime session using the best available provider."""

    import onnxruntime as ort

    available = ort.get_available_providers()

    logger.info("Available ONNX Runtime providers: %s", available)

    if (
        "CUDAExecutionProvider" in available
        or "TensorRTExecutionProvider" in available
    ):
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception as exc:
                logger.warning("Could not preload CUDA DLLs: %s", exc)

    selected = [
        provider
        for provider in GPU_PROVIDER_PRIORITY
        if provider in available
    ]

    if not selected:
        logger.warning("No GPU execution provider available. Using CPU.")

    provider_options = {
        "TensorRTExecutionProvider": {"device_id": 0},
        "CUDAExecutionProvider": {"device_id": 0},
        "ROCMExecutionProvider": {"device_id": 0},
        "MIGraphXExecutionProvider": {"device_id": 0},
        "DmlExecutionProvider": {"device_id": 0},
        "OpenVINOExecutionProvider": {"device_type": "GPU_FP32"},
        "CANNExecutionProvider": {"device_id": 0},
    }

    session_options = ort.SessionOptions()

    if "DmlExecutionProvider" in selected:
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.enable_mem_pattern = False

    for index in range(len(selected) + 1):
        candidates = selected[index:]

        providers = [
            (
                provider,
                provider_options[provider],
            )
            if provider in provider_options
            else provider
            for provider in candidates
        ]

        providers.append("CPUExecutionProvider")

        try:
            session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers,
            )

            active = session.get_providers()

            logger.info("ONNX model: %s", model_path.name)
            logger.info("Requested providers: %s", providers)
            logger.info("Session providers: %s", active)

            if active[0] == "CPUExecutionProvider":
                logger.warning(
                    "ONNX Runtime is using CPU. "
                    "No GPU provider successfully initialized."
                )
            else:
                logger.info(
                    "ONNX Runtime selected: %s",
                    active[0],
                )

            return session

        except Exception:
            if index < len(selected):
                logger.exception(
                    "Failed to initialize %s. Trying the next provider.",
                    selected[index],
                )
            else:
                logger.exception("All GPU providers failed.")

    raise RuntimeError(
        f"Could not create ONNX Runtime session for {model_path}"
    )
