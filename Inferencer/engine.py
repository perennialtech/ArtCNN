# SPDX-FileCopyrightText: 2025 João Chrisóstomo
# SPDX-FileCopyrightText: 2026 perennialtech contributors
# SPDX-License-Identifier: MIT

import os

import onnxruntime as ort
import numpy as np


def default_providers():
    if "DmlExecutionProvider" in ort.get_available_providers():
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class OpenVINOModel:
    def __init__(self, model, threads):
        try:
            import openvino as ov
        except ImportError as error:
            raise RuntimeError("The OpenVINO CPU backend requires the openvino package.") from error
        config = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"}
        if threads is not None:
            config["INFERENCE_NUM_THREADS"] = threads
        compiled = ov.Core().compile_model(str(model), "CPU", config)
        self.input = compiled.input()
        self.output = compiled.output()
        self.request = compiled.create_infer_request()

    def run(self, tensor):
        return self.request.infer({0: tensor})[self.output]


class Engine:
    def __init__(self, model, providers=None, *, cpu_threads=None, cpu_backend=None, cpu_tile=None):
        if cpu_threads is not None and cpu_threads < 1:
            raise ValueError("CPU thread count must be positive.")
        self.model = model
        self.tile_plan = None
        if cpu_tile is not None:
            if cpu_tile < 1 or cpu_backend not in (None, "onnxruntime"):
                raise ValueError("CPU tiling requires a positive tile edge and the ONNX Runtime CPU backend.")
            from tiling import TilePlan, model_footprint

            field = model_footprint(model)
        if cpu_backend == "openvino":
            self.session = OpenVINOModel(model, cpu_threads)
            port = self.session.input
            self.input_channels = port.partial_shape[1].get_length() if port.partial_shape[1].is_static else None
            self.input_dtype = port.element_type.to_dtype()
            self.run_tensor = self.session.run
        elif cpu_backend in (None, "onnxruntime"):
            if cpu_backend == "onnxruntime" or cpu_threads is not None or cpu_tile is not None:
                providers = ["CPUExecutionProvider"]
            options = ort.SessionOptions()
            if cpu_tile is not None:
                options.intra_op_num_threads = 1
            elif cpu_threads is not None:
                options.intra_op_num_threads = cpu_threads
            self.providers = providers or default_providers()
            if tuple(self.providers) == ("CPUExecutionProvider",):
                # Repeated dynamic frames can retain the arena's first allocations and a separate memory pattern.
                options.enable_mem_pattern = False
            self.session = ort.InferenceSession(model, sess_options=options, providers=self.providers)
            input_info = self.session.get_inputs()[0]
            self.input_shape = input_info.shape
            self.input_type = input_info.type
            self.input_channels = self.input_shape[1]
            self.input_dtype = np.float16 if self.input_type == "tensor(float16)" else None
            self.input_name = input_info.name
            output_info = self.session.get_outputs()[0]
            self.output_name = output_info.name
            if cpu_tile is not None:
                types = {"tensor(float)": np.float32, "tensor(float16)": np.float16}
                if (len(input_info.shape) != 4 or len(output_info.shape) != 4
                        or any(isinstance(size, int) for size in input_info.shape[2:])
                        or not isinstance(output_info.shape[1], int) or output_info.type not in types):
                    raise ValueError("CPU tiling requires dynamic NCHW input and float image output with known channels.")
                workers = cpu_threads if cpu_threads is not None else min(4, os.cpu_count() or 1)
                self.tile_plan = TilePlan(field, cpu_tile, workers, output_info.shape[1], types[output_info.type])
            self.run_tensor = self.run_onnxruntime
        else:
            raise ValueError(f"Unknown CPU backend: {cpu_backend}")

    def run_onnxruntime(self, tensor):
        return self.session.run([self.output_name], {self.input_name: tensor})[0]

    def fix_input_shape(self, input):
        image = input
        channels = self.input_channels
        if image.ndim == 2:
            return image[None, None]
        if image.ndim == 3:
            if image.shape[-1] == channels or (not isinstance(channels, int) and image.shape[-1] in (1, 3)):
                return image.transpose(2, 0, 1)[None]
            return image[None]
        if image.ndim == 4:
            if image.shape[1] != channels and (
                image.shape[-1] == channels or (not isinstance(channels, int) and image.shape[-1] in (1, 3))
            ):
                return image.transpose(0, 3, 1, 2)
            return image
        raise ValueError("Expected an HW, HWC, CHW, NHWC or NCHW image array.")

    def fix_output_shape(self, output):
        if isinstance(output, (list, tuple)):
            output, = output
        if output.ndim == 4:
            output = output.transpose(0, 2, 3, 1)
            return output[0] if output.shape[0] == 1 else output
        if output.ndim == 3:
            return output.transpose(1, 2, 0)
        if output.ndim == 2:
            return output[..., None]
        return output

    def run(self, input):
        tensor = np.ascontiguousarray(self.fix_input_shape(input), dtype=self.input_dtype)
        if self.tile_plan is not None:
            return self.fix_output_shape(self.tile_plan.run(self.run_tensor, tensor))
        return self.fix_output_shape(self.run_tensor(tensor))
