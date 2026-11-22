from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Footprint:
    """Scale and maximum input-pixel dependency of a graph tensor."""

    scale: int = 1
    rows: int = 0
    columns: int = 0


def model_footprint(path):
    """Find the halo of a spatially local, fully convolutional ONNX graph."""
    try:
        import onnx
        from onnx import helper, numpy_helper
    except ImportError as error:
        raise RuntimeError("CPU tiling requires the onnx package.") from error

    graph = onnx.load(path, load_external_data=False).graph
    constants = {tensor.name: tensor for tensor in graph.initializer}
    inputs = [node.name for node in graph.input if node.name not in constants]
    if len(inputs) != 1 or len(graph.output) != 1:
        raise ValueError("CPU tiling requires one image input and one output.")
    fields = {inputs[0]: Footprint()}

    def reject(node):
        raise ValueError(f"CPU tiling does not support {node.op_type} in {node.name or node.output[0]}.")

    def spatially_constant(name):
        return name in constants and all(size == 1 for size in constants[name].dims[-2:])

    pointwise = {"Add", "Cast", "Clip", "Div", "Identity", "LeakyRelu", "Mul", "Relu", "Sigmoid", "Sub"}
    for node in graph.node:
        if node.domain not in ("", "ai.onnx") or len(node.output) != 1:
            reject(node)
        attrs = {item.name: helper.get_attribute_value(item) for item in node.attribute}
        sources = [fields[name] for name in node.input if name in fields]
        if node.op_type == "Constant" and "value" in attrs:
            constants[node.output[0]] = attrs["value"]
            continue
        if not sources or any(field.scale != sources[0].scale for field in sources):
            reject(node)
        scale = sources[0].scale
        rows = max(field.rows for field in sources)
        columns = max(field.columns for field in sources)

        if node.op_type == "Conv":
            weights = constants.get(node.input[1])
            if len(sources) != 1 or scale != 1 or weights is None or len(weights.dims) != 4:
                reject(node)
            dilations = attrs.get("dilations", [1, 1])
            reach_y, reach_x = ((size - 1) * dilation for size, dilation in zip(weights.dims[-2:], dilations))
            if reach_y % 2 or reach_x % 2 or attrs.get("strides", [1, 1]) != [1, 1]:
                reject(node)
            pad_y, pad_x = reach_y // 2, reach_x // 2
            auto_pad = attrs.get("auto_pad", b"NOTSET")
            if auto_pad not in (b"SAME_UPPER", b"SAME_LOWER"):
                if auto_pad not in (b"NOTSET", b"VALID") or attrs.get("pads", [0] * 4) != [pad_y, pad_x, pad_y, pad_x]:
                    reject(node)
            if auto_pad == b"VALID" and (pad_y or pad_x):
                reject(node)
            if len(node.input) == 3 and node.input[2] not in constants:
                reject(node)
            rows, columns = rows + pad_y, columns + pad_x
        elif node.op_type == "DepthToSpace":
            scale *= attrs["blocksize"]
        elif node.op_type == "Resize":
            if (set(attrs) - {"mode", "coordinate_transformation_mode", "nearest_mode"}
                    or len(node.input) != 3 or node.input[1]
                    or attrs.get("mode") != b"linear"
                    or attrs.get("coordinate_transformation_mode", b"half_pixel") != b"half_pixel"
                    or node.input[2] not in constants):
                reject(node)
            factors = numpy_helper.to_array(constants[node.input[2]], base_dir=str(Path(path).parent))
            if not np.array_equal(factors, [1, 1, 2, 2]):
                reject(node)
            scale *= 2
            rows, columns = rows + 1, columns + 1
        elif node.op_type in pointwise:
            if any(name and name not in fields and not spatially_constant(name) for name in node.input):
                reject(node)
        else:
            reject(node)
        fields[node.output[0]] = Footprint(scale, rows, columns)

    output = graph.output[0].name
    if output not in fields:
        raise ValueError("CPU tiling requires an output derived from the image.")
    return fields[output]


@dataclass(frozen=True)
class TilePlan:
    field: Footprint
    edge: int
    workers: int
    channels: int
    dtype: np.dtype

    def run(self, infer, tensor):
        batch, _, height, width = tensor.shape
        if max(height, width) <= self.edge:
            return infer(tensor)
        scale = self.field.scale
        output = np.empty((batch, self.channels, height * scale, width * scale), dtype=self.dtype)

        def evaluate(rectangle):
            y, x = rectangle
            y_end, x_end = min(y + self.edge, height), min(x + self.edge, width)
            top, left = max(0, y - self.field.rows), max(0, x - self.field.columns)
            bottom = min(height, y_end + self.field.rows)
            right = min(width, x_end + self.field.columns)
            tile = np.ascontiguousarray(tensor[:, :, top:bottom, left:right])
            pred = infer(tile)
            output[:, :, y*scale:y_end*scale, x*scale:x_end*scale] = pred[
                :, :, (y-top)*scale:(y_end-top)*scale, (x-left)*scale:(x_end-left)*scale
            ]

        rectangles = [(y, x) for y in range(0, height, self.edge) for x in range(0, width, self.edge)]
        if self.workers == 1:
            for rectangle in rectangles:
                evaluate(rectangle)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for _ in pool.map(evaluate, rectangles):
                    pass
        return output
