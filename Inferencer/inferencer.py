from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from inout import load_image, to_pil_image
from utils import (
    rgb2ycbcr_bt709,
    ycbcr2rgb_bt709,
    bilinear,
    stack,
    unstack,
    self_ensemble_augment,
    self_ensemble_combine,
)
from engine import Engine


class ImageInferencer:
    """Reuse the ONNX sessions for each frame and ensemble orientation."""

    def __init__(self, model, task, chroma_model=None, **engine_options):
        if task in {"chroma", "luma-chroma"} and chroma_model is None:
            raise ValueError(f"Chroma model is required for {task}.")
        if task != "chroma" and model is None:
            raise ValueError(f"Luma or color model is required for {task}.")
        self.task = task
        self.engine = Engine(chroma_model if task == "chroma" else model, **engine_options)
        self.chroma_engine = Engine(chroma_model, **engine_options) if task == "luma-chroma" else None

    def run(self, image):
        if self.task == "ycbcr":
            return ycbcr2rgb_bt709(self.engine.run(rgb2ycbcr_bt709(image)))
        if self.task == "chroma":
            ycbcr = rgb2ycbcr_bt709(image)
            y, _, _ = unstack(ycbcr)
            pred_cb, pred_cr = unstack(self.engine.run(ycbcr))
            return ycbcr2rgb_bt709(stack(y, pred_cb, pred_cr))
        if self.task == "luma-chroma":
            y, cb, cr = unstack(rgb2ycbcr_bt709(image))
            pred_y = self.engine.run(y)
            chroma = stack(pred_y, bilinear(cb, 2.0), bilinear(cr, 2.0))
            pred_cb, pred_cr = unstack(self.chroma_engine.run(chroma))
            return ycbcr2rgb_bt709(stack(pred_y, pred_cb, pred_cr))
        if self.task == "luma-luma-luma":
            return stack(*(self.engine.run(channel) for channel in unstack(image)))
        return self.engine.run(image)


def process_image(image, task, model, chroma_model=None):
    return ImageInferencer(model, task, chroma_model).run(image)


def run_inference(inputs, task, model, chroma_model=None, self_ensemble=False, mode="L", **engine_options):
    inferencer = None
    output_model = chroma_model if task == "chroma" else model
    pending = None
    with ThreadPoolExecutor(max_workers=1) as encoder:
        for input_path in inputs:
            image, icc_profile = load_image(input_path, mode=mode)
            if inferencer is None:
                inferencer = ImageInferencer(model, task, chroma_model, **engine_options)
            if self_ensemble:
                views = self_ensemble_augment(image)
                pred = self_ensemble_combine(inferencer.run(view) for view in views)
            else:
                pred = inferencer.run(image)
            output = to_pil_image(pred, mode=mode)
            # The encoder only needs the byte image. Release the runtime's buffer before the next frame.
            del pred
            if pending is not None:
                pending.result()
            filename = f"{input_path.stem}_{Path(output_model).stem}_{task}.png"
            pending = encoder.submit(output.save, filename, icc_profile=icc_profile)
        if pending is not None:
            pending.result()


def main():
    parser = ArgumentParser(description="ONNX Inferencer")
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("-m", "--model", help="ArtCNN Model")
    parser.add_argument("-cm", "--chroma-model", help="ArtCNN Chroma Model")
    parser.add_argument("-t", "--task", help="Task to perform", default="luma",
                        choices=["luma", "rgb", "ycbcr", "chroma", "luma-chroma", "luma-luma-luma"])
    parser.add_argument("--self-ensemble", action="store_true", help="Enable self-ensemble inference")
    cpu = parser.add_mutually_exclusive_group()
    cpu.add_argument("--cpu", action="store_true", help="Use ONNX Runtime's CPU provider")
    cpu.add_argument("--cpu-backend", choices=["onnxruntime", "openvino"], help="Select a CPU runtime")
    parser.add_argument("--cpu-threads", type=int, help="Set the CPU inference thread count")
    parser.add_argument("--cpu-tile", type=int, metavar="PIXELS", help="Run ONNX Runtime CPU inference in overlapping tiles")
    args = parser.parse_args()
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive.")

    if args.cpu_tile is not None and (args.cpu_tile < 1 or args.cpu_backend == "openvino"):
        parser.error("--cpu-tile requires a positive edge and the ONNX Runtime CPU backend.")

    input_path = Path(args.input)
    if input_path.is_dir():
        inputs = sorted(input_path.glob("*"))
    elif input_path.is_file():
        inputs = [input_path]
    else:
        raise FileNotFoundError(f"Input path {input_path} does not exist or is not a file/directory.")

    if args.task != "chroma" and not args.model:
        parser.error("--model is required for this task.")
    if args.task in {"chroma", "luma-chroma"} and not args.chroma_model:
        parser.error("--chroma-model is required for this task.")

    mode = "L" if args.task == "luma" else "RGB"
    backend = "onnxruntime" if args.cpu else args.cpu_backend
    run_inference(inputs, args.task, args.model, args.chroma_model, args.self_ensemble, mode,
                  cpu_backend=backend, cpu_threads=args.cpu_threads, cpu_tile=args.cpu_tile)


if __name__ == "__main__":
    main()
