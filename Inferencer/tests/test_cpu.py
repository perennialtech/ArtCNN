import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import Engine
from inferencer import ImageInferencer
from tiling import model_footprint
from utils import self_ensemble_augment, self_ensemble_combine


ROOT = Path(__file__).resolve().parents[2]
STOCK = ROOT / "ONNX"
C16 = STOCK / "ArtCNN_C4F16.onnx"


class CpuInferenceTests(unittest.TestCase):
    def test_narrow_nchw_and_single_pixel_image(self):
        engine = Engine(C16, cpu_threads=1)
        for height, width in [(1, 1), (4, 1), (5, 3)]:
            array = np.linspace(0, 1, height * width, dtype=np.float32).reshape(height, width)
            single = engine.run(array[..., None])
            nchw = engine.run(array[None, None])
            self.assertEqual(single.shape, (height * 2, width * 2, 1))
            np.testing.assert_array_equal(single, nchw)

    def test_ensemble_restores_views_and_matches_stacked_mean(self):
        rng = np.random.default_rng(89)
        image = rng.random((7, 11, 3), dtype=np.float32)
        views = self_ensemble_augment(image)
        self.assertTrue(all(np.shares_memory(view, image) for view in views))
        np.testing.assert_allclose(self_ensemble_combine(iter(views)), image, atol=2e-7, rtol=0)
        for dtype in (np.float16, np.float32):
            predictions = rng.random((8, *image.shape), dtype=np.float32).astype(dtype)
            estimates = (self_ensemble_augment(prediction)[index] for index, prediction in enumerate(predictions))
            actual = self_ensemble_combine(estimates)
            np.testing.assert_array_equal(actual, np.mean(predictions, axis=0))

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "onnx is required for CPU tiling")
    def test_tiled_stock_outputs_match_across_borders(self):
        rng = np.random.default_rng(11)
        for name, radius in (("ArtCNN_C4F16", 7), ("ArtCNN_R8F64", 27), ("ArtCNN_R8F64_Chroma", 27)):
            model = STOCK / (name + ".onnx")
            with self.subTest(model=name):
                field = model_footprint(model)
                self.assertEqual((field.rows, field.columns), (radius, radius))
                channels = 3 if name.endswith("_Chroma") else 1
                array = rng.random((65, 77, channels), dtype=np.float32)
                whole = Engine(model, cpu_threads=1).run(array)
                tiled = Engine(model, cpu_threads=3, cpu_tile=37).run(array)
                np.testing.assert_array_equal(whole, tiled)

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "onnx is required for CPU tiling")
    def test_spatial_reduction_refused_by_tiler(self):
        import onnx
        from onnx import TensorProto, helper

        graph = helper.make_graph(
            [helper.make_node("GlobalAveragePool", ["in"], ["out"])], "unsupported",
            [helper.make_tensor_value_info("in", TensorProto.FLOAT, [1, 1, "h", "w"])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, 1, 1])],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "global.onnx"
            onnx.save(helper.make_model(graph), path)
            with self.assertRaisesRegex(ValueError, "GlobalAveragePool"):
                model_footprint(path)

    @unittest.skipUnless(importlib.util.find_spec("openvino"), "OpenVINO is optional")
    def test_openvino_handles_multiple_shapes_without_overwriting_prior_result(self):
        rng = np.random.default_rng(11)
        quick = Engine(C16, cpu_backend="openvino", cpu_threads=2)
        stock = Engine(C16, cpu_threads=1)
        for height, width in ((31, 45), (1, 3), (45, 31)):
            array = rng.random((height, width, 1), dtype=np.float32)
            result = quick.run(array)
            np.testing.assert_allclose(result, stock.run(array), atol=2e-6, rtol=0)
            saved = result.copy()
            quick.run(1 - array)
            np.testing.assert_array_equal(result, saved)

    def test_reused_sessions_in_combined_chroma_pipeline(self):
        chroma = STOCK / "ArtCNN_R8F64_Chroma.onnx"
        inferencer = ImageInferencer(C16, "luma-chroma", chroma, cpu_threads=1)
        rng = np.random.default_rng(11)
        for height, width in ((19, 27), (1, 3)):
            image = rng.random((height, width, 3), dtype=np.float32)
            result = inferencer.run(image)
            actual = ImageInferencer(C16, "luma-chroma", chroma, cpu_threads=1).run(image)
            self.assertEqual(result.shape, (2 * height, 2 * width, 3))
            np.testing.assert_array_equal(result, actual)


if __name__ == "__main__":
    unittest.main()
