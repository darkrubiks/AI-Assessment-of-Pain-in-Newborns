"""Synthetic, dataset-free checks for the pain-sign display options and layout.

Run from the repository root: python -m unittest discover -s tests -p test_pain_sign_layout.py
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ttt_optimized as pain


def synthetic_inputs():
    time = np.linspace(0.0, 20.0, 161)
    probability = 0.5 + 0.25 * np.sin(time)
    ps = pain.PainSignResult(time, probability, np.full(time.size, 0.02), 0.5, 0.02, 1)
    regions = pd.DataFrame({"time_s": time})
    for idx, name in enumerate(pain.REGION_COLOR_MAP):
        regions[name] = 1.1 + np.sin(time / 3.0 + idx)
    # Colored tiles stand in for images; no patient data or face assets are used.
    tile = np.zeros((64, 64, 3), dtype=np.float32)
    tile[..., 0] = np.linspace(0.1, 0.8, 64)[None, :]
    tile[..., 1] = np.linspace(0.8, 0.1, 64)[:, None]
    strip_row = np.tile(tile, (1, 8, 1))
    strip = np.vstack([strip_row, strip_row])
    return ps, regions, tile, strip


class PainSignLayoutTests(unittest.TestCase):
    def setUp(self):
        self.ps, self.regions, self.tile, self.strip = synthetic_inputs()
        self.thresholds = pain.get_thresholds_for_model("NCNN_FINAL")

    def tearDown(self):
        plt.close("all")

    def figure(self, **kwargs):
        return pain._build_single_model_pain_sign_figure(
            video_name="synthetic_Pain", ps=self.ps, thresholds=self.thresholds,
            true_label=1, model_name="NCNN_FINAL", region_df=self.regions,
            strip_img=self.strip, **kwargs,
        )

    def test_legends_do_not_overlap_data_or_header(self):
        for scale in (0.65, 0.9, 1.0):
            with self.subTest(scale=scale):
                fig = self.figure(show_visual_rows=False, show_average_face_row=False, figure_scale=scale)
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                header, signal, regions = fig.axes
                header_legend = header.get_legend().get_window_extent(renderer)
                title = header.texts[0].get_window_extent(renderer)
                region_legend = regions.get_legend().get_window_extent(renderer)
                self.assertFalse(title.overlaps(header_legend))
                self.assertFalse(header_legend.overlaps(signal.get_window_extent(renderer)))
                self.assertFalse(region_legend.overlaps(regions.get_window_extent(renderer)))
                self.assertFalse(region_legend.overlaps(signal.get_window_extent(renderer)))
                self.assertGreaterEqual(region_legend.y0, regions.get_window_extent(renderer).y1)
                self.assertGreaterEqual(region_legend.x0, regions.get_window_extent(renderer).x0)
                self.assertLessEqual(region_legend.x1, regions.get_window_extent(renderer).x1)
                # Every entry occupies the same horizontal row above the plot.
                text_y = [text.get_window_extent(renderer).y0 for text in regions.get_legend().texts]
                np.testing.assert_allclose(text_y, text_y[0])
                self.assertGreaterEqual(region_legend.x0, 0)
                self.assertLessEqual(region_legend.x1, fig.bbox.x1)
                self.assertGreaterEqual(region_legend.y0, 0)
                self.assertLessEqual(region_legend.y1, fig.bbox.y1)
                self.assertEqual(len(regions.get_legend().texts), len(pain.REGION_COLOR_MAP))
                self.assertLess(
                    fig.bbox.width - signal.get_window_extent(renderer).x1,
                    0.03 * fig.bbox.width,
                )
                self.assertAlmostEqual(signal.get_position().x0, regions.get_position().x0)
                self.assertAlmostEqual(signal.get_position().x1, regions.get_position().x1)
                np.testing.assert_array_equal(signal.lines[0].get_ydata(), self.ps.p_hat)
                self.assertNotIsInstance(fig.get_layout_engine(), matplotlib.layout_engine.TightLayoutEngine)
                plt.close(fig)

    def test_optional_panels_and_animation_geometry(self):
        for visual_rows in (False, True):
            for average_face in (False, True):
                with self.subTest(visual_rows=visual_rows, average_face=average_face):
                    common = dict(
                        show_visual_rows=visual_rows, show_average_face_row=average_face,
                        live_frame_img=self.tile, live_overlay_img=self.tile,
                        live_average_face_img=self.tile,
                    )
                    first = self.figure(reveal_until_s=0.0, **common)
                    last = self.figure(reveal_until_s=20.0, **common)
                    first.canvas.draw()
                    last.canvas.draw()
                    expected_panels = (2 if visual_rows else 0) + int(average_face)
                    self.assertEqual(sum(bool(ax.images) for ax in last.axes), expected_panels)
                    np.testing.assert_allclose(first.get_size_inches(), last.get_size_inches())
                    np.testing.assert_allclose(
                        [ax.get_position().bounds for ax in first.axes],
                        [ax.get_position().bounds for ax in last.axes],
                    )
                    plt.close(first)
                    plt.close(last)

    def test_strip_rows_align_with_time_plots_and_keep_square_pixels(self):
        fig = self.figure(show_visual_rows=True, average_face_row_img=self.strip[:64])
        fig.canvas.draw()
        signal = fig.axes[1]
        for ax in (ax for ax in fig.axes if ax.images):
            self.assertAlmostEqual(ax.get_position().x0, signal.get_position().x0)
            self.assertAlmostEqual(ax.get_position().x1, signal.get_position().x1)
            box = ax.get_window_extent()
            height, width = ax.images[0].get_array().shape[:2]
            self.assertAlmostEqual(box.width / width, box.height / height)

    def test_live_panels_keep_time_axis_visible_without_label_overlap(self):
        for has_regions in (False, True):
            for show_frames in (False, True):
                for scale in (0.65, 0.9):
                    with self.subTest(regions=has_regions, frames=show_frames, scale=scale):
                        fig = pain._build_single_model_pain_sign_figure(
                            video_name="synthetic_Pain", ps=self.ps, thresholds=self.thresholds,
                            true_label=1, model_name="NCNN_FINAL", strip_img=self.strip,
                            region_df=self.regions if has_regions else None,
                            show_visual_rows=show_frames, show_average_face_row=True,
                            live_frame_img=self.tile, live_overlay_img=self.tile,
                            live_average_face_img=self.tile, figure_scale=scale,
                            reveal_until_s=10.0,
                        )
                        fig.canvas.draw()
                        renderer = fig.canvas.get_renderer()
                        time_axes = [ax for ax in fig.axes if ax.get_xlabel() == "Tempo [s]"]
                        self.assertEqual(len(time_axes), 1)
                        ax_time = time_axes[0]
                        tick_labels = ax_time.get_xticklabels()
                        self.assertTrue(tick_labels)
                        self.assertTrue(all(label.get_visible() for label in tick_labels))
                        time_text = [ax_time.xaxis.label, *tick_labels]
                        for ax_image in (ax for ax in fig.axes if ax.images):
                            for text in time_text:
                                box = text.get_window_extent(renderer)
                                self.assertFalse(box.overlaps(ax_image.get_window_extent(renderer)))
                                self.assertFalse(box.overlaps(ax_image.title.get_window_extent(renderer)))
                        plt.close(fig)

    def test_runner_passes_toggle_to_pdf_and_clip_without_changing_predictions(self):
        outputs = []
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            for average_face in (False, True):
                with self.subTest(average_face=average_face), \
                     patch.object(pain, "load_video_strip", return_value=self.strip) as load_strip, \
                     patch.object(pain, "extract_region_scores_video", return_value=self.regions), \
                     patch.object(pain, "plot_pain_sign") as pdf, \
                     patch.object(pain, "save_pain_sign_evolution_clip") as clip:
                    output = pain.run_pain_sign_report(
                        {"synthetic_Pain": {"TrainAll": np.tile(self.ps.p_hat[:, None], (1, 4))}},
                        model_name="NCNN_FINAL", path_icopevid_frames=tmp,
                        xai_root=tmp, out_dir=tmp, mcdp=True,
                        hide_visual_rows=True, show_average_face_row=average_face,
                        generate_video_clips=True,
                    )
                    self.assertIs(pdf.call_args.kwargs["show_average_face_row"], average_face)
                    self.assertIs(clip.call_args.kwargs["show_average_face_row"], average_face)
                    self.assertEqual(load_strip.call_count, int(average_face))
                    outputs.append(output)
        for key in ("preds", "probs", "labels"):
            np.testing.assert_array_equal(outputs[0][key], outputs[1][key])

    def test_hidden_average_face_does_not_load_missing_assets(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp, \
             patch.object(pain, "_load_average_face_assets", side_effect=AssertionError("Assets loaded")):
            pdf_path = Path(tmp) / "synthetic.pdf"
            pain.plot_pain_sign(
                video_name="synthetic_Pain", strip_img=self.strip, ps=self.ps,
                thresholds=self.thresholds, true_label=1, model_name="NCNN_FINAL",
                region_df=self.regions, save_path=pdf_path,
                show_visual_rows=False, show_average_face_row=False,
                average_face_background_path=Path(tmp) / "missing.png",
                average_face_keypoints_path=Path(tmp) / "missing.pkl",
            )
            self.assertTrue(pdf_path.is_file())

    def test_missing_dataset_has_clear_error(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "Dataset frames not found"):
                pain.run_pain_sign_report(
                    {}, model_name="NCNN_FINAL", path_icopevid_frames=Path(tmp) / "missing",
                    xai_root=tmp, out_dir=tmp, mcdp=True,
                )


if __name__ == "__main__":
    unittest.main()
