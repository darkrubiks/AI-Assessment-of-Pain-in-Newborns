"""Dataset-free regression checks for pain-sign classification and model passing."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ttt_optimized as pain


class PainSignTypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {
            f"synthetic_{name}": {"TrainAll": np.tile(signal[:, None], (1, 4))}
            for name, signal in {
                "stable": np.full(120, 0.1),
                "irregular": np.tile([0.1, 0.2, 0.3, 0.4], 30),
                "unstable": np.tile([0.1, 0.9], 60),
            }.items()
        }
        cls.thresholds = pain.get_thresholds_for_model("NCNN_FINAL")
        cls.model = pain._fit_notebook_sign_kmeans_all_models(
            {"NCNN_FINAL": cls.results}, mcdp=True, ma_window=1, duration_s=12.0,
        )
        if cls.model is None:
            raise AssertionError("KMeans fitting failed for distinct synthetic curves")

    def tearDown(self):
        plt.close("all")

    def pain_sign(self, name):
        return pain.compute_pain_sign(
            self.results[f"synthetic_{name}"], mcdp=True,
            theta_1=self.thresholds.theta_1, ma_window=1, duration_s=12.0,
        )

    def plot_kwargs(self):
        return dict(
            video_name="synthetic_stable", strip_img=None, ps=self.pain_sign("stable"),
            thresholds=self.thresholds, true_label=0, model_name="NCNN_FINAL",
            show_visual_rows=False, show_average_face_row=False,
            sign_kmeans_model=self.model,
        )

    def test_fitted_model_labels_reach_single_model_legend(self):
        for expected in ("stable", "irregular", "unstable"):
            with self.subTest(expected=expected):
                kwargs = self.plot_kwargs()
                kwargs["ps"] = self.pain_sign(expected)
                fig = pain._build_single_model_pain_sign_figure(**kwargs)
                label = fig.axes[1].lines[0].get_label()
                self.assertIn(pain._sign_type_to_ptbr(expected), label)
                self.assertNotIn(pain._sign_type_to_ptbr("indeterminate"), label)
                plt.close(fig)

    def test_near_threshold_curve_remains_indeterminate_with_model(self):
        ps = self.pain_sign("stable")
        ps.p_hat = np.full_like(ps.p_hat, self.thresholds.theta_1)
        with patch.object(self.model.kmeans, "predict") as predict:
            result = pain.classify_pain_sign_type(
                ps, theta_1=self.thresholds.theta_1, theta_3=self.thresholds.theta_3,
                sign_kmeans_model=self.model,
            )
        self.assertEqual(result, "indeterminate")
        predict.assert_not_called()

    def test_pdf_and_every_clip_frame_receive_supplied_model(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp, \
             patch.object(pain, "_build_single_model_pain_sign_figure") as build, \
             patch.object(pain, "_release_figure"), \
             patch.object(pain, "_figure_to_bgr_array", return_value=np.zeros((2, 2, 3), dtype=np.uint8)), \
             patch.object(pain, "_open_video_writer") as open_writer:
            pain.plot_pain_sign(save_path=Path(tmp) / "synthetic.pdf", **self.plot_kwargs())
            self.assertIs(build.call_args.kwargs["sign_kmeans_model"], self.model)
            build.reset_mock()
            pain.save_pain_sign_evolution_clip(
                save_path=Path(tmp) / "synthetic.mp4", clip_frame_step=60,
                hold_last_s=0, **self.plot_kwargs(),
            )
            self.assertGreater(build.call_count, 1)
            for call in build.call_args_list:
                self.assertIs(call.kwargs["sign_kmeans_model"], self.model)
            self.assertEqual(open_writer.return_value.write.call_count, build.call_count)
            open_writer.return_value.release.assert_called_once()

    def test_runner_fits_once_and_passes_model_to_pdf_and_clip(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp, \
             patch.object(pain, "_fit_notebook_sign_kmeans_all_models", wraps=pain._fit_notebook_sign_kmeans_all_models) as fit, \
             patch.object(pain, "plot_pain_sign") as pdf, \
             patch.object(pain, "save_pain_sign_evolution_clip") as clip:
            output = pain.run_pain_sign_report(
                self.results, model_name="NCNN_FINAL", path_icopevid_frames=tmp,
                xai_root=tmp, out_dir=tmp, mcdp=True, ma_window=1, duration_s=12.0,
                hide_visual_rows=True, show_average_face_row=False,
                include_region_curves=False, generate_video_clips=True,
            )
            fit.assert_called_once_with(
                {"NCNN_FINAL": self.results}, mcdp=True, ma_window=1, duration_s=12.0,
            )
            self.assertEqual(pdf.call_count, len(self.results))
            self.assertEqual(clip.call_count, len(self.results))
            fitted = pdf.call_args.kwargs["sign_kmeans_model"]
            self.assertIsNotNone(fitted)
            for call in [*pdf.call_args_list, *clip.call_args_list]:
                self.assertIs(call.kwargs["sign_kmeans_model"], fitted)
            baseline = [self.pain_sign(name) for name in ("stable", "irregular", "unstable")]
            np.testing.assert_array_equal(output["preds"], [ps.pred_label for ps in baseline])
            np.testing.assert_array_equal(output["probs"], [ps.p_summary for ps in baseline])
            np.testing.assert_array_equal(output["labels"], [0, 0, 0])

    def test_supplied_pooled_model_is_reused_in_parallel_workers(self):
        with patch.object(pain, "_fit_notebook_sign_kmeans_all_models") as fit, \
             patch.object(pain, "ProcessPoolExecutor") as pool:
            executor = pool.return_value.__enter__.return_value
            executor.submit.return_value.result.return_value = (0, 0, 0.1)
            pain.run_pain_sign_report(
                self.results, model_name="NCNN_FINAL", path_icopevid_frames="unused",
                xai_root="unused", out_dir="unused", mcdp=True,
                hide_visual_rows=True, include_region_curves=False,
                show_average_face_row=False, sign_kmeans_model=self.model, num_workers=2,
            )
            fit.assert_not_called()
            self.assertEqual(executor.submit.call_count, len(self.results))
            for call in executor.submit.call_args_list:
                self.assertIs(call.kwargs["sign_kmeans_model"], self.model)

    def test_insufficient_curves_warn_and_preserve_indeterminate_fallback(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp, \
             patch.object(pain, "plot_pain_sign") as pdf, \
             patch("builtins.print") as warn:
            pain.run_pain_sign_report(
                {"synthetic_stable": self.results["synthetic_stable"]},
                model_name="NCNN_FINAL", path_icopevid_frames=tmp, xai_root=tmp,
                out_dir=tmp, mcdp=True, hide_visual_rows=True,
                include_region_curves=False, show_average_face_row=False,
            )
            self.assertIsNone(pdf.call_args.kwargs["sign_kmeans_model"])
            self.assertIn("could not fit", warn.call_args.args[0])
            self.assertEqual(pain.classify_pain_sign_type(
                self.pain_sign("stable"), theta_1=self.thresholds.theta_1,
                theta_3=self.thresholds.theta_3, sign_kmeans_model=None,
            ), "indeterminate")

    def test_predictions_only_does_not_fit_model(self):
        with patch.object(pain, "_fit_notebook_sign_kmeans_all_models") as fit:
            pain.run_pain_sign_report(
                self.results, model_name="NCNN_FINAL", path_icopevid_frames="unused",
                xai_root="unused", out_dir="unused", mcdp=True,
                hide_visual_rows=True, include_region_curves=False,
                show_average_face_row=False, generate_pdf_reports=False,
                generate_video_clips=False,
            )
            fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
