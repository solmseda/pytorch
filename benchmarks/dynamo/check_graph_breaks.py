import argparse
import os
import sys
import textwrap

import pandas as pd


METRICS = ("graph_breaks", "recompiles")


# Hack to have something similar to DISABLED_TEST. These models are flaky.

flaky_models = {
    "yolov3",
    "detectron2_maskrcnn_r_101_c4",
    "XGLMForCausalLM",  # discovered in https://github.com/pytorch/pytorch/pull/128148
    "detectron2_fcos_r_50_fpn",
}


def get_field(csv, model_name: str, field: str):
    try:
        value = csv.loc[csv["name"] == model_name][field].item()
        if isinstance(value, float) and pd.isna(value):
            return None
        return value
    except Exception:
        return None


def check_graph_breaks(actual_csv, expected_csv, expected_filename):
    results = {
        metric: {"failed": [], "improved": []} for metric in METRICS
    }

    def evaluate_metric(
        model: str,
        metric: str,
        actual,
        expected,
        flaky: bool,
    ) -> tuple[str, bool, str | None]:
        if actual is None:
            return "SKIP", False, None

        if expected is None:
            results[metric]["improved"].append(model)
            status = "MISSING:"
            msg = f"{model:34}  {status:19} {metric}={actual}, expected=None"
            return status, True, msg

        if actual == expected:
            status = "PASS_BUT_FLAKY" if flaky else "PASS"
            if flaky:
                msg = f"{model:34}  {status:19} {metric}={actual}, expected={expected}"
                return status, True, msg
            return status, False, None

        if actual > expected:
            if flaky:
                status = "FAIL_BUT_FLAKY:"
            else:
                status = "FAIL:"
                results[metric]["failed"].append(model)
            msg = f"{model:34}  {status:19} {metric}={actual}, expected={expected}"
            return status, True, msg

        # actual < expected
        if flaky:
            status = "IMPROVED_BUT_FLAKY:"
        else:
            status = "IMPROVED:"
            results[metric]["improved"].append(model)
        msg = f"{model:34}  {status:19} {metric}={actual}, expected={expected}"
        return status, True, msg

    if "rocm" in expected_filename:
        flaky_models.update(
            {
                "alexnet",
                "demucs",
                "densenet121",
                "detectron2_fcos_r_50_fpn",
                "doctr_det_predictor",
                "doctr_reco_predictor",
                "levit_128",
                "llava",
                "microbench_unbacked_tolist_sum",
                "resnet50",
                "resnet152",
                "sam",
                "sam_fast",
                "stable_diffusion_text_encoder",
                "stable_diffusion_unet",
                "timm_efficientdet",
                "torchrec_dlrm",
                "vgg16",
                # LLM
                "meta-llama/Llama-3.2-1B",
                "google/gemma-2-2b",
                "google/gemma-3-4b-it",
                "openai/whisper-tiny",
                "Qwen/Qwen3-0.6B",
                "mistralai/Mistral-7B-Instruct-v0.3",
                "openai/gpt-oss-20b",
            }
        )

    for model in actual_csv["name"]:
        graph_breaks = get_field(actual_csv, model, "graph_breaks")
        expected_graph_breaks = get_field(expected_csv, model, "graph_breaks")
        recompiles = get_field(actual_csv, model, "recompiles")
        expected_recompiles = get_field(expected_csv, model, "recompiles")
        flaky = model in flaky_models

        outputs = []
        for metric, actual, expected in (
            ("graph_breaks", graph_breaks, expected_graph_breaks),
            ("recompiles", recompiles, expected_recompiles),
        ):
            status, should_print, msg = evaluate_metric(
                model, metric, actual, expected, flaky
            )
            if should_print and msg:
                outputs.append(msg)

        if outputs:
            for line in outputs:
                print(line)

    msg = ""
    has_changes = False
    for metric in METRICS:
        failed = results[metric]["failed"]
        improved = results[metric]["improved"]
        if not failed and not improved:
            continue
        has_changes = True
        title = metric.replace("_", " ")
        if failed:
            msg += textwrap.dedent(
                f"""
            Error: {len(failed)} models have new dynamo {title} regressions:
                {" ".join(failed)}

            """
            )
        if improved:
            msg += textwrap.dedent(
                f"""
            Improvement: {len(improved)} models improved dynamo {title}:
                {" ".join(improved)}

            """
            )
    if msg:
        sha = os.getenv("SHA1", "{your CI commit sha}")
        msg += textwrap.dedent(
            f"""
        If this change is expected, you can update `{expected_filename}` to reflect the new baseline.
        from pytorch/pytorch root, run
        `python benchmarks/dynamo/ci_expected_accuracy/update_expected.py {sha}`
        and then `git add` the resulting local changes to expected CSVs to your commit.
        """
        )
    return has_changes, msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", type=str, required=True)
    parser.add_argument("--expected", type=str, required=True)
    args = parser.parse_args()

    actual = pd.read_csv(args.actual)
    expected = pd.read_csv(args.expected)

    failed, msg = check_graph_breaks(actual, expected, args.expected)
    if failed:
        print(msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
