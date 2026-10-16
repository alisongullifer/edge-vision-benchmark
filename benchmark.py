import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from train import SmallCNN, latency, split_data

def benchmark(args):
    source = Path("results/local")
    original_report = json.loads(
        (source / "report.json").read_text()
    )

    # Match the original benchmark settings.
    settings = original_report["environment"]
    torch.set_num_threads(settings["threads"])
    torch.backends.quantized.engine = settings["quantization_engine"]

    # Restore the original trained FP32 model.
    fp32 = SmallCNN().eval()
    fp32.load_state_dict(
        torch.load(source / "cnn_fp32.pt", weights_only=True)
    )

    # Build the matching quantized architecture, then restore
    # the saved state. Only Linear layers are quantized.
    linear_int8 = torch.ao.quantization.quantize_dynamic(
        SmallCNN().eval(),
        {nn.Linear},
        dtype=torch.qint8,
    )
    linear_int8.load_state_dict(
        torch.load(source / "cnn_linear_int8.pt", weights_only=True)
    )

    # Use the same test image as the original benchmark.
    x, _, _, _, _ = split_data(original_report["seed"])
    test_index = original_report["split_indices"]["test"][0]
    sample = x[test_index:test_index + 1]

    results = []

    # Alternate timing order across runs.
    variants = [
        ("cnn_fp32", fp32),
        ("cnn_linear_int8", linear_int8),
    ]
    if args.run % 2 == 0:
        variants.reverse()

    for name, model in variants:
        results.append({
            "model": name,
            **latency(
                model,
                sample,
                settings["repeats_per_round"],
            ),
        })

    report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "run": args.run,
        "background_activity": args.activity,
        "retrained": False,
        "quantization_scope": "Linear layers only; convolutions FP32",
        "settings": {
            "threads": settings["threads"],
            "quantization_engine": settings["quantization_engine"],
            "batch_size": 1,
            "warmups": 30,
            "rounds": 5,
            "repeats_per_round": settings["repeats_per_round"],
            "test_image_index": test_index,
        },
        "timing_order": [name for name, _ in variants],
        "results": results,
    }

    output = Path("results/day5")
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"benchmark_run_{args.run}.json"
    path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--activity", required=True)
    benchmark(parser.parse_args())