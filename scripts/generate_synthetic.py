"""Generate fixed-seed synthetic scenario samples and their manifests."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bias_detector.synthetic import SCENARIOS, generate_scenario, scenario_expected_status

SAMPLES = ROOT / "data" / "samples"
SAMPLES.mkdir(parents=True, exist_ok=True)

manifest = {"seed_base": 0, "runs_per_scenario": 100, "scenarios": {}}
for index, (name, description) in enumerate(SCENARIOS.items(), start=1):
    frame = generate_scenario(name, seed=42)
    output_name = f"synthetic_s{index}.csv"
    frame.to_csv(SAMPLES / output_name, index=False)
    manifest["scenarios"][name] = {
        "description": description,
        "file": output_name,
        "rows": len(frame),
        "expected_status": scenario_expected_status(name),
        "evaluated_seeds": list(range(100)),
    }
(SAMPLES / "synthetic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
