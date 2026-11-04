"""Re-render existing data into a new subdirectory; never changes experiment data."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .report import make_plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--subdirectory", default="figures_reviewed")
    args = parser.parse_args()
    root = args.run_directory.resolve()
    output = root / args.subdirectory
    output.mkdir(exist_ok=False)
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    with (root / "expected.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("question_penalty", "success", "energy", "questions", "observations",
                    "aborts", "objective", "root_plan_ms", "expanded_states"):
            if row[key] != "":
                row[key] = float(row[key])
    make_plots(output, cfg, manifest, rows, source_directory=root)
    provenance_file = output / "figure_provenance.json"
    provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
    provenance["source_sha256"] = hashlib.sha256((root / "expected.csv").read_bytes()).hexdigest()
    provenance["plot_code_sha256"] = hashlib.sha256(Path(__file__).with_name("report.py").read_bytes()).hexdigest()
    provenance["revision_reason"] = "Show coincident MAP point with an open marker; use a nonnegative color scale for all-zero gaps. Raw results unchanged."
    provenance_file.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
