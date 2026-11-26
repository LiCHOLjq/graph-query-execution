"""Check the integrated research artifact in temporary working directories."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent


def run(folder, *args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", MPLBACKEND="Agg")
    subprocess.run([sys.executable, *args], cwd=folder, env=env, check=True)


def main():
    for line in (ROOT / "CHECKSUMS.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise SystemExit("Checksum mismatch: " + name)
    exact = ROOT / "exact-plans"
    run(exact, "verify_paper_statistics.py")
    run(exact, "-m", "unittest", "discover", "-s", "artifact", "-p", "test_engine.py")
    run(ROOT, "comparison/replay_routes.py", "--check")
    with TemporaryDirectory(prefix="graph-query-check-") as temporary:
        work = Path(temporary)
        scratch = work / "exact"
        shutil.copytree(exact, scratch)
        run(scratch, "artifact/audit_results.py", "artifact/results/main")
        run(scratch, "matched_controls/native/analyze.py")
        run(scratch, "matched_controls/callbacks/analyze_callbacks.py")
        run(scratch, "ordered_access/analyze.py", "--history-root", ".")
        output = work / "policy"
        run(ROOT / "policy-selection", "reproduce.py", "--output", str(output))
        report = json.loads((output / "reproduction_report.json").read_text())
        if report["status"] != "PASS" or len(report["reference_tables"]) != 40:
            raise SystemExit("Policy analysis did not reproduce the expected 40 tables")
    print("PASS: execution plans, policy selection, and joint replay")


if __name__ == "__main__":
    main()
