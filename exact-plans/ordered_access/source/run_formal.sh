#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
test ! -e protocol.json || { echo 'Refusing to overwrite a frozen measurement.' >&2; exit 2; }
g++ -O3 -DNDEBUG -std=c++17 -Wall -Wextra source/benchmark.cpp -o benchmark 2>compile.log
g++ -O3 -DNDEBUG -std=c++17 -Wall -Wextra source/clock_probe.cpp -o clock_probe 2>clock_compile.log
taskset -c 4 ./benchmark test >tests.log
python3 source/freeze_protocol.py >protocol_freeze.log
taskset -c 4 python3 source/environment.py before
taskset -c 4 ./clock_probe >clock_before.json
date --utc --iso-8601=seconds >STARTED
for cohort in matched dimension real main degree; do
 /usr/bin/time -v taskset -c 4 ./benchmark "$cohort" results inputs >"${cohort}_stdout.log" 2>"${cohort}_run.log"
done
taskset -c 4 ./clock_probe >clock_after.json
taskset -c 4 python3 source/environment.py after
date --utc --iso-8601=seconds >COMPLETE
