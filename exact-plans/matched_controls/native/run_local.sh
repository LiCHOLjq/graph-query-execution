#!/bin/bash
set -euo pipefail
test ! -e k_sweep_raw.csv || { echo 'Refusing to overwrite an existing raw cohort; use a new output copy.' >&2; exit 2; }
g++ -O3 -DNDEBUG -std=c++17 -Wall -Wextra k_sweep.cpp -o k_sweep 2>compile.log
g++ -O3 -DNDEBUG -std=c++17 -Wall -Wextra clock_probe.cpp -o clock_probe 2>clock_compile.log
taskset -c 4 ./k_sweep test > tests.log
taskset -c 4 python3 environment.py
date --utc --iso-8601=seconds > STARTED
/usr/bin/time -v taskset -c 4 ./k_sweep k_sweep_raw.csv >run_stdout.log 2>run.log
taskset -c 4 ./clock_probe >clock_probe.json
date --utc --iso-8601=seconds > COMPLETE
