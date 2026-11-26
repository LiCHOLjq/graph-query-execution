#!/bin/bash
set -euo pipefail
g++ -O3 -DNDEBUG -std=c++17 -Wall -Wextra degree_sweep.cpp -o degree_sweep
g++ -O3 -DNDEBUG -std=c++17 -Wall -Wextra dimension_gaussian.cpp -o dimension_gaussian
g++ -O3 -DNDEBUG -std=c++17 -Wall -Wextra dimension_origin.cpp -o dimension_origin
g++ -O3 -DNDEBUG -std=c++17 -Wall -Wextra allocation_control.cpp -o allocation_control
taskset -c 4 ./degree_sweep test > degree_tests.log
taskset -c 4 ./dimension_gaussian test > dimension_tests.log
taskset -c 4 ./allocation_control test > allocation_tests.log
taskset -c 4 python3 environment.py
/usr/bin/time -v taskset -c 4 ./dimension_origin sensitivity dimension_origin_raw.csv >dimension_origin.log 2>dimension_origin_run.log
/usr/bin/time -v taskset -c 4 ./dimension_gaussian sensitivity dimension_gaussian_raw.csv >dimension.log 2>dimension_run.log
/usr/bin/time -v taskset -c 4 ./degree_sweep degree_sweep_raw.csv >degree.log 2>degree_run.log
/usr/bin/time -v taskset -c 4 ./allocation_control allocation_raw.csv >allocation.log 2>allocation_run.log
g++ -O3 -DNDEBUG -std=c++17 clock_probe.cpp -o clock_probe
taskset -c 4 ./clock_probe > clock_probe.json
date --utc --iso-8601=seconds > COMPLETE
