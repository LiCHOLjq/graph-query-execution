import datetime, hashlib, json, os, platform, subprocess
from pathlib import Path

def read(p):
    try: return Path(p).read_text().strip()
    except OSError: return None

files = ['/sys/devices/system/cpu/cpu4/cpufreq/scaling_governor',
         '/sys/devices/system/cpu/cpu4/cpufreq/scaling_driver',
         '/sys/devices/system/cpu/intel_pstate/no_turbo',
         '/sys/devices/system/cpu/smt/active',
         '/sys/devices/system/cpu/cpu4/topology/thread_siblings_list']
sources = ['k_sweep.cpp','strong_base.cpp','k_sweep','clock_probe.cpp','clock_probe','protocol.json','run_local.sh']
env = dict(prepared_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    execution='MATCHED local Ubuntu WSL fallback, independent matched cohort; do not pool absolute times with NATIVE/SENSITIVITY',
    platform=platform.platform(), cpu_affinity=sorted(os.sched_getaffinity(0)),
    compiler=subprocess.check_output(['g++','--version'],text=True),
    flags='-O3 -DNDEBUG -std=c++17 -Wall -Wextra',
    cpuinfo=read('/proc/cpuinfo').split('\n\n')[0],
    lscpu=subprocess.check_output(['lscpu'],text=True),
    settings={p:read(p) for p in files},
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
    timing='steady_clock; one warmup per method/query; 3 shuffled repetitions; graph construction, oracle, diagnostic traces and CSV writes excluded; per-request allocations and all method work included; returned-answer destruction excluded as NATIVE/SENSITIVITY',
    limitations='WSL CPU affinity is virtual CPU affinity. Exclusive physical-core ownership and frequency stability are not enforced.')
Path('environment.json').write_text(json.dumps(env,indent=2))
