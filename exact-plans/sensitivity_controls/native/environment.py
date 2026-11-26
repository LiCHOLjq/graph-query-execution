import os,json,subprocess,platform,hashlib,datetime
from pathlib import Path
def read(p):
 try:return Path(p).read_text().strip()
 except Exception:return None
files=['/proc/meminfo','/sys/devices/system/cpu/cpu4/cpufreq/scaling_governor','/sys/devices/system/cpu/cpu4/cpufreq/scaling_driver','/sys/devices/system/cpu/intel_pstate/no_turbo','/sys/devices/system/cpu/smt/active','/sys/devices/system/cpu/cpu4/topology/thread_siblings_list']
sources=[str(p) for p in Path('.').glob('*.cpp')]+['degree_sweep','dimension_gaussian','dimension_origin','allocation_control','protocol.json']
env={'prepared_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution':'Local Ubuntu WSL fallback, separate from NATIVE Xeon server','platform':platform.platform(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'compiler':subprocess.check_output(['g++','--version'],text=True),'flags':'-O3 -DNDEBUG -std=c++17 -Wall -Wextra','cpuinfo':read('/proc/cpuinfo').split('\n\n')[0],'lscpu':subprocess.check_output(['lscpu'],text=True),'settings':{p:read(p) for p in files},'hashes':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},'gc':'not applicable: native C++','timing':'steady_clock; one warmup per method/query, three shuffled measured repetitions; graph construction, oracle, CSV writes excluded; per-request allocations and all method work included; returned answer destruction excluded as NATIVE'}
Path('environment.json').write_text(json.dumps(env,indent=2))
