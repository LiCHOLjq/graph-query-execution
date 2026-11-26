import datetime,hashlib,json,os,platform,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent.parent
def read(p):
 try:return Path(p).read_text().strip()
 except OSError:return None
paths=['/proc/meminfo','/sys/devices/system/cpu/cpu4/cpufreq/scaling_governor','/sys/devices/system/cpu/cpu4/cpufreq/scaling_driver','/sys/devices/system/cpu/intel_pstate/no_turbo','/sys/devices/system/cpu/smt/active','/sys/devices/system/cpu/cpu4/topology/thread_siblings_list']
files=[*sorted((P/'source').glob('*')),P/'benchmark',P/'protocol.json']
data={'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'platform':platform.platform(),'affinity':sorted(os.sched_getaffinity(0)),'compiler':subprocess.check_output(['g++','--version'],text=True),'flags':'-O3 -DNDEBUG -std=c++17 -Wall -Wextra','cpuinfo':read('/proc/cpuinfo').split('\n\n')[0],'lscpu':subprocess.check_output(['lscpu'],text=True),'observed_settings':{f:read(f) for f in paths},'sha256':{str(x.relative_to(P)):hashlib.sha256(x.read_bytes()).hexdigest() for x in files if x.is_file()},'limitations':'WSL virtual CPU affinity only; no exclusive physical-core ownership, frequency or turbo control; no Python garbage collector in timed C++ kernels.'}
(P/('environment_'+(sys.argv[1] if len(sys.argv)>1 else 'before')+'.json')).write_text(json.dumps(data,indent=2))
