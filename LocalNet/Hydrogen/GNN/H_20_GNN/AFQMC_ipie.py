import os
import sys
import ctypes
import time
import json
import numpy as np

# =============================================================================
# 0. CUDA HACK
# =============================================================================
cuda_root = "/oscar/rt/9.6/25/spack/x86_64_v3/cuda-12.9.0-cinrl2oeqemd3szbcakkugp2vtk2fh5t"
os.environ['CUDA_HOME'] = cuda_root
os.environ['CPATH'] = os.path.join(cuda_root, 'include')
os.environ['PATH'] = os.path.join(cuda_root, 'bin') + ":" + os.environ.get('PATH', '')
os.environ['NUMBA_CUDA_DRIVER'] = "/lib64/libcuda.so"
os.environ['LD_LIBRARY_PATH'] = f"{os.path.join(cuda_root, 'nvvm', 'lib64')}:{os.path.join(cuda_root, 'targets', 'x86_64-linux', 'lib')}:{os.path.join(cuda_root, 'lib64')}:/lib64:" + os.environ.get('LD_LIBRARY_PATH', '')
os.environ["IPIE_USE_GPU"] = "1"

try: 
    ctypes.CDLL(os.path.join(cuda_root, "nvvm", "lib64", "libnvvm.so"), mode=ctypes.RTLD_GLOBAL)
except: 
    pass

from pyscf import gto, scf
from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk
from ipie.qmc.afqmc import AFQMC
from ipie.utils.mpi import MPIHandler
import ipie.estimators.local_energy_sd
from ipie.analysis.autocorr import reblock_by_autocorr
from ipie.analysis.extraction import extract_observable

try:
    import cupy as cp
    has_cupy = True
except: 
    has_cupy = False

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
N_ATOMS = 20
SYSTEM_NAME = f"H{N_ATOMS}"
WALKERS = 2048
TEST_SEED = 999
NUM_BLOCKS = 70
STEPS_PER_BLOCK = 10

comm = MPIHandler()
rank = comm.rank
if has_cupy and cp.cuda.runtime.getDeviceCount() > 0: 
    cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()

chk_file, ham_file, wfn_file = f"scf_h{N_ATOMS}.chk", f"ham_h{N_ATOMS}.h5", f"wfn_h{N_ATOMS}.h5"

if rank == 0:
    mol = gto.M(atom=[("H", 0.74 * j, 0, 0) for j in range(N_ATOMS)], basis="sto-6g", verbose=0)
    mf = scf.UHF(mol)
    mf.chkfile = chk_file
    if not os.path.exists(chk_file) or not os.path.exists(wfn_file):
        mf.kernel()
        gen_ipie_input_from_pyscf_chk(chk_file, hamil_file=ham_file, wfn_file=wfn_file, verbose=0, chol_cut=1e-5)

comm.comm.Barrier()

afqmc = AFQMC.build_from_hdf5(num_elec=(N_ATOMS//2, N_ATOMS//2), ham_file=ham_file, wfn_file=wfn_file, num_blocks=NUM_BLOCKS, num_steps_per_block=STEPS_PER_BLOCK, num_walkers=WALKERS, seed=TEST_SEED, verbose=0)
if has_cupy: 
    afqmc.cuda = True
afqmc.mpi_handler = comm

local_walkers = WALKERS // comm.size
if rank < (WALKERS % comm.size): 
    local_walkers += 1
afqmc.nwalkers = local_walkers

# =============================================================================
# 2. MICRO-BENCHMARK INTERCEPT
# =============================================================================
baseline_tracker = {"total_time_sec": 0.0, "calls": 0}

def make_timed_func(orig_func):
    def timed_local_energy(system, hamiltonian, walkers, trial):
        if has_cupy: cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        
        res = orig_func(system, hamiltonian, walkers, trial)
        
        if has_cupy: cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        
        baseline_tracker["total_time_sec"] += (t1 - t0)
        baseline_tracker["calls"] += 1
        return res
    return timed_local_energy

target_names = [
    "local_energy_single_det_uhf", 
    "local_energy_single_det_batch_gpu", 
    "local_energy_single_det_uhf_batch_gpu",
    "local_energy_single_det_uhf_batch"
]

for name in target_names:
    if hasattr(ipie.estimators.local_energy_sd, name):
        original_func = getattr(ipie.estimators.local_energy_sd, name)
        timed_func = make_timed_func(original_func)
        
        setattr(ipie.estimators.local_energy_sd, name, timed_func)
        
        for mod_name, module in list(sys.modules.items()):
            if module and mod_name.startswith("ipie"):
                try:
                    for attr_name, attr_value in vars(module).items():
                        if attr_value is original_func: 
                            setattr(module, attr_name, timed_func)
                except: 
                    pass

# =============================================================================
# 3. RUN
# =============================================================================
if rank == 0: 
    print("\n" + "#"*60 + "\n### STARTING CLASSICAL IPIE GPU PRODUCTION RUN ###\n" + "#"*60)

afqmc.run()

# =============================================================================
# 4. METRICS & EXTRACTION
# =============================================================================
if rank == 0:
    # 1. Physics Extraction
    est_file = afqmc.estimators.filename if hasattr(afqmc.estimators, 'filename') else "estimates.0.h5"
    qmc_data = extract_observable(est_file, "energy")
    df_ac = reblock_by_autocorr(qmc_data["ETotal"][1:], verbose=0)
    final_energy = float(df_ac["ETotal_ac"].iloc[0])
    final_error = float(df_ac["ETotal_error_ac"].iloc[0])

    # 2. Timing
    avg_proxy_time = baseline_tracker["total_time_sec"] / baseline_tracker["calls"] if baseline_tracker["calls"] > 0 else 0

    # 3. Intrinsic Data Storage (ERI / Cholesky)
    hamil_bytes = 0
    
    # Locate the hamiltonian object robustly across ipie versions
    hamil_obj = getattr(afqmc, 'hamiltonian', getattr(afqmc, 'hamil', None))
    
    if hamil_obj is not None:
        if hasattr(hamil_obj, 'chol'):
            hamil_bytes += hamil_obj.chol.nbytes
            
        if hasattr(hamil_obj, 'H1'):
            if hasattr(hamil_obj.H1, 'nbytes'):
                hamil_bytes += hamil_obj.H1.nbytes
            else:
                hamil_bytes += hamil_obj.H1[0].nbytes + hamil_obj.H1[1].nbytes
    else:
        print("WARNING: Could not locate the Hamiltonian object to compute memory!")

    hamil_mb = hamil_bytes / (1024 ** 2)

    metrics = {
        "N_atoms": N_ATOMS,
        "backend": "Classical",
        "results": {
            "final_energy_ha": round(final_energy, 6),
            "final_error_ha": round(final_error, 6)
        },
        "local_energy_proxy": {
            "avg_time_sec": round(avg_proxy_time, 6),
            "cholesky_and_h1_mb": round(hamil_mb, 4),
            "total_intrinsic_memory_mb": round(hamil_mb, 4)
        }
    }
    
    with open(f"scaling_metrics_Classical_{SYSTEM_NAME}.json", "w") as f: 
        json.dump(metrics, f, indent=4)
        
    print(f"\n{'='*50}")
    print("CLASSICAL METRICS SUMMARY")
    print(f"{'='*50}")
    print(json.dumps(metrics, indent=4))