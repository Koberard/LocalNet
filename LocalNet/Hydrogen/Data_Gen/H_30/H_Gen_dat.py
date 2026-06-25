import numpy as np
import os
from pyscf import gto, scf
from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk
from ipie.qmc.afqmc import AFQMC
from ipie.estimators.estimator_base import EstimatorBase
from ipie.estimators.energy import local_energy_batch
from ipie.utils.mpi import MPIHandler

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
N_ATOMS = 30           # Target System (e.g., 10, 20)
WALKERS = 2048         # Total Walkers across all ranks

# ENSEMBLE CONFIGURATION
NUM_RUNS          = 50  # Independent runs
TOTAL_PROD_BLOCKS = 70 # Production blocks per run
BURN_IN_BLOCKS    = 0  # Burn-in blocks per run
TOTAL_RUN_LENGTH  = BURN_IN_BLOCKS + TOTAL_PROD_BLOCKS
STEPS_PER_BLOCK   = 10
CHECKPOINT_FREQ   = 50 # Save to disk every 50 blocks
SAMPLES_PER_BLOCK = 10 # Number of walkers to sample per block

# =============================================================================
# 2. ROBUST ESTIMATOR WITH VERIFICATION
# =============================================================================
class DensityExtractor(EstimatorBase):
    def __init__(self, ham, trial, C_a, C_b, h1_ao, S_ao, rank, run_idx, burn_in=0, num_samples=10, verbose=True):
        super().__init__()
        self._shape = (1,) 
        self._scalar_estimator = True 
        
        self.rank = rank
        self.run_idx = run_idx
        self.burn_in = burn_in
        self.num_samples = num_samples
        self.verbose = verbose
        self.counter = 0
        
        # Save PySCF matrices
        self.C_a   = C_a
        self.C_b   = C_b
        self.h1_ao = h1_ao
        self.S_ao  = S_ao
        
        self.nalpha = trial.nalpha
        self.nbeta  = trial.nbeta
        self.ecore  = ham.ecore
        
        # We need the trial wavefunction to explicitly build the 1-RDM
        self.Psi_T_a = trial.psi[:, :trial.nalpha].astype(np.complex128)
        self.Psi_T_b = trial.psi[:, trial.nalpha:].astype(np.complex128)
        
        self._buffer = {"GA": [], "GB": [], "E": [], "W": []}
        
        self.use_gpu = False
        try:
            import cupy as cp
            self.cp = cp
            self.use_gpu = True
            self.C_a_gpu = cp.array(self.C_a)
            self.C_b_gpu = cp.array(self.C_b)
            self.Psi_T_a_gpu = cp.array(self.Psi_T_a)
            self.Psi_T_b_gpu = cp.array(self.Psi_T_b)
        except ImportError:
            pass

    def compute_estimator(self, system, hamiltonian, trial, walkers):
        self.counter += 1
        
        if self.counter <= self.burn_in:
            if self.verbose and self.counter % 10 == 0:
                print(f"    ... Burn-in Block {self.counter}/{self.burn_in}", flush=True)
            return

        nbasis = hamiltonian.nbasis
        nwalkers = walkers.nwalkers
        n_samp = min(self.num_samples, nwalkers)
        
        # 1. Extract Current Walkers
        if hasattr(walkers, 'phia'):
            phi_a = walkers.phia[:n_samp]
            phi_b = walkers.phib[:n_samp]
        else:
            phi_a = walkers.phi[:n_samp, :, :self.nalpha]
            phi_b = walkers.phi[:n_samp, :, self.nalpha:]

        # =====================================================================
        # EXACT 1-RDM CONSTRUCTION (Bypassing internal ipie buffers)
        # =====================================================================
        if self.use_gpu and isinstance(walkers.weight, self.cp.ndarray):
            if not isinstance(phi_a, self.cp.ndarray): phi_a = self.cp.array(phi_a)
            if not isinstance(phi_b, self.cp.ndarray): phi_b = self.cp.array(phi_b)
            xp = self.cp
            Psi_a, Psi_b = self.Psi_T_a_gpu, self.Psi_T_b_gpu
            C_sim = self.C_a_gpu # ipie uses Alpha MOs as the simulation basis for BOTH spins
        else:
            if hasattr(phi_a, 'get'): phi_a = phi_a.get()
            if hasattr(phi_b, 'get'): phi_b = phi_b.get()
            xp = np
            Psi_a, Psi_b = self.Psi_T_a, self.Psi_T_b
            C_sim = self.C_a # ipie uses Alpha MOs as the simulation basis for BOTH spins

        # Overlap matrices: O = Psi_T^H @ phi
        O_a = xp.einsum('ui, wuj -> wij', Psi_a.conj(), phi_a)
        O_b = xp.einsum('ui, wuj -> wij', Psi_b.conj(), phi_b)

        invO_a = xp.linalg.inv(O_a)
        invO_b = xp.linalg.inv(O_b)

        # Build Half-rotated Green's function: right = invO @ Psi_T^H
        right_a = xp.einsum('wij, ju -> wiu', invO_a, Psi_a.conj().T)
        right_b = xp.einsum('wij, ju -> wiu', invO_b, Psi_b.conj().T)

        # Full Density in Simulation Basis: G_MO = phi @ right
        G_mo_a = xp.einsum('wvi, wiu -> wvu', phi_a, right_a)
        G_mo_b = xp.einsum('wvi, wiu -> wvu', phi_b, right_b)

        # Transform MO -> AO using the universal simulation basis (C_sim)
        tmp_a = xp.einsum("wvu, ku -> wvk", G_mo_a, C_sim.conj())
        tmp_b = xp.einsum("wvu, ku -> wvk", G_mo_b, C_sim.conj())
        
        P_ao_a = xp.einsum("qv, wvk -> wqk", C_sim, tmp_a)
        P_ao_b = xp.einsum("qv, wvk -> wqk", C_sim, tmp_b)
        
        if self.use_gpu:
            P_ao_a = self.cp.asnumpy(P_ao_a)
            P_ao_b = self.cp.asnumpy(P_ao_b)
            weights = self.cp.asnumpy(walkers.weight[:n_samp])
        else:
            weights = walkers.weight[:n_samp]
            if hasattr(weights, 'get'): weights = weights.get()

        # =====================================================================
        # Energy Extraction and Verification
        # =====================================================================
        local_E_full = local_energy_batch(system, hamiltonian, walkers, trial)[:n_samp]
        if hasattr(local_E_full, 'get'): local_E_full = local_E_full.get()
        
        local_E_tot = local_E_full[:, 0]
        local_E_1b  = local_E_full[:, 1]

        N_alpha_batch = np.einsum('wij,ji->w', P_ao_a, self.S_ao).real
        N_beta_batch  = np.einsum('wij,ji->w', P_ao_b, self.S_ao).real
        
        assert np.allclose(N_alpha_batch, self.nalpha, atol=1e-5), f"Rank {self.rank}: Alpha trace mismatch! {N_alpha_batch}"
        assert np.allclose(N_beta_batch, self.nbeta, atol=1e-5), f"Rank {self.rank}: Beta trace mismatch! {N_beta_batch}"

        PSP_a = P_ao_a @ self.S_ao @ P_ao_a
        idem_err = np.max(np.linalg.norm(PSP_a - P_ao_a, axis=(1,2)))
        assert idem_err < 1e-5, f"Rank {self.rank}: Idempotency violation! Max Error = {idem_err:.2e}"

        P_tot = P_ao_a + P_ao_b
        e1b_manual_batch = np.einsum('wij,ji->w', P_tot, self.h1_ao).real + self.ecore
        e1b_err = np.max(np.abs(e1b_manual_batch - local_E_1b.real))
        assert e1b_err < 1e-5, f"Rank {self.rank}: 1-Body Energy mismatch! Max Error = {e1b_err:.2e}"

        # --- BUFFER DATA ---
        self._buffer["GA"].append(P_ao_a)
        self._buffer["GB"].append(P_ao_b)
        self._buffer["E"].append(local_E_tot)
        self._buffer["W"].append(weights)
        
        prod_step = self.counter - self.burn_in
        if prod_step > 0 and prod_step % CHECKPOINT_FREQ == 0:
            self.flush_to_disk(prod_step)

    def flush_to_disk(self, step_idx):
        filename = f"data_checkpoint_h{N_ATOMS}_run{self.run_idx}_rank{self.rank}_step{step_idx}.npz"
        np.savez_compressed(
            filename, 
            GA=np.array(self._buffer["GA"]),
            GB=np.array(self._buffer["GB"]),
            E=np.array(self._buffer["E"]),
            W=np.array(self._buffer["W"])
        )
        if self.verbose:
            print(f"    [Rank {self.rank}] Saved checkpoint: {filename}", flush=True)
        self._buffer = {"GA": [], "GB": [], "E": [], "W": []}

    @property
    def names(self): return ["DensityExtractor"]
    @property
    def shape(self): return self._shape
    @property
    def data(self):  return np.zeros(self._shape, dtype=np.complex128)

# =============================================================================
# 3. SETUP & RUN
# =============================================================================
comm = MPIHandler()
rank = comm.rank
nprocs = comm.size

chk_file = f"scf_h{N_ATOMS}.chk"
ham_file = f"ham_h{N_ATOMS}.h5"
wfn_file = f"wfn_h{N_ATOMS}.h5"

if rank == 0:
    print(f">>> H{N_ATOMS} AFQMC Training Data Generation (Ensemble Mode)")
    print(f"    Ranks: {nprocs} | Walkers Total: {WALKERS}")
    print(f"    Runs: {NUM_RUNS} | Total Samples Target: {NUM_RUNS * TOTAL_PROD_BLOCKS * SAMPLES_PER_BLOCK}")

# Generate Inputs & Extract Matrices (Rank 0 only)
C_a, C_b, h1_ao, S_ao = None, None, None, None
if rank == 0:
    mol = gto.M(atom=[("H", 0.74 * j, 0, 0) for j in range(N_ATOMS)], basis="sto-6g", verbose=0)
    mf = scf.UHF(mol)
    mf.chkfile = chk_file
    
    if not os.path.exists(wfn_file):
        print(f"    Generating PySCF baseline and integral files for H{N_ATOMS}...")
        mf.kernel()
        gen_ipie_input_from_pyscf_chk(chk_file, hamil_file=ham_file, wfn_file=wfn_file, verbose=0, chol_cut=1e-5)
    else:
        print(f"    Loading existing PySCF baseline for H{N_ATOMS}...")
        mf.__dict__.update(scf.chkfile.load(chk_file, 'scf'))
    
    C_a   = mf.mo_coeff[0]
    C_b   = mf.mo_coeff[1]
    h1_ao = mf.get_hcore()
    S_ao  = mf.get_ovlp()

# Broadcast matrices to all MPI ranks
C_a   = comm.comm.bcast(C_a, root=0)
C_b   = comm.comm.bcast(C_b, root=0)
h1_ao = comm.comm.bcast(h1_ao, root=0)
S_ao  = comm.comm.bcast(S_ao, root=0)

comm.comm.Barrier()

# --- ENSEMBLE LOOP ---
for run_idx in range(NUM_RUNS):
    if rank == 0:
        print(f"\n" + "="*50)
        print(f">>> Starting Independent Run {run_idx + 1}/{NUM_RUNS}")
        print("="*50)

    afqmc = AFQMC.build_from_hdf5(
        num_elec=(N_ATOMS//2, N_ATOMS//2),
        ham_file=ham_file,
        wfn_file=wfn_file,
        num_blocks=TOTAL_RUN_LENGTH,
        num_steps_per_block=STEPS_PER_BLOCK,
        num_walkers=WALKERS,
        seed=42 + run_idx,  
        verbose=0
    )

    afqmc.mpi_handler = comm
    if nprocs > 1:
        local_walkers = WALKERS // nprocs
        if rank < (WALKERS % nprocs): local_walkers += 1
        afqmc.nwalkers = local_walkers

    extractor = DensityExtractor(
        afqmc.hamiltonian, 
        afqmc.trial, 
        C_a=C_a, C_b=C_b, 
        h1_ao=h1_ao, S_ao=S_ao,
        rank=rank, 
        run_idx=run_idx,  
        burn_in=BURN_IN_BLOCKS, 
        num_samples=SAMPLES_PER_BLOCK,
        verbose=(rank == 0)
    )

    afqmc.run(additional_estimators={"Density": extractor})

    if len(extractor._buffer["E"]) > 0:
        extractor.flush_to_disk("FINAL")
        
    comm.comm.Barrier()

if rank == 0:
    print("\n>>> ALL ENSEMBLE SIMULATIONS COMPLETE.")