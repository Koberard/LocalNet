import os
import sys
import ctypes
import time
import json
import numpy as np
import joblib

# =============================================================================
# 0. CUDA HACK (Oscar Cluster Specifics)
# =============================================================================
oscar_cuda_path = "/oscar/rt/9.6/25/spack/x86_64_v3/cuda-12.9.0-cinrl2oeqemd3szbcakkugp2vtk2fh5t"
nvvm_lib_dir = os.path.join(oscar_cuda_path, "nvvm", "lib64")
nvrtc_lib_dir = os.path.join(oscar_cuda_path, "targets", "x86_64-linux", "lib")
standard_lib_dir = os.path.join(oscar_cuda_path, "lib64")
os.environ['CUDA_HOME'] = oscar_cuda_path
os.environ['CPATH'] = os.path.join(oscar_cuda_path, 'include')
os.environ['PATH'] = os.path.join(oscar_cuda_path, 'bin') + ":" + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = f"{nvvm_lib_dir}:{nvrtc_lib_dir}:{standard_lib_dir}:/lib64:" + os.environ.get('LD_LIBRARY_PATH', '')
os.environ["IPIE_USE_GPU"] = "1"

try:
    ctypes.CDLL(os.path.join(nvvm_lib_dir, "libnvvm.so"), mode=ctypes.RTLD_GLOBAL)
except: 
    pass

import tensorflow as tf
from tensorflow.keras import layers, models

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

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus: 
        tf.config.experimental.set_memory_growth(gpu, True)

# =============================================================================
# 1. ACENE CONFIGURATION & ENSEMBLE SETUP
# =============================================================================
ACENE_N = 4 
N_ATOMS = 6 * ACENE_N + 6  
SYSTEM_NAME = f"A_{ACENE_N}" 

WALKERS = 2048
TEST_SEED = 999
TOTAL_PROD_BLOCKS = 200 
BURN_IN_BLOCKS = 0 
TOTAL_RUN_LENGTH = BURN_IN_BLOCKS + TOTAL_PROD_BLOCKS
STEPS_PER_BLOCK = 40

XYZ_DIR = '/oscar/scratch/kberard/DL_Research/Local-Net/Paper_Results/Scaling/A_Scaling_More_Walkers/XYZ/'
XYZ_FILE = os.path.join(XYZ_DIR, f"{SYSTEM_NAME}.xyz")

DEPLOY_DIR = "deployment_objects"
WEIGHTS_PATH = os.path.join(DEPLOY_DIR, f"GNN_{SYSTEM_NAME}_DeltaHF.weights.h5")

# =============================================================================
# 2. MODEL ARCHITECTURE & UTILS
# =============================================================================
@tf.keras.utils.register_keras_serializable()
class DistanceEmbedding(layers.Layer):
    def __init__(self, n_rbf=32, r_min=0.0, r_max=5.0, **kwargs):
        super().__init__(**kwargs)
        self.n_rbf = n_rbf
        self.centers = tf.linspace(r_min, r_max, n_rbf)
        self.gamma = (r_max - r_min) / n_rbf

    def call(self, distances):
        return tf.exp(-(distances[..., None] - self.centers)**2 / self.gamma**2)

@tf.keras.utils.register_keras_serializable()
class BroadcastStatic(layers.Layer):
    def call(self, inputs):
        x_static, x_batch_ref = inputs
        batch_size = tf.shape(x_batch_ref)[0]
        expanded = tf.expand_dims(x_static, 0)
        return tf.tile(expanded, [batch_size, 1, 1, 1])

@tf.keras.utils.register_keras_serializable()
class GraphInteraction(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        edge_shape = input_shape[1] 
        self.update_mlp = models.Sequential([
            layers.Dense(self.units, activation='swish', kernel_initializer='he_normal'),
            layers.Dense(self.units, activation='swish', kernel_initializer='he_normal')
        ])
        self.update_mlp.build(edge_shape)
        super().build(input_shape)

    def call(self, inputs):
        node_feats, edge_feats, adjacency = inputs
        messages = self.update_mlp(edge_feats)
        mask = tf.expand_dims(adjacency, axis=0)       
        mask = tf.expand_dims(mask, axis=-1)            
        messages = messages * mask
        aggr_messages = tf.reduce_mean(messages, axis=2)
        return node_feats + aggr_messages

@tf.keras.utils.register_keras_serializable()
class SumPooling(layers.Layer):
    def call(self, x): 
        return tf.reduce_sum(x, axis=1)

def build_gnn_model(dist_matrix, adj_matrix):
    input_nodes = layers.Input(shape=(N_ATOMS, 1), name="Node_Density")
    input_edges_dyn = layers.Input(shape=(N_ATOMS, N_ATOMS, 2), name="Edge_Density_Matrix")

    static_dist = tf.constant(dist_matrix, dtype=tf.float32) 
    static_adj  = tf.constant(adj_matrix, dtype=tf.float32)  

    rbf_layer = DistanceEmbedding(n_rbf=32)
    dist_embedding = rbf_layer(static_dist) 

    dist_feats = BroadcastStatic()([dist_embedding, input_nodes])
    combined_edges = layers.Concatenate()([input_edges_dyn, dist_feats])

    x = layers.Dense(64, activation='swish')(input_nodes)
    
    for _ in range(2): 
        x = GraphInteraction(64)([x, combined_edges, static_adj])
        x = layers.LayerNormalization()(x)
    
    x = layers.Dense(32, activation='swish')(x)
    atomic_energies = layers.Dense(1, name="Atomic_Energy_Pred")(x) 
    total_energy = SumPooling(name="Sum_Pooling")(atomic_energies)

    return models.Model(inputs=[input_nodes, input_edges_dyn], outputs=total_energy)

# =============================================================================
# 3. PHYSICS & PROXY PATCH
# =============================================================================
def get_dynamic_operators(mol):
    S = mol.intor('int1e_ovlp')
    h_core_ao = mol.intor('int1e_nuc') + mol.intor('int1e_kin')
    e, v = np.linalg.eigh(S)
    mask = e > 1e-15
    S_inv_sqrt = v[:, mask] @ np.diag(e[mask]**(-0.5)) @ v[:, mask].T
    S_sqrt = v[:, mask] @ np.diag(e[mask]**(0.5)) @ v[:, mask].T
    h_core_lowdin = S_inv_sqrt.T @ h_core_ao @ S_inv_sqrt
    return S_inv_sqrt, S_sqrt, h_core_lowdin, S

def create_ml_local_energy_patch(ml_model, y_scaler, P_hf_ref, E_hf_ref, S_sqrt, h_core_dyn, C_a, C_b, aoslices, use_gpu, n_atoms):
    xp = cp if use_gpu else np
    P_hf_ref_xp = xp.asarray(P_hf_ref)
    S_sqrt_xp = xp.asarray(S_sqrt)
    h_core_dyn_xp = xp.asarray(h_core_dyn)
    C_a_xp, C_b_xp = xp.asarray(C_a), xp.asarray(C_b)

    @tf.function(reduce_retracing=True)
    def fast_predict(inputs): 
        return ml_model(inputs, training=False)

    tracker = {"total_time_sec": 0.0, "calls": 0}

    def local_energy_single_det_uhf(system, hamiltonian, walkers, trial):
        if use_gpu: cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()

        nwalkers = walkers.nwalkers
        nalpha = trial.nalpha
        phi_a = walkers.phia if hasattr(walkers, 'phia') else walkers.phi[:, :, :nalpha]
        phi_b = walkers.phib if hasattr(walkers, 'phib') else walkers.phi[:, :, nalpha:]
        
        Psi_T_a, Psi_T_b = xp.asarray(trial.psi[:, :nalpha]), xp.asarray(trial.psi[:, nalpha:])
        phi_a, phi_b = xp.asarray(phi_a), xp.asarray(phi_b)

        O_a = xp.einsum('ui, wuj -> wij', Psi_T_a.conj(), phi_a)
        O_b = xp.einsum('ui, wuj -> wij', Psi_T_b.conj(), phi_b)
        invO_a, invO_b = xp.linalg.inv(O_a), xp.linalg.inv(O_b)
        
        G_mo_a = xp.einsum('wvi, wiu -> wvu', phi_a, xp.einsum('wij, ju -> wiu', invO_a, Psi_T_a.conj().T))
        G_mo_b = xp.einsum('wvi, wiu -> wvu', phi_b, xp.einsum('wij, ju -> wiu', invO_b, Psi_T_b.conj().T))
        
        # P_ao generation matching UHF
        P_ao = (xp.einsum("qi, wij, pj -> wqp", C_a_xp, G_mo_a, C_a_xp.conj()) + 
                xp.einsum("qi, wij, pj -> wqp", C_b_xp, G_mo_b, C_b_xp.conj()))

        P_lowdin = xp.einsum('ai, wib, bj -> waj', S_sqrt_xp, P_ao, S_sqrt_xp)
        
        delta_P = cp.asnumpy(P_lowdin - P_hf_ref_xp) if use_gpu else P_lowdin - P_hf_ref_xp
        
        # Acene specific: Contract full basis set down to atomic nodes/edges
        delta_P_atom = np.zeros((nwalkers, n_atoms, n_atoms), dtype=np.complex128)
        for i in range(n_atoms):
            p0, p1 = aoslices[i][2], aoslices[i][3]
            for j in range(n_atoms):
                q0, q1 = aoslices[j][2], aoslices[j][3]
                delta_P_atom[:, i, j] = np.sum(delta_P[:, p0:p1, q0:q1], axis=(1, 2))
        
        X_nodes = np.real(np.diagonal(delta_P_atom, axis1=1, axis2=2))
        X_nodes = X_nodes.reshape(nwalkers, n_atoms, 1).astype(np.float32)
        X_edges = np.stack([np.real(delta_P_atom), np.imag(delta_P_atom)], axis=-1).astype(np.float32)
        
        preds_scaled = fast_predict([X_nodes, X_edges]).numpy()
        E_corr_delta = y_scaler.inverse_transform(preds_scaled).flatten()

        E_1B_delta = cp.asnumpy(xp.einsum('ij, wji -> w', h_core_dyn_xp, P_lowdin - P_hf_ref_xp).real) if use_gpu else xp.einsum('ij, wji -> w', h_core_dyn_xp, P_lowdin - P_hf_ref_xp).real

        energy_out = xp.zeros((nwalkers, 3), dtype=xp.complex128)
        energy_out[:, 0] = E_hf_ref + xp.asarray(E_1B_delta) + xp.asarray(E_corr_delta)
        energy_out[:, 1] = E_hf_ref + xp.asarray(E_1B_delta)
        energy_out[:, 2] = xp.asarray(E_corr_delta)
        print("here")
        
        if use_gpu: cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        tracker["total_time_sec"] += (t1 - t0)
        tracker["calls"] += 1

        is_gpu_walker = hasattr(walkers.weight, 'device') or 'cupy' in str(type(walkers.weight))
        return cp.asarray(energy_out) if (is_gpu_walker and use_gpu) else cp.asnumpy(energy_out) if use_gpu else energy_out

    return local_energy_single_det_uhf, tracker

# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================
comm = MPIHandler()
rank = comm.rank
if has_cupy and cp.cuda.runtime.getDeviceCount() > 0: 
    cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()

chk_file = f"scf_{SYSTEM_NAME}.chk"
ham_file = f"ham_{SYSTEM_NAME}.h5"
wfn_file = f"wfn_{SYSTEM_NAME}.h5"

if rank == 0:
    print(f">>> Rank 0: Initializing Physics for {SYSTEM_NAME} and Loading Weights...")
    mol = gto.M(atom=XYZ_FILE, basis="sto-6g", verbose=0, spin=0)
    mf = scf.UHF(mol)
    mf.chkfile = chk_file
    
    if not os.path.exists(chk_file) or not os.path.exists(wfn_file):
        mf.kernel()
        gen_ipie_input_from_pyscf_chk(chk_file, hamil_file=ham_file, wfn_file=wfn_file, verbose=0, chol_cut=1e-5)
    else:
        mf.kernel()
    
    E_HF = mf.e_tot
    C_a = mf.mo_coeff[0] if np.ndim(mf.mo_coeff) == 3 else mf.mo_coeff
    C_b = mf.mo_coeff[1] if np.ndim(mf.mo_coeff) == 3 else mf.mo_coeff
    aoslices = mol.aoslice_by_atom()
    
    _, S_sqrt, h_core, _ = get_dynamic_operators(mol)
    P_hf_ref = S_sqrt @ (mf.make_rdm1()[0] + mf.make_rdm1()[1]) @ S_sqrt  
    
    dist_matrix = np.load(os.path.join(DEPLOY_DIR, f"dist_matrix_{SYSTEM_NAME}.npy"))
    adj_matrix = np.load(os.path.join(DEPLOY_DIR, f"adj_matrix_{SYSTEM_NAME}.npy"))
    y_scaler = joblib.load(os.path.join(DEPLOY_DIR, f"y_scaler_{SYSTEM_NAME}.save"))

    print(">>> Rebuilding GNN Architecture...")
    ml_model = build_gnn_model(dist_matrix, adj_matrix)
    ml_model.load_weights(WEIGHTS_PATH)
    
    dummy_nodes = np.zeros((1, N_ATOMS, 1), dtype=np.float32)
    dummy_edges = np.zeros((1, N_ATOMS, N_ATOMS, 2), dtype=np.float32)
    ml_model([dummy_nodes, dummy_edges])
else: 
    E_HF = C_a = C_b = S_sqrt = h_core = ml_model = y_scaler = P_hf_ref = aoslices = mol = mf = None

E_HF = comm.comm.bcast(E_HF, root=0)
C_a = comm.comm.bcast(C_a, root=0)
C_b = comm.comm.bcast(C_b, root=0)
S_sqrt = comm.comm.bcast(S_sqrt, root=0)
h_core = comm.comm.bcast(h_core, root=0)
aoslices = comm.comm.bcast(aoslices, root=0)
mol_nelectron = comm.comm.bcast(mol.nelectron if rank == 0 else None, root=0)

afqmc = AFQMC.build_from_hdf5(
    num_elec=(mol_nelectron // 2, mol_nelectron // 2), 
    ham_file=ham_file, 
    wfn_file=wfn_file, 
    num_walkers=WALKERS, 
    num_blocks=TOTAL_RUN_LENGTH, 
    num_steps_per_block=STEPS_PER_BLOCK, 
    verbose=0, 
    seed=TEST_SEED
)

if has_cupy: 
    afqmc.cuda = True
afqmc.mpi_handler = comm

local_walkers = WALKERS // comm.size
if rank < (WALKERS % comm.size): 
    local_walkers += 1
afqmc.nwalkers = local_walkers

ml_proxy, loop_tracker = create_ml_local_energy_patch(ml_model, y_scaler, P_hf_ref, E_HF, S_sqrt, h_core, C_a, C_b, aoslices, has_cupy, N_ATOMS)

# --- ROBUST MONKEYPATCHING ---
targets = [getattr(ipie.estimators.local_energy_sd, f) for f in ["local_energy_single_det_uhf", "local_energy_single_det_batch_gpu", "local_energy_single_det_uhf_batch_gpu", "local_energy_single_det_uhf_batch"] if hasattr(ipie.estimators.local_energy_sd, f)]

for mod_name, module in list(sys.modules.items()):
    if module and mod_name.startswith("ipie"):
        try:
            for attr_name, attr_value in vars(module).items():
                if attr_value in targets: 
                    setattr(module, attr_name, ml_proxy)
        except: 
            pass
            
if hasattr(afqmc, 'propagator'): afqmc.propagator.local_energy = ml_proxy
if hasattr(afqmc, 'estimators'):
    try: afqmc.estimators['energy'].local_energy = ml_proxy
    except: pass

if rank == 0: 
    print("\n" + "#"*60 + "\n### STARTING IPIE GNN-SURROGATE PRODUCTION RUN ###\n" + "#"*60)

afqmc.run()

# =============================================================================
# 5. ANALYSIS & METRICS
# =============================================================================
if rank == 0:
    # 1. Physics Extraction
    est_file = afqmc.estimators.filename if hasattr(afqmc.estimators, 'filename') else "estimates.0.h5"
    qmc_data = extract_observable(est_file, "energy")
    
    # Extract raw array and drop burn-in for correlated error analysis
    raw_etotal_array = np.array(qmc_data["ETotal"])[BURN_IN_BLOCKS:]
    
    df_ac = reblock_by_autocorr(raw_etotal_array, verbose=0)
    final_energy = float(df_ac["ETotal_ac"].iloc[0])
    final_error = float(df_ac["ETotal_error_ac"].iloc[0])

    # 2. Timing
    avg_proxy_time = loop_tracker["total_time_sec"] / loop_tracker["calls"] if loop_tracker["calls"] > 0 else 0

    # 3. Intrinsic Data Storage (Model Weights + Graph Batch)
    model_params = ml_model.count_params()
    model_weights_mb = (model_params * 4) / (1024 ** 2)  # 4 bytes for float32
    
    # Calculate the size of the nodes and edges for the entire batch of walkers
    graph_nodes_bytes = afqmc.nwalkers * N_ATOMS * 1 * 4
    graph_edges_bytes = afqmc.nwalkers * N_ATOMS * N_ATOMS * 2 * 4
    graph_batch_mb = (graph_nodes_bytes + graph_edges_bytes) / (1024 ** 2)
    
    total_gnn_mb = model_weights_mb + graph_batch_mb

    metrics = {
        "N_atoms": N_ATOMS,
        "backend": "GNN",
        "results": {
            "final_energy_ha": round(final_energy, 6),
            "final_error_ha": round(final_error, 6),
            "raw_block_energies": raw_etotal_array.tolist()
        },
        "local_energy_proxy": {
            "avg_time_sec": round(avg_proxy_time, 6),
            "model_weights_mb": round(model_weights_mb, 4),
            "graph_batch_mb": round(graph_batch_mb, 4),
            "total_intrinsic_memory_mb": round(total_gnn_mb, 4)
        }
    }
    
    with open(f"scaling_metrics_GNN_{SYSTEM_NAME}.json", "w") as f: 
        json.dump(metrics, f, indent=4)
        
    print(f"\n{'='*50}")
    print("GNN METRICS SUMMARY")
    print(f"{'='*50}")
    print_metrics = metrics.copy()
    print_metrics["results"].pop("raw_block_energies")
    print(json.dumps(print_metrics, indent=4))