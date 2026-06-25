import numpy as np
import glob
import os
import joblib
import tensorflow as tf
from pyscf import gto, scf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
ACENE_N = 6  # 1 = Benzene, 2 = Naphthalene, 3 = Anthracene, ..., 6 = Hexacene
N_ATOMS = 6 * ACENE_N + 6  
SYSTEM_NAME = f"A_{ACENE_N}" 

# Directories and File Paths
XYZ_DIR = '/oscar/scratch/kberard/DL_Research/Local-Net/Paper_Results/Scaling/A_Scaling_More_Walkers/XYZ/'
XYZ_FILE = os.path.join(XYZ_DIR, f"{SYSTEM_NAME}.xyz")
CHECKPOINT_PATTERN = f"../{SYSTEM_NAME}/Training_Data/data_checkpoint_{SYSTEM_NAME}_run*_rank*_step*.npz"

DEPLOY_DIR = "deployment_objects"
os.makedirs(DEPLOY_DIR, exist_ok=True)
MODEL_WEIGHTS_PATH = os.path.join(DEPLOY_DIR, f"GNN_{SYSTEM_NAME}_DeltaHF.weights.h5")

print(f">>> STARTING GRAPH DELTA-LEARNING FOR: {SYSTEM_NAME} ({N_ATOMS} atoms)")

# =============================================================================
# 1. PHYSICS & GEOMETRY SETUP
# =============================================================================
print(f"\n>>> 1. Loading Geometry & Graph Topology from XYZ...")

if not os.path.exists(XYZ_FILE):
    raise FileNotFoundError(f"XYZ file missing: {XYZ_FILE}")

# Initialize PySCF Molecule directly from the exact simulation geometry
mol = gto.M(
    atom=XYZ_FILE, 
    basis="sto-6g", 
    verbose=0, 
    spin=0 # Explicitly defining closed-shell singlet
)

mf = scf.UHF(mol)
mf.kernel()

# --- A. Static Graph Features (Distances & Adjacency) ---
coords = mol.atom_coords() # Note: PySCF stores these internally as Bohr
diff = coords[:, None, :] - coords[None, :, :]
dist_matrix = np.linalg.norm(diff, axis=-1)

# Aromatic systems need a slightly wider cutoff to capture the hexagonal graph
# 3.0 Angstroms (~5.6 Bohr) captures fused nearest neighbors perfectly.
R_CUTOFF = 3.0 / 0.529177 
adj_matrix = (dist_matrix < R_CUTOFF).astype(np.float32)
np.fill_diagonal(adj_matrix, 0.0)

np.save(os.path.join(DEPLOY_DIR, f"dist_matrix_{SYSTEM_NAME}.npy"), dist_matrix)
np.save(os.path.join(DEPLOY_DIR, f"adj_matrix_{SYSTEM_NAME}.npy"), adj_matrix)

# --- B. Standard Operators ---
S = mf.get_ovlp()
eigvals, eigvecs = np.linalg.eigh(S)
S_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

h_core_ao = mf.get_hcore()
h_core_lowdin = S_inv_sqrt @ h_core_ao @ S_inv_sqrt

# Safely sum Alpha and Beta density matrices for aromatics
rdm1_alpha, rdm1_beta = mf.make_rdm1()
P_hf_ao = rdm1_alpha + rdm1_beta
P_hf_lowdin = S_sqrt @ P_hf_ao @ S_sqrt 
E_hf = mf.e_tot

# =============================================================================
# 2. CUSTOM GNN LAYERS
# =============================================================================
@tf.keras.utils.register_keras_serializable()
class DistanceEmbedding(layers.Layer):
    def __init__(self, n_rbf=16, r_min=0.0, r_max=5.0, **kwargs):
        super().__init__(**kwargs)
        self.n_rbf = n_rbf
        self.r_min = r_min
        self.r_max = r_max
        self.centers = tf.linspace(r_min, r_max, n_rbf)
        self.gamma = (r_max - r_min) / n_rbf

    def call(self, distances):
        return tf.exp(-(distances[..., None] - self.centers)**2 / self.gamma**2)
    
    def get_config(self):
        config = super().get_config()
        config.update({"n_rbf": self.n_rbf, "r_min": self.r_min, "r_max": self.r_max})
        return config

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

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

# =============================================================================
# 3. DATA LOADING & PROCESSING (UPDATED FOR ACENES)
# =============================================================================
print(f"\n>>> 3. Loading & Graph Formatting...")
files = sorted(glob.glob(CHECKPOINT_PATTERN))
if not files:
    raise FileNotFoundError(f"No files found matching: {CHECKPOINT_PATTERN}")

nbasis = h_core_ao.shape[0]

def load_checkpoint(f):
    try:
        with np.load(f) as data:
            return data['GA'], data['GB'], data['E']
    except Exception as e:
        print(f"Error loading {f}: {e}")
        return None, None, None

results = joblib.Parallel(n_jobs=-1)(joblib.delayed(load_checkpoint)(f) for f in files)
results = [r for r in results if r[0] is not None]

GA_raw = np.concatenate([r[0] for r in results], axis=0).reshape(-1, nbasis, nbasis)
GB_raw = np.concatenate([r[1] for r in results], axis=0).reshape(-1, nbasis, nbasis)
E_raw  = np.concatenate([r[2] for r in results], axis=0).real.reshape(-1)
del results

print("    Lowdin Transformation...")
P_total = S_sqrt @ (GA_raw + GB_raw) @ S_sqrt
delta_P = P_total - P_hf_lowdin 

# 1. COMPUTE EXACT PHYSICS TARGETS (Using full nbasis x nbasis matrix)
delta_E = E_raw - E_hf
E_1B_delta = np.einsum('ij, bji -> b', h_core_lowdin, delta_P).real
y_corr = delta_E - E_1B_delta

# 2. CONTRACT DENSITY MATRIX TO ATOMIC GRAPH (36x36 -> 12x12)
print("    Contracting basis functions to atomic graph nodes...")
delta_P_atom = np.zeros((delta_P.shape[0], N_ATOMS, N_ATOMS), dtype=delta_P.dtype)
aoslices = mol.aoslice_by_atom()

for i in range(N_ATOMS):
    p0, p1 = aoslices[i][2], aoslices[i][3]
    for j in range(N_ATOMS):
        q0, q1 = aoslices[j][2], aoslices[j][3]
        # Sum the matrix block for atom i and atom j
        delta_P_atom[:, i, j] = np.sum(delta_P[:, p0:p1, q0:q1], axis=(1, 2))

# 3. EXTRACT GRAPH FEATURES (Now properly sized N_ATOMS x N_ATOMS)
X_nodes_raw = np.diagonal(delta_P_atom, axis1=1, axis2=2)[..., None] 
X_edges_real = np.real(delta_P_atom)[..., None] 
X_edges_imag = np.imag(delta_P_atom)[..., None] 
X_edges_dynamic = np.concatenate([X_edges_real, X_edges_imag], axis=-1)

# Filter Outliers
med = np.median(y_corr)
mad = np.median(np.abs(y_corr - med))
mask = np.abs(y_corr - med) < 5 * mad

X_nodes = X_nodes_raw[mask]
X_edges = X_edges_dynamic[mask]
y_target = y_corr[mask]

print("\n>>> SANITIZING DATA...")
X_nodes = np.real(X_nodes).astype(np.float32)
X_edges = X_edges.astype(np.float32)

nan_mask_nodes = np.isnan(X_nodes).any(axis=(1, 2))
nan_mask_edges = np.isnan(X_edges).any(axis=(1, 2, 3))
nan_mask_targets = np.isnan(y_target)
valid_mask = ~(nan_mask_nodes | nan_mask_edges | nan_mask_targets)

print(f"    Original samples: {len(y_target)}")
print(f"    Removed {np.sum(~valid_mask)} corrupted samples (NaNs).")

X_nodes = X_nodes[valid_mask]
X_edges = X_edges[valid_mask]
y_target = y_target[valid_mask]

y_scaler = StandardScaler()
y_target_scaled = y_scaler.fit_transform(y_target.reshape(-1, 1)).flatten()
joblib.dump(y_scaler, os.path.join(DEPLOY_DIR, f"y_scaler_{SYSTEM_NAME}.save"))

indices = np.arange(len(y_target_scaled))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
print("    Data sanitization complete.")
# =============================================================================
# 4. BUILD GNN MODEL
# =============================================================================
print("\n>>> 4. Building Physics-Informed GNN...")

input_nodes = layers.Input(shape=(N_ATOMS, 1), name="Node_Density")
input_edges_dyn = layers.Input(shape=(N_ATOMS, N_ATOMS, 2), name="Edge_Density_Matrix")

static_dist = tf.constant(dist_matrix, dtype=tf.float32) 
static_adj  = tf.constant(adj_matrix, dtype=tf.float32)  
rbf_layer = DistanceEmbedding(n_rbf=32)
dist_embedding = rbf_layer(static_dist)

@tf.keras.utils.register_keras_serializable()
class BroadcastStatic(layers.Layer):
    def call(self, inputs):
        x_static, x_batch_ref = inputs
        batch_size = tf.shape(x_batch_ref)[0]
        expanded = tf.expand_dims(x_static, 0)
        return tf.tile(expanded, [batch_size, 1, 1, 1])

dist_feats = BroadcastStatic()([dist_embedding, input_nodes])
combined_edges = layers.Concatenate()([input_edges_dyn, dist_feats])

x = layers.Dense(64, activation='swish')(input_nodes)

for _ in range(2): 
    x = GraphInteraction(64)([x, combined_edges, static_adj])
    x = layers.LayerNormalization()(x)
    
x = layers.Dense(32, activation='swish')(x)
atomic_energies = layers.Dense(1, name="Atomic_Energy_Pred")(x) 

@tf.keras.utils.register_keras_serializable()
class SumPooling(layers.Layer):
    def call(self, x): return tf.reduce_sum(x, axis=1)

total_energy = SumPooling(name="Sum_Pooling")(atomic_energies)

model = models.Model(inputs=[input_nodes, input_edges_dyn], outputs=total_energy)

model.compile(
    loss='huber', 
    optimizer=optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
    metrics=['mae']
)

# =============================================================================
# 5. TRAINING & EVALUATION
# =============================================================================
print("\n>>> 5. Training...")
model.fit(
    [X_nodes[train_idx], X_edges[train_idx]], y_target_scaled[train_idx],
    validation_data=([X_nodes[test_idx], X_edges[test_idx]], y_target_scaled[test_idx]),
    epochs=500, batch_size=64, verbose=1,
    callbacks=[
        callbacks.EarlyStopping(patience=30, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
    ]
)

print("\n>>> 6. Final Evaluation...")
preds_scaled = model.predict([X_nodes[test_idx], X_edges[test_idx]]).flatten()
truth_scaled = y_target_scaled[test_idx]

preds_real = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
truth_real = y_scaler.inverse_transform(truth_scaled.reshape(-1, 1)).flatten()

mae_total = np.mean(np.abs(preds_real - truth_real)) * 1000
mae_per_atom = mae_total / N_ATOMS

print(f"    Total MAE:        {mae_total:.2f} mHa")
print(f"    MAE per Atom:     {mae_per_atom:.4f} mHa")

model.save_weights(MODEL_WEIGHTS_PATH)
print(f"    Weights saved to: {MODEL_WEIGHTS_PATH}")