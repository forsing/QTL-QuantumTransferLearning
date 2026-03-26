# Quantum Transfer Learning (QTL) for Lottery Prediction
# Lottery prediction generated using a hybrid model with a classical feature transfer layer and a quantum head.
# Quantum Regression Model with Qiskit


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize

from qiskit_machine_learning.utils import algorithm_globals
import random

# ================= SEED PARAMETERS =================
SEED = 39
random.seed(SEED)
np.random.seed(SEED)
algorithm_globals.random_seed = SEED
# ==================================================


# Use the existing dataframe
df_raw = pd.read_csv('/Users/4c/Desktop/GHQ/data/loto7hh_4586_k24.csv')
# 4548 historical draws of Lotto 7/39 (Serbia)

_MIN_POS = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
_MAX_POS = np.array([33, 34, 35, 36, 37, 38, 39], dtype=int)


def quantum_transfer_learning_predict(df):
    df = df.copy()
    cols = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6', 'Num7']
    predictions = {}
    
    # Model Hyperparameters
    num_qubits = 1
    train_window = 20
    
    # QTL Logic: We use a "Pre-trained" (fixed) classical layer to transform features
    # before they enter the Quantum Circuit. This simulates transferring knowledge 
    # from a classical domain (feature extraction) to a quantum domain.
    
    # Define Quantum Circuit (The Quantum "Head")
    x_param = ParameterVector('x', 1)
    theta_param = ParameterVector('theta', 2)
    qc = QuantumCircuit(num_qubits)
    qc.ry(x_param[0], 0) # Encoding transferred features
    qc.rz(theta_param[0], 0) # Trainable weight
    qc.ry(theta_param[1], 0) # Trainable weight
    
    observable = SparsePauliOp('Z')
    estimator = StatevectorEstimator()
    
    for idx, col in enumerate(cols):
        # 1. Feature Engineering: 1 Lag
        df[f'{col}_lag'] = df[col].shift(1)
        df_model = df.dropna().tail(train_window + 1)
        
        X = df_model[[f'{col}_lag']].values
        y = df_model[col].values
        
        # 2. Scaling
        scaler_x = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler_x.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # 3. "Transfer" Layer: Fixed Classical Transformation
        # This simulated pre-trained layer extracts non-linear features from the lags.
        def transfer_layer(x):
            return np.tanh(x * 1.5 + 0.5) 
            
        X_trans = transfer_layer(X_scaled)
        
        # 4. Training the Quantum Head
        def eval_qnode(x_val, params):
            # Bind values: [transferred_x, theta0, theta1]
            param_values = [x_val, params[0], params[1]]
            pub = (qc, observable, param_values)
            job = estimator.run([pub])
            evs = job.result()[0].data.evs
            return float(np.real(np.asarray(evs).reshape(-1)[0]))

        def cost_fn(params):
            mse = 0.0
            for i in range(len(X_trans)-1):
                pred = eval_qnode(X_trans[i][0], params)
                mse += (pred - y_scaled[i])**2
            return mse / (len(X_trans)-1)

        # v2: robustniji fit - više startova i više iteracija
        best_x = None
        best_cost = float("inf")
        for _ in range(4):
            init_params = np.random.uniform(0, 2*np.pi, 2)
            res = minimize(cost_fn, init_params, method='COBYLA', options={'maxiter': 180, 'rhobeg': 0.25})
            c = float(res.fun)
            if c < best_cost:
                best_cost = c
                best_x = res.x
        
        # 5. Prediction for the Next Draw
        x_next_trans = transfer_layer(X_scaled[-1:])
        y_pred_scaled = eval_qnode(x_next_trans[0][0], best_x)
        
        # Inverse scale back to lottery number range
        pred_final = scaler_y.inverse_transform(np.array([[y_pred_scaled]]))
        lo, hi = int(_MIN_POS[idx]), int(_MAX_POS[idx])
        predictions[col] = int(round(np.clip(pred_final[0][0], lo, hi)))
        
    return predictions

print()
print("Computing predictions using Quantum Transfer Learning (QTL) ...")
print()
q_qtl_results = quantum_transfer_learning_predict(df_raw)

# Format for display
q_qtl_df = pd.DataFrame([q_qtl_results])
# q_qtl_df.index = ['Quantum Transfer Learning (QTL) Prediction']

print()
print("Lottery prediction generated using a hybrid model with a classical feature transfer layer and a quantum head.")
print()

print()
print("Quantum Transfer Learning (QTL) Results:")
print(q_qtl_df.to_string(index=True))
print()
"""
Computing predictions using Quantum Transfer Learning (QTL) ...


Lottery prediction generated using a hybrid model with a classical feature transfer layer and a quantum head.


Quantum Transfer Learning (QTL) Results:
   Num1  Num2  Num3  Num4  Num5  Num6  Num7
0     7    x    y    z    26    29    35
"""



"""
Quantum Transfer Learning (QTL).

df.copy() da se ulaz ne menja usput
train_window 15 -> 20
stabilniji quantum head fit: COBYLA sa 4 restarta i 180 iteracija
sigurno čitanje evs kao float
izlaz clip po poziciji (1..33, 2..34, ..., 7..39)

(v2: train_window=20; robustniji COBYLA fit (4 starta, 180 iteracija); clip po poziciji; df.copy()).
"""
