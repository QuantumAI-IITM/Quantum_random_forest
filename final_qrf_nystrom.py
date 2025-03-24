import numpy as np
import pandas as pd
from quantum_random_forest import QuantumRandomForest
from nystrom import Nystrom
from data_construction import load_iris_dataset
from split_function import SplitFunction, SplitCriterion
from pqc import PQC

# Load Iris dataset
train_data, test_data = load_iris_dataset()

# Define Quantum Split Criterion
criterion = SplitCriterion.init_info_gain('clas')

# Initialize PQC for QRF
n_qubits = 4  # Adjust based on dataset dimensionality
num_params = 2 * n_qubits  # Example parameterization
pqc_sample_num = 'exact'  # Exact sampling for PQC evaluations
embed_type = 'as_params_all'  # Feature embedding method
pqc = PQC.init_rand_arch(n_qubits, num_params, pqc_sample_num, embed=embed_type)

# Define Quantum Split Function with Nystrom Approximation
split_fn = SplitFunction(criterion=criterion, split_num=2, pqc=pqc, 
                         pqc_sample_num=pqc_sample_num, embed=embed_type, 
                         branch_var='param_rand', num_rand_meas_q=n_qubits, 
                         nystrom=True, svm_c=5)
split_fn.reset_nystrom_transform()

# Initialize and Train Quantum Random Forest
qrf = QuantumRandomForest(n_qubits=n_qubits, dt_type='qke', 
                          num_trees=5, max_depth=4, split_fn=split_fn)
qrf.train(train_data)

# Evaluate the Model
y_pred = qrf.predict(test_data.X)
accuracy = np.mean(y_pred == test_data.y)
print(f'Quantum Random Forest Accuracy on Iris Dataset: {accuracy * 100:.2f}%')
