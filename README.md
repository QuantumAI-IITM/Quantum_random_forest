# Quantum_random_forest
A kernel-based quantum random forest for improved classification

Firstly the Quantum Random Forest Algorithm utilizes the structure of Kernel based non linear SVMs, along with the decision tree classifier architecture for classification based tasks. The main Idea of the arcitecture is having a random forest architecture, with every tree, having splitting decision made using Kernel based SVM, where the Kernel is determined ny Quantum Kernel Estimation, and the Nystrom Technique is used to approximate the kernel for larger datasets, to reduce time complexity. 

The link to the paper - https://arxiv.org/abs/2210.02355

So i've tried to implement the paper, and tries to compare the results with some standard Classical machine learning models, and also the Quantum Decision tree model, with standard architecture. 

In the file - breast_cancer_qrf.ipynb, Ive implemented the artitecture as per the paper, along with the preprocessing mentioned as a part of the process. Ive used the breast cancer dataset for the same, and this model yields an accuracy score of - 0.89 

Ive also compared it with some models like - 
1) Random Forest (Classical) which yeilds an accuracy of - 0.96
2) RBF Kernel SVM with nystrom approximation - 0.62
3) Quantum Decision tree - 0.64 
I am also attaching the detailed classification reports of each of the models in the folder results 
