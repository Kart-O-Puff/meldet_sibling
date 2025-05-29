# MelDet: Melodic Similarity Detection


This repository implements **MelDet**—a structurally aware melodic similarity scoring model—alongside baseline approaches (Sum Common with Jaccard-like normalization, Original Tversky, Hungarian, and Cosine Similarity). It provides end-to-end scripts for preprocessing MIDI data, computing similarity scores, and evaluating performance. 


## Dependencies
Python 3.8+
Music21
Numpy
Pandas
Matplotlib
Scipy
Seaborn
Sklearn.metrics
Scikit_posthocs
Ast
Pathlib
Csv
Time
Typing
Platform
Psutil
Cpuinfo


## Usage
1. Data Preprocessing
Convert raw MIDI files into n-gram pitch and rhythm sequences: run data_preprocess_2.py.


2. Compute Similarity Scores
Run each approach on processed data:
meldet_approach.py
hungarian_approach.py
sumcommon_approach.py
tversky_approach.py
cosine_similarity_approach.py
Use the interactive menu to visualize similarity matrices or print tables.
CSV reports are saved in `results/{approach}/similarity_report.csv`.


3. Evaluate Model Performance
Generate evaluation metrics (MSE, AUC-ROC, AUC-PR, F1) and graphs: run evaluation_2.py


4. Reproduce Experiments
Repeat steps 1–3 with different window size and step size values to test performance trade-offs.


## Contact


For questions or contributions, please open an issue or contact the researchers.