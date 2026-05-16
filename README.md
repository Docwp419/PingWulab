# TITLE

Branched evolution and macroevolution-like transitions shape head and neck squamous cell carcinoma across space and stage.

# SHORT TITLE

Branched and macroevolution-like evolution in HNSCC.

# ABSTRACT

Head and neck squamous cell carcinoma (HNSCC) remains evolutionarily unresolved, both within established tumors and across the transition from high-risk (HR) mucosa to malignancy.Here, we integrated deep whole-exome sequencing of HR mucosa, tumor adjacent tissue (TAT), and multiregion tumor samples with TRONCO-derived clonal topologies, topology-constrained Bayesian MCMC inference, and monotonic XGBoost classification to define evolutionary mode. Established HNSCC was predominantly branched, with only a minority of cases showing near-neutral or linear patterns. Subclone mixing xenografts showed preferential outgrowth of dominant lineages in vivo, supporting active competition within a branching architecture. Downsampling of a densely sampled multiregion tumor demonstrated that sparse sampling underestimates subclonal diversity and destabilizes mode assignment, whereas intermediate regional coverage approached phylogenetic stability in this case. Spatially explicit three-dimensional simulations further supported branched evolution as the dominant outcome under realistic growth conditions. Across the HR mucosa–TAT–tumor continuum, breakpoint modeling revealed an early mutational transition followed by later copy number dominated remodeling, supporting a non-linear, macroevolution-like trajectory of HNSCC development.

## The repository contains two main computational modules:

1. **Evolution-mode classification module**, which uses topology-derived features, phylogeny-informed heterogeneity metrics, and a monotonic XGBoost classifier to classify tumor evolutionary patterns.
2. **3D deme-based subclone simulation module**, which models mutation-driven subclone generation, clonal growth, topology formation, subclone abundance, and spatial distribution in a three-dimensional tumor space.

# USAGE INSTRUCTIONS

## Repository structure

The repository is organized into two main folders:

```text
.
├── folder1_evolution_mode_classifier
│   ├── evolution_mode_training_set.xlsx
│   ├── sample_subclone_topology_and_relative_abundance_normalized.xlsx
│   └── monotonic_xgboost_evolution_mode_classifier.py
│
└── folder2_deme3d_subclone_simulation
    └── deme3d_subclone_simulation.py
```

---

## 1. folder1_evolution_mode_classifier

### Purpose

The folder `folder1_evolution_mode_classifier` contains the data files and Python script used for tumor evolutionary-mode classification.

The script:

```text
monotonic_xgboost_evolution_mode_classifier.py
```

implements a monotonic XGBoost-based classifier that integrates subclonal topology, relative subclone abundance, within-layer imbalance metrics, and phylogeny-informed heterogeneity features to classify HNSCC evolutionary modes.

The main classification outputs include:

- `Linear`
- `Near-meutral`
- `Branched`

The model first applies a rule-based linear-tree check. Non-linear cases are then classified using a monotonic XGBoost binary classifier for `Near-meutral` versus `Branched`.

### Input files

This folder should contain the following input files:

```text
evolution_mode_training_set.xlsx
sample_subclone_topology_and_relative_abundance_normalized.xlsx
```

#### `evolution_mode_training_set.xlsx`

This file is used as the training dataset. It should contain subclone topology and relative abundance information for samples with known evolutionary labels.

Recommended columns include:

```text
sample
clone
prop
label
```

where:

- `sample` indicates the sample or case ID.
- `clone` indicates the subclone/node ID, such as `G0`, `G0.1`, `G0.1.1`.
- `prop` indicates the relative abundance or proportion of each subclone.
- `label` indicates the known evolutionary pattern, such as `Neutral`, `Branched`, or `Linear`.

#### `sample_subclone_topology_and_relative_abundance_normalized.xlsx`

This file is used for model prediction. It should contain sample-level subclone topology and normalized relative abundance information.

Recommended columns include:

```text
sample
clone
prop
```

If a label column is present, it will not be required for prediction.

### Main output

After running the classifier script, the main output file will be generated as:

```text
HNSCC_evolution_mode_predictions.xlsx
```

This output workbook contains:

- Predicted evolutionary mode for each sample
- Neutral probability estimated by the classifier
- Decision threshold used for classification
- Feature values used for prediction
- Model metadata

### Running environment

Recommended environment:

```text
Python >= 3.8
```

Required Python packages:

```text
numpy
pandas
scikit-learn
xgboost
joblib
openpyxl
```

### Installation

Install the required Python packages using:

```bash
pip install numpy pandas scikit-learn xgboost joblib openpyxl
```

If using Anaconda, the following commands may also be used:

```bash
conda create -n hnscc_evolution python=3.8
conda activate hnscc_evolution
pip install numpy pandas scikit-learn xgboost joblib openpyxl
```

### How to run

Enter the folder:

```bash
cd folder1_evolution_mode_classifier
```

Run the script:

```bash
python monotonic_xgboost_evolution_mode_classifier.py
```

Before running, make sure that the following files are present in the same working directory or that the paths inside the script have been correctly configured:

```text
evolution_mode_training_set.xlsx
sample_subclone_topology_and_relative_abundance_normalized.xlsx
```

### Notes

The classifier uses monotonic constraints to encode biological assumptions about evolutionary imbalance. Features reflecting stronger dominance, imbalance, or edge-concentrated heterogeneity are constrained to decrease the probability of a neutral pattern, whereas scale-corrected global evenness features are constrained to increase the probability of a neutral pattern.

---

## 2. folder2_deme3d_subclone_simulation

### Purpose

The folder `folder2_deme3d_subclone_simulation` contains the 3D spatial subclone simulation script:

```text
deme3d_subclone_simulation.py
```

This script implements a spatially explicit deme-based tumor evolution model to simulate HNSCC subclone generation, clonal expansion, subclonal topology, abundance dynamics, and spatial distribution.

The simulation models tumor growth on a three-dimensional lattice of demes. Each deme has a finite carrying capacity and interacts with its local neighboring demes. During simulation, cells may undergo:

- Birth
- Death
- Driver mutation
- Passenger mutation
- Stochastic branching
- Local competition
- Migration
- Environmental carrying-capacity limitation
- Transient CIN-like mutational burst

### Main outputs

For each simulation run, the script can generate:

```text
filtered_clone_counts_t*.xlsx
clone_growth_curves_filtered_t*.pdf
phylogenetic_tree_filtered_t*.pdf
z_plane_t*_*.pdf
```

These outputs include:

- Filtered subclone abundance tables
- Subclone proportions
- Driver-gene information
- Diversity metrics such as PHI, Shannon index, and Simpson index
- Subclone growth curves
- Phylogenetic topology trees
- Spatial Z-plane visualizations of subclone distribution

### Running environment

Recommended environment:

```text
Python >= 3.8
```

Required Python packages:

```text
numpy
pandas
matplotlib
networkx
openpyxl
```

Optional Python package:

```text
numba
```

Optional external tools:

```text
Graphviz
```

`numba` can accelerate numerical operations but is not required. If `numba` is not installed, the script will run using a fallback implementation.

`Graphviz` is optional and may improve phylogenetic tree layout quality when available. If unavailable, the script will use an internal fallback tree layout.

### Installation

Install the required Python packages using:

```bash
pip install numpy pandas matplotlib networkx openpyxl
```

Optional acceleration and layout-related packages can be installed using:

```bash
pip install numba pydot pygraphviz
```

If using Anaconda, the following commands may be used:

```bash
conda create -n hnscc_deme3d python=3.8
conda activate hnscc_deme3d
pip install numpy pandas matplotlib networkx openpyxl numba pydot
```

Graphviz may need to be installed separately depending on the operating system.

### How to run

Enter the folder:

```bash
cd folder2_deme3d_subclone_simulation
```

Run the script:

```bash
python deme3d_subclone_simulation.py
```

Before running, users may edit the following parameters inside the script:

```python
RUNS = 1
SEED_BASE = None
t_max = 250.0
C_deme = int(10**3)
K_env_total_cells = int(1e7)
p_d = 1e-5
```

To make the simulation reproducible, set a fixed random seed:

```python
SEED_BASE = 12345
```

If `SEED_BASE = None`, simulations will remain stochastic.


### Notes

This simulation is intended as a mechanistic computational framework for exploring spatially constrained, mutation-driven, branched subclonal evolution in HNSCC. It is not intended to reconstruct the exact evolutionary history of any individual tumor. Instead, it provides a controlled model system for examining how mutation rate, driver acquisition, local competition, spatial migration, environmental constraints, and transient CIN-like acceleration may shape tumor evolutionary architecture.

---

## General reproducibility notes

All scripts rely on stochastic or machine-learning procedures. To support reproducibility:

- Keep the input Excel files unchanged when reproducing reported results.
- Use fixed random seeds where possible.
- Record package versions for formal reproducibility.
- Run scripts from the corresponding folder to avoid path-related errors.
- Ensure that input file names match those specified in the script or modify the paths accordingly.

Suggested package-version recording command:

```bash
pip freeze > requirements.txt
```

A `requirements.txt` file may then be included in the repository to facilitate environment reconstruction.
