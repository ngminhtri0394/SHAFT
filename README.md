# SHAFT: Symmetry-aware Hierarchical Architecture for Flow-based Traversal

We propose the Symmetry-aware Hierarchical Architecture for Flow-based Traversal (SHAFT), a novel generative model employing a hierarchical exploration strategy to efficiently exploit the symmetry of the materials space to generate crystal structures given desired properties.

---

## Installation

To get started with SHAFT, you'll need to create a conda environment with all the necessary dependencies.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ngminhtri0394/SHAFT.git
    cd SHAFT
    ```

2.  **Create and activate the conda environment:**
    The `environment.yml` file contains all the required packages. Create the environment and activate it using the following commands:
    ```bash
    conda env create -f environment.yml
    conda activate SHAFT
    pip install alignn==2023.5.3
    pip install torch_geometric
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
    conda install dglteam::dgl-cuda11.7
    ```
    This will install all the necessary dependencies for the project.

---

## Usage

The main script for training the SHAFT model is `train_SHAFT.py`. This script uses Hydra for configuration management, making it easy to customize the training process.


### Training the Model

To start training the model, run the following command from the root directory of the project:

```bash
python train_SHAFT.py
```
The sampled crytal structure will be saved at `hydra/singlerun/date/SHAFT_date/saved_data/`, the policy model are saved at `hydra/singlerun/date/SHAFT_date/saved_data/`

