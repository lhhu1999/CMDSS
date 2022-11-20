# MdDTI
MdDTI: Multi-dimensional drug-target interaction prediction by preserving the consistency of attention distribution.

## Setup and dependencies
Dependencies:
* python 3.6
* pytorch >=1.2
* numpy
* sklearn
* tqdm
* rdkit

## Resources
* RawData:
    - interaction: The five interaction datasets used in the paper. Including the raw data and random shuffling data (random seed: 1234).
    - affinity: The two affinity datasets used in the paper. Including the raw data and random shuffling data (random seed: 1234).
* handle_interaction: Preprocessing the interaction datasets.
    - data_shaffle.py
    - extract_smiles_and_sdf.py
    - extract_drug_adjacency.py
    - extract_atomic_coordinate.py
    - encode.py: Drug and target encoding representation.
    - extract_icmf.py: Decompose drugs by ICMF method.
* handle_dude: Preprocessing the balanced and unbalanced datasets.
    - Reference handle_interaction.
* handle_affinity: Preprocessing the affinity datasets.
    - Reference handle_interaction
* datasets: Input data of the model.
* query_network: Target query network.
    - encoder_residues.py: Encoder module.
    - decoder_2D_skeletons.py: 2D Decoder module.
    - decoder_3D_skeletons.py: 3D Decoder module.
* config.py: Part of the hyperparameters.
* model.py: MdDTI model architecture.
* train_affinity.py: Train and test the model on the Kd and EC50 datasets.
* train_Davis_KIBA.py: Train and test the model on the Davis and KIBA datasets.
* train_interaction.py: Train and test the model on the HUMAN and C.ELEGANS datasets.
* train_unbalanced_datasets.py: Train and test the model on the dude(1:1), dude(1:3) and dude(1:5) datasets.

## Run
