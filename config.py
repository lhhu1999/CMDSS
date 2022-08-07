class Config:
    def __init__(self):
        self.mode = 'gpu'

        # affinity
        self.lr_affinity = 0.001    # EC50,Kd,IC50: 0.001   Ki: 0.002
        self.decay_interval_affinity = 20   # EC50,Kd: 10   Ki,IC50: 20
        self.epochs_affinity = 100

        # interaction
        self.K_FOLD = 5
        self.lr_interaction = 0.0001      # human,celegans: 0.0001  Davis: 0.0005
        self.decay_interval_interaction = 10     # human,celegans: 10  Davis: 15
        self.epochs_interaction = 50     # human,celegans: 50  Davis: 150

        # common
        self.batch_size = 16
        self.atom_dim = 64
        self.residues_dim = 64
        self.weight_decay = 1e-5
        self.lr_decay = 0.5
        self.max_length_skeleton = 150
        self.max_length_residue = 1000
