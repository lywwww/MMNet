import numpy as np
import os


class Config(object):
    def __init__(self):
        # Trainging
        self.learning_rate = 0.001
        self.weight_decay = 0.001
        self.num_folds = 20

        self.epochs = 100
        self.batch_size = 128
        self.evaluate_every = 50
        self.n_class = 5  # Final output classes

        self.result = ''
        self.model_pth = ''
        
