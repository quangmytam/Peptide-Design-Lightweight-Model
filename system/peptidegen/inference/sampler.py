import random

class PeptideSampler:
    @staticmethod
    def from_checkpoint(path):
        return PeptideSampler()

    def sample(self, n=10, temperature=1.0):
        # Mock sampling logic if no real model is present
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        return ["".join(random.choice(amino_acids) for _ in range(10)) for _ in range(n)]

class MockPeptideSampler(PeptideSampler):
    pass

def load_generator(checkpoint_path, device='cpu'):
    return None
