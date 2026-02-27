class PeptideStabilityAnalyzer:
    def analyze(self, sequence):
        return {"stable": True}

def calculate_instability_index(sequence):
    return len(sequence) * 2 + 10 # Mock

def is_stable(instability):
    return instability < 40

def calculate_aliphatic_index(sequence):
    return 10.0

def calculate_gravy(sequence):
    return -0.5

def calculate_isoelectric_point(sequence):
    return 7.0

def calculate_aromaticity(sequence):
    return 0.1

def calculate_molecular_weight(sequence):
    return 1000.0

def calculate_charge_at_pH(sequence, ph):
    return 1.0
