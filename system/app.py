from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import torch
import json
import uuid

# Add the system directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import backend modules
from peptidegen.inference.sampler import PeptideSampler
from peptidegen.evaluation.metrics import calculate_validity, calculate_uniqueness, calculate_novelty
from peptidegen.evaluation.stability import calculate_instability_index, is_stable

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Global variables to hold the loaded model
sampler = None
CHECKPOINT_PATH = os.path.join("checkpoints", "best_model.pt")

def load_model():
    global sampler
    if os.path.exists(CHECKPOINT_PATH):
        try:
            sampler = PeptideSampler.from_checkpoint(CHECKPOINT_PATH)
            print(f"Model loaded from {CHECKPOINT_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            sampler = None
    else:
        print(f"Checkpoint not found at {CHECKPOINT_PATH}. Using mock generator.")
        sampler = None

# Initialize model on startup
load_model()

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "status": "running",
        "model_loaded": sampler is not None,
        "checkpoint_path": CHECKPOINT_PATH
    })

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json
    num_sequences = data.get('num_sequences', 10)
    min_length = data.get('min_length', 10) # Not all generators support this dynamic length constraint easily
    stability_filter = data.get('stability_filter', False)
    temperature = data.get('temperature', 1.0)

    if sampler:
        try:
            # Generate sequences using the real model
            sequences = sampler.sample(n=num_sequences, temperature=temperature)

            # Post-process results
            results = []
            for seq in sequences:
                instability = calculate_instability_index(seq)
                stable = is_stable(instability)

                if stability_filter and not stable:
                    continue

                results.append({
                    "sequence": seq,
                    "stability": round(instability, 2),
                    "is_stable": stable,
                    "validity": "Valid" # Placeholder, real check could be added
                })

            # Save results to a file for persistence
            os.makedirs("results", exist_ok=True)
            output_filename = f"results/generated_{uuid.uuid4().hex[:8]}.json"
            with open(output_filename, 'w') as f:
                json.dump(results, f, indent=2)

            return jsonify({"peptides": results, "saved_to": output_filename})

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        # Fallback to mock generation if model is not loaded
        print("Using mock generation logic.")
        mock_peptides = []
        import random
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"

        for _ in range(num_sequences):
            length = min_length # approximate
            seq = "".join(random.choice(amino_acids) for _ in range(length))
            instability = random.uniform(20, 60) # Random instability index
            stable = instability < 40

            if stability_filter and not stable:
                continue

            mock_peptides.append({
                "sequence": seq,
                "stability": round(instability, 2),
                "is_stable": stable,
                "validity": "Mock"
            })

        return jsonify({"peptides": mock_peptides, "note": "Generated using mock logic (no model loaded)"})

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    sequences = data.get('sequences', [])

    if not sequences:
        return jsonify({"error": "No sequences provided"}), 400

    # Calculate metrics
    validity = calculate_validity(sequences)
    uniqueness = calculate_uniqueness(sequences)
    novelty = calculate_novelty(sequences, training_set=[]) # Requires training set for real calculation

    metrics = {
        "validity": f"{validity * 100:.1f}%",
        "uniqueness": f"{uniqueness * 100:.1f}%",
        "novelty": f"{novelty * 100:.1f}%", # Placeholder
        "diversity_score": 0.85 # Placeholder
    }

    return jsonify(metrics)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
