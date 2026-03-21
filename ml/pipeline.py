"""
ml/pipeline.py
Run the full ML training pipeline.
    python -m ml.pipeline
    # or
    python railpulse/ml/pipeline.py
"""
from ml._pipeline import main as run

if __name__ == "__main__":
    run()
