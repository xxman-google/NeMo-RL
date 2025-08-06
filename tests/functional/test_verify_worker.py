import os
import ray
from nemo_rl.environments.math_environment import MathEnvConfig, MathEnvironmentMetadata, QAVerifyWorker

def main():
    cfg = MathEnvConfig(
        {
            "verifier_type": "qa",
            "verifier_metadata_key": "ground_truth",
            "grader_model_name": "gpt-4o",
            "grader_api_key": os.getenv("GRADER_API_KEY")
        })
    verifier = QAVerifyWorker.remote(cfg)
    pred_data_batch = [
        {"prompt": "Who was awarded the Oceanography Society's Jerlov Award in 2018?", "response": "Dr. Annick Bricaud Honored with 2018 Jerlov Award"},
        {"prompt": "In which year did Melbourne's Monash Gallery of Art (MGA) rebrand and become the Museum of Australian Photography (MAPh)?", "response": "Monash Gallery of Art Rebranded as Museum of Australian Photography in 2032"}
    ]
    metadata_batch = [
        MathEnvironmentMetadata ({ "ground_truth": "Annick Bricaud"}),
        MathEnvironmentMetadata ({ "ground_truth": "2023"}),
    ]
    
    outputs = ray.get(verifier.verify.remote(pred_data_batch, metadata_batch))

    print("== Eval Result ==")
    for i, (score, gt, pred) in enumerate(outputs):
        print(f"Example {i+1}")
        print(f"  Ground Truth : {gt}")
        print(f"  Prediction   : {pred}")
        print(f"  Score        : {score}")
        print()

if __name__ == "__main__":
    main()