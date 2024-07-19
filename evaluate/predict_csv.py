import argparse
import torch
from molscribe import MolScribe
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def postprocess_smiles(input_smiles, scores):
    result = []
    broken_num, low_score = 0, 0
    all_smiles = len(input_smiles)

    for num in range(all_smiles):
        if '.' in input_smiles[num]:
            result.append(None)
            broken_num += 1
        elif "invalid" in input_smiles[num]:
            result.append(None)
        elif scores[num] < 0.88:
            # eliminate low quality images
            low_score += 1
            result.append(None)
        else:
            result.append(input_smiles[num])
    return result, broken_num/all_smiles, low_score/all_smiles

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, required=True)
    parser.add_argument('--image_path', type=str, default=None, required=True)
    parser.add_argument("-n", "--batch_size", default=100, type=int)
    args = parser.parse_args()

    device = torch.device('cuda')
    model = MolScribe(args.model_path, device)
    all_path = pd.read_csv(args.image_path)
    smiles, molblock, token_scores, edges_scores = model.predict_images_from_csv(all_path["image_path"], args.batch_size)
    # all_path.insert(all_path.shape[1], 'Predicted_SMILES', smiles)
    smiles, broken_rate, low_score_rate = postprocess_smiles(smiles, token_scores)
    print(f"broken rate provided by the molscribe is: {broken_rate}")
    print(f"low quality rate provided by the molscribe is: {low_score_rate}")
    # print(smiles)
    # post process
    # all_path.insert(all_path.shape[1], 'SMILES', smiles)
    all_path["SMILES"] = smiles
    # all_path.insert(all_path.shape[1], 'token_scores', token_scores)
    # all_path.insert(all_path.shape[1], 'edges_scores', edges_scores)
    all_path.to_csv(args.image_path, index=False)
    print(f"save to {args.image_path}")
    print("done")
