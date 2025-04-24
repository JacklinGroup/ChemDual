"""
Code from https://github.com/blender-nlp/MolT5

```bibtex
@article{edwards2022translation,
  title={Translation between Molecules and Natural Language},
  author={Edwards, Carl and Lai, Tuan and Ros, Kevin and Honke, Garrett and Ji, Heng},
  journal={arXiv preprint arXiv:2204.11817},
  year={2022}
}
```
"""

import argparse
import json
import os.path as osp
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def evaluate(input_file, morgan_r, verbose=False):
    outputs = []
    bad_mols = 0
    correct = 0
    total = 0
    with open(osp.join(input_file)) as f:
        for line in f.readlines():
            try:
                line = json.loads(line)
                gt_smi = line["label"]
                ot_smi = line["predict"]
                correct += len(set(gt_smi.split(".")) & set(ot_smi.split("."))) / max(
                    len(set(gt_smi.split("."))), len(set(ot_smi.split(".")))
                )
                total += 1

                gt_m = Chem.MolFromSmiles(gt_smi)
                ot_m = Chem.MolFromSmiles(ot_smi)

                if ot_m == None:
                    raise ValueError("Bad SMILES")
                outputs.append((gt_m, ot_m))
            except:
                bad_mols += 1
    validity_score = len(outputs) / (len(outputs) + bad_mols)
    if verbose:
        print("validity:", validity_score)
        print(
            f"Accuracy: {correct:.2f} / {total} = {(correct / total):.2f}",
        )

    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []

    for gt_m, ot_m in tqdm(outputs):

        MACCS_sims.append(
            DataStructs.FingerprintSimilarity(
                MACCSkeys.GenMACCSKeys(gt_m),
                MACCSkeys.GenMACCSKeys(ot_m),
                metric=DataStructs.TanimotoSimilarity,
            )
        )
        RDK_sims.append(
            DataStructs.FingerprintSimilarity(
                Chem.RDKFingerprint(gt_m),
                Chem.RDKFingerprint(ot_m),
                metric=DataStructs.TanimotoSimilarity,
            )
        )
        morgan_sims.append(
            DataStructs.TanimotoSimilarity(
                AllChem.GetMorganFingerprint(gt_m, morgan_r),
                AllChem.GetMorganFingerprint(ot_m, morgan_r),
            )
        )

    maccs_sims_score = np.mean(MACCS_sims)
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)
    if verbose:
        print("Average RDK Similarity:", rdk_sims_score)
        print("Average MACCS Similarity:", maccs_sims_score)
        print("Average Morgan Similarity:", morgan_sims_score)
    return validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="custom/result/ChemDualv3_w_o_forward_mol_retrosynthesis_test_copy.jsonl",
        help="path where test generations are saved",
    )
    parser.add_argument(
        "--morgan_r", type=int, default=2, help="morgan fingerprint radius"
    )
    args = parser.parse_args()

    evaluate(args.input_file, args.morgan_r, True)
