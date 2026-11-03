# HWANG: Explainable Quantum Artificial Intelligence for Target Deconvolution in Phenotypic Drug Discovery

This repository provides the code and prepared dataset for HWANG, a quantum kernel model for target deconvolution in phenotypic drug discovery.

Phenotypic drug discovery identifies biologically active compounds through their cellular effects, but the molecular targets responsible for those effects can be unknown. HWANG combines chemical features with Cell Painting-derived morphological profiles to rank candidate gene targets. On the RxRx3-core benchmark, HWANG achieved a median average precision (AP) of 0.816 and a mean AP of 0.670 across 484 compounds in ten-fold compound-cluster cross-validation. Feature attribution analyses identified chemical similarity to annotated reference compounds as the dominant predictive feature, while model component analyses showed that input scaling affected quantum kernel performance.

## Dataset

`data/prepared/` contains the chemical features, morphological embeddings, compound and gene metadata, and compound clusters used in the benchmark. `data/annotations/compound_gene_interactions.csv` contains the compound–gene activity annotations. The prepared data are derived from [RxRx3-core](https://huggingface.co/datasets/recursionpharma/rxrx3-core), and the annotations are from [EFAAR benchmarking](https://github.com/recursionpharma/EFAAR_benchmarking).

`data/zz_fold_params.json` contains the selected ZZ-kernel hyperparameters for each of the ten outer folds.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

**HWANG target prediction (ten-fold cross-validation):**

```bash
python scripts/run_hwang.py --data data/prepared --annotations data/annotations --output outputs
```

The script refits the ZZ kernel model using the selected hyperparameters for each fold and saves the compound–gene scores and benchmark metrics. Set `PYTHONHASHSEED=0` before running to reproduce the reported evaluation.

## Citation

Shin H, Hwang KY. *Explainable quantum artificial intelligence for target deconvolution in phenotypic drug discovery*. Manuscript in preparation.

Contact: [chahong@korea.ac.kr](mailto:chahong@korea.ac.kr)

## Code Availability

This repository contains the ZZ kernel model and fold-specific hyperparameters used for the HWANG benchmark. Quantum circuits are evaluated using noiseless statevector simulation. The benchmark uses sampled unannotated genes as negatives. The code reproduces cross-validation models and does not provide a single model fitted to the complete dataset.
