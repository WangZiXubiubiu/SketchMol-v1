Official implementation of SketchMol.

# Installation
	git clone https://github.com/WangZiXubiubi/SketchMol-v1.git
	cd SketchMol-v1
	pip install -r requirements.txt
	conda activate sketchmol
 
# Usage 
## Example: Sample from molecule property logp=2 MW=350 (Supports up to 7 attributes LogP,QED,MW,TPSA,HBD,HBA,RB)
	CUDA_VISIBLE_DEVICES=<gpu_ids> python scripts/sample_diffusion_condition_continuousV2.py -p "LogP:2 MW:350" -r /path/model.ckpt
## Example: Inpaint molecule MW=300
	CUDA_VISIBLE_DEVICES=<gpu_ids> python scripts/inpaint_continuousV2.py -p "MW:300" -r /path/model.ckpt --validation_dataset /path/dataset.csv

# Train your own SketchMol
## Stage0: Creating images (by RDKit)
	python 
## Stage1: Train autoencoder 
	CUDA_VISIBLE_DEVICES=<gpu_ids> python main.py --base configs/autoencoder/autoencoder_kl_pubchem400w_32x32x4.yaml -t --gpus 0,
## Stage2: Train diffusion model
	CUDA_VISIBLE_DEVICES=<gpu_ids> python main.py --base configs/ld_molecules/pubchem400w_conditional_various_continuous_32x32x4.yaml -t --gpus 0,
## Stage3: 

#### Some of the code is built from ( ) ( ), thanks to their extraordinary work.

# Contact
If you have any questions, please feel free to contact [Zixu Wang](s2230167@u.tsukuba.ac.jp).

# License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
