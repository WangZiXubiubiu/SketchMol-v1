SketchMol. 

## Installation
#### The following command will help you create an environment named sketchmol.
	git clone https://github.com/WangZiXubiubi/SketchMol-v1.git
	cd SketchMol-v1
	conda env create -f environment.yaml 
	conda activate sketchmol # to use sketchmol

#### SketchMol used an open-source OCR recognition method called Molscribe(github.com/thomas0809/MolScribe). Thanks for their marvelous work. However, we haven't resolved the environment conflict issues, so you may need to install the Molscribe environment separately.
	git clone git@github.com:thomas0809/MolScribe.git
	cd MolScribe
 	conda create --name molscribe python=3.7
  	conda activate molscribe
	python setup.py install
 	# we added some functions to enable Molscribe to process the data generated by SketchMol in csv
 	cp -r ~/SketchMol-v1/evaluate/molscribe ~/MolScribe/
  	cp ~/SketchMol-v1/evaluate/predict_csv.py ~/MolScribe/

## Train your own SketchMol
### Before training your own SketchMol, please review the Points to Note for SketchMol (please scroll down) in practical applications. This is the process for physichemical-constrained molecular generation. When working on your own task, please design the labels in advance within the data processing class. 
### Stage0: Create images (by RDKit) 
#### It is recommended to increase the number of CPUs by adjust --num_workers to accelerate the generation process. Please input the path where you want to save the training images.
	python ./data_process/mp_mol_image_generation_for_diffusion.py --num_workers 10 --path_ori path_to_your_directory # image generation
	python ./data_process/mp_scaffold_ori_generation.py --num_workers 10 --path_ori path_to_your_directory # you can also generate scaffold images 
 	python ./data_process/mp_convert_valid_to_invalid.py --num_workers 10 --path_ori path_to_your_directory # Use random deletion of lines to construct invalid images.
  	python ./data_process/calculate_property.py --path_csv path_to_your_csv # calculate various physichemical properties
### Stage1: Train autoencoder 
	CUDA_VISIBLE_DEVICES=<gpu_ids> python main.py --base configs/autoencoder/autoencoder_kl_pubchem400w_32x32x4.yaml -t --gpus 0,
### Stage2: Train diffusion model (don't forget load the autoencoder ckpt in yaml)
	CUDA_VISIBLE_DEVICES=<gpu_ids> python main.py --base configs/ld_molecules/pubchem400w_conditional_various_continuous_32x32x4.yaml -t --gpus 0,
### Stage3: Adjust diffusion model (RLME, Provide molecules that do not meet the expectations for fine-tuning the diffusion model. ）
This is the process in the article for physichemical-constrained molecular generation. You can adjust this process according to the tasks you expect. 
### Stage3.1: Sample some images from the trained model. Sampling is very time-consuming, so please determine the sample size based on your specific task. 
	CUDA_VISIBLE_DEVICES=<gpu_ids> python scripts/sample_diffusion_condition_continuousV2.py -r /path/model.ckpt --conditional_count 40 # sample some images this can come from various samping methods based on your specific task.
 	CUDA_VISIBLE_DEVICES=<gpu_ids> python ～/MolScribe-v1/predict_csv.py --model_path ./ckpt_from_molscribe/swin_base_char_aux_200k.pth --image_path path_to_your_generated_csv.csv
### Stage3.2: Evaluate images. 
In this step, we will save poor molecular images into a separate CSV file. (5% size of the training set). You can add some rules for "invalid" images such as those with too many heterocycles and so on.
For unconditional generation task: I saved 100% invalid images.
For property-constrained generation: I saved over 90% images for invalid images. The model can easily understand basic molecular properties.
For target fine-tuning: I saved over 90% of the images that failed to satisfy the activity discriminator, guiding the model from PubChem to ChEMBL molecule images.

	python evaluate/low_quality_image_various_condtion_continuousV2.py # output a csv path containing all the unsuitable ones. 
### Stage3.3: Adjust the diffusion model. 
 ### Stage3.4: Repeat the above process 1-2 times. 
 ### Sample some examples. When training on your own dataset, remember to adjust the input regular expression according to your new labels. 
#### Sample from single physicochemical property (Supports up to 7 attributes LogP, QED, MW, TPSA, HBD, HBA, RB). The following is an example of sampling molecular images with a molecular weight of 400. You can download the pretrained model from [here](https://drive.google.com/drive/folders/1EM-fCxsRqlGcNaqkWNQUp3N67xGwwpny?usp=drive_link).
	CUDA_VISIBLE_DEVICES=<gpu_ids> python scripts/sample_diffusion_condition_continuousV2.py --scale 1.2 --scale_pro 6.3 --post mw_400 -p "MW:400" -r /path/diffusion_model.ckpt
 	CUDA_VISIBLE_DEVICES=<gpu_ids> python scripts/sample_diffusion_condition_continuousV2.py --scale 4.5 --tri false --post mw_400 -p "MW:400" -r /path/diffusion_model.ckpt
#### Sample from multi physicochemical properties. The following is an example of sampling molecular images with MW=330, HBD=1 & HBA=4. 
	CUDA_VISIBLE_DEVICES=<gpu_ids> python scripts/sample_diffusion_condition_continuousV2.py --scale 1.2 --scale_pro 6.3 --post mw_330_hbd_1_hba_4 -p "MW:330 HBD:1 HBA:4" -r /path/diffusion_model.ckpt
 	CUDA_VISIBLE_DEVICES=<gpu_ids> python scripts/sample_diffusion_condition_continuousV2.py --scale 4.5 --tri false --post mw_330_hbd_1_hba_4 -p "MW:330 HBD:1 HBA:4" -r /path/diffusion_model.ckpt
#### Inpaint from single physicochemical property. The following is an example of sampling molecular image with LogP=4.
	CUDA_VISIBLE_DEVICES=<gpu_ids> python scripts/inpaint_continuousV2.py --scale 1.3 --scale_pro 7.5 --post logp_4 -p "LogP:4" -r /path/model.ckpt --validation_dataset ./scripts/inpainting_csv/example_inpaint_property.csv --zoom_factor 0.98
#### Inpaint from multi physicochemical properties. The following is an example of sampling molecular images with TPSA=40, HBD=1 & HBA=3.
	CUDA_VISIBLE_DEVICES=<gpu_ids> python scripts/inpaint_continuousV2.py --scale 1.0 --scale_pro 8.6 --post tpsa_40_hbd_1_hba_3 -p "TPSA:40 HBD:1 HBA:3" -r /path/model.ckpt --validation_dataset ./scripts/inpainting_csv/example_inpaint_property.csv --zoom_factor 0.98
#### You may redesign a molecule from multi angles.
	CUDA_VISIBLE_DEVICES=<gpu_ids> python scripts/inpaint_continuousV2.py --conditional_count 15 --post multi_logp_4 -p "LogP:4" --scale 2.1 --scale_pro 7.8 -r /path/model.ckpt --validation_dataset ./scripts/inpainting_csv/example_inpaint_multi_angle_logp.csv --zoom_factor 0.98
 
## Examples for Target-Based Molecular Image Generation 
The training logic is similar to the above process. You can train a target-specific SketchMol model yourself or u can download the pretrained SketchMol model for EP4 & yaml from [here](https://drive.google.com/drive/folders/1viEL8vjfpaml9nEUgxyOq_w8IYZyh7YO?usp=drive_link). This can be used to get a general understanding of the SketchMol generation process (a Jupyter notebook version is provided in the scripts folder).
#### Sample molecules for the target protein. The following is an example of a sampling molecular images for EP4. 
	CUDA_VISIBLE_DEVICES=<gpu_ids> python scripts/sample_diffusion_condition_single_protein.py --post example_ep4 -p "EP4:1" -r /path/ep4_model.ckpt 
#### Generate a complete molecule starting from the required fragment. The following is an example of EP4 fragment growing. 
	CUDA_VISIBLE_DEVICES=<gpu_ids> python scripts/inpaint_target.py --post example_ep4_fragment_grow -p "EP4:1" -r /path/ep4_model.ckpt --validation_dataset ./scripts/inpainting_csv/example_fragment_grow.csv --zoom_factor 0.98
#### You may redesign an inhibitor from multi angles.
	CUDA_VISIBLE_DEVICES=<gpu_ids> python scripts/inpaint_target.py --post example_ep4_multi_angle_redesign --conditional_count 15 --scale 2.1 --scale_pro 7.8 -p "EP4:1" -r /path/ep4_model.ckpt --validation_dataset ./scripts/inpainting_csv/example_inpaint_multi_angle_ep4.csv --zoom_factor 0.98

## Detailed explanation of the hyperparameters for sampling molecualr images:
	-r # path to your pretrained ckpt 
 	-p # The expected property string. Please input in the format "Property1:Value1 Property2:Value2". For detailed usage, see the Examples. 
 	--conditional_count # batch size
  	--scale # The degree of orientation towards valid image. The default is 1-20. It can be float. You can also generate images outside this region.
   	--scale_pro # The degree of orientation towards the desired features. The default is 1-20. It can be float.
    	--tri true/false # if false: close scale_pro and use scale to control valid and property together
   	--post # Suffix for the folder name used to save the results.
    	
## Batch conversion from generated images
	# change to molscribe conda environment
	CUDA_VISIBLE_DEVICES=<gpu_ids> python ～/MolScribe/predict_csv.py --model_path ./ckpt_from_molscribe/swin_base_char_aux_200k.pth --image_path path_to_your_generated_csv.csv

## tips: 
1. The output files of SketchMol: After sampling is completed, a CSV file containing the results will be saved, and its path will be printed. The "image_path" column in this csv contains the storage locations of all generated molecular images. MolScribe/predict_csv.py will read all the generated image addresses under this column and add the converted SMILES as a new column.
2. For the inpainting file: You need to additionally input a CSV file (--validation_dataset your_inpainting.csv). The Path column stores the molecules to be processed, and the "Path_keep" column stores the image region you want to retain (sketchmol will mask other areas). For detailed usage, see the Examples.
3. In addition to using --scale to adjust the generation direction, you can also stimulate the model to produce more diverse results in two ways. Here, we take the case of --tri False as an example.
	
 	3.1. Forward Stimulation: When you provide a condition such as "HBA:3", the model mainly generates molecules with a molecular weight less than 300. In this case, you can stimulate the model to generate more diverse molecules by specifying an additional condition (e.g., "HBA:3 MW:380"). This method is particularly useful when generating molecules with higher out-of-domain properties.

	3.2. Reverse Stimulation: When you input "MW:300" as a condition, the generated molecules’ HBD, HBA, and RB values may concentrate at lower numbers. Without changing the input condition, you can set the ddim negvalue to "MW:300 HBD:0 HBA:1 RB:1". As a result, the overall distribution of MW remains largely unchanged, but HBD, HBA, and RB will shift to higher values.
   
5. After training is completed, the probability of the model generating truly low-quality molecules is very low. However, Molscribe tends to assign a relatively low confidence score to many molecules (especially for structurally simple molecules), even though it can recognize their molecular structures. Therefore, please comment out the relevant confidence section during evaluation after training.
6. Since we feed poor-quality images back into model training, we can, of course, also feed high-quality generated data back into the model. This is particularly useful in situations where data is scarce. You can augment your dataset with generated samples by following these steps: 1. Sample molecules from the pretrained model. 2. Filter the generated molecules using an external evaluation model, keeping only those that meet your requirements. 3. Use Tanimoto similarity (with an adjustable threshold, e.g., <0.6 or <0.8) to filter out molecules that are too similar both to the training set and to each other. 4. Label the remaining molecules and finetune the model together with your original training set.

## Points to Note for SketchMol
1. Compared to the currently well-established graph- and sequence-based molecular design methods, SketchMol is a new but naive approach. The current model still has many limitations in both design and application. The training efficiency of SketchMol is not particularly ideal (requiring multiple rounds of training and sampling), so I strongly recommend conducting experiments on a small dataset first. Here, I will point out some issues I frequently encountered during my own use.
2. The sampling process of SketchMol is not stable, and the data generated during sampling is particularly influenced by the --scale and --scale_pro parameters. If you just want to generate some successful examples, there is no need to adjust these parameters significantly. However, if you aim to generate a desired data distribution, you will need to search and fine-tune these parameters. Moreover, the adjustment of the negative vectors (negvalue in code) also affects the direction of sampling. Obtaining an ideal result is extremely time-consuming.
3. The recognition process of SketchMol is merely a compromise solution. OCR recognition software cannot fully identify all generated structures, and this issue becomes more pronounced as molecular size increases. Based on my manual evaluation, approximately 2-15% (along with the structure complexity increases) of the generated structures exhibit recognition errors. Moreover, SketchMol itself has a low probability of generating molecular structures that do not adhere to the input requirements (similar to some natural image generation models). When calculating success rates, these are considered failed cases. However, when tracking molecular generation distributions, especially for continuous properties like QED, _I only consider the middle [10%-90%] as a reference for the generated distribution._ This is because some anomalous generation results can skew the assessment of the main distribution position and affect the evaluation of --scale and --scale_pro tuning.
4. SketchMol has not been trained on any inpainting task. The inpainting process described in the paper is designed based on the characteristics of diffusion models. Therefore, compared to direct generation, the success rate and diversity of inpainting are lower (depending on the specific image). As a result, it is necessary to use OCR software and RDKit to filter out some unreasonable generated samples after inpainting. In my experiments, I only considered readable inpainting samples for evaluation. Specifically, during the inpainting process, SketchMol’s boundary handling capability (for mask edges) is not very robust. Many molecular images exhibit broken bonds or distorted molecular structures at the edges of the filled regions. 
 5. The reason I did not truly use a reinforcement learning model is: (1) Small molecules have too many inherent properties. When introducing new pharmaceutical properties such as toxicity and metabolism, the model becomes significantly more complex. (2) Diffusion models can naturally use labels to adjust the sampling direction, making it easy to incorporate "rewards" within the labels. 

#### Updation:
1. In previous versions, I mistakenly penalized many molecules with excessively high SA and MW, which caused the model to be "afraid" of generating large molecules. After removing these restrictions, the model's understanding of MW has significantly improved, and now you can generate larger molecules by using "MW:800". This does not affect the conclusion, since the readability of large molecules in the 256x256 images is just too poor.
2. Due to the introduction of the generation-adjustment process, the training of SketchMol has become extremely slow, unstable, and is highly susceptible to the influence of external model "rewards". In our next work, we will propose a more stable image-based molecular design model. 

## Acknowledgements
We would like to thank the authors of the following open-source projects, which have been very helpful in the development of SketchMol:
- [MolScribe](https://github.com/thomas0809/MolScribe) - An excellent open-source molecular OCR tool with fast speed and great performance.
- [LDM](https://github.com/CompVis/latent-diffusion) - An excellent open-source image generation algorithm, and one of the representative works of diffusion models.
- [pylsd](https://github.com/primetang/pylsd) - A highly useful open-source line detection tool.
- [imagemol](https://github.com/HongxinXiang/ImageMol) - A highly accurate image-based molecular property prediction tool.

Their work has greatly facilitated the progress and success of our project. If their work has also been helpful to you, please cite them.

## Contact
Since I am leaving the school, I may not be able to receive information related to this GitHub account. If you have any questions, please prioritize contacting wangzx9631@gmail.com.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
