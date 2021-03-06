## Active Decision Boundary Annotation with Deep Generative Models
Miriam W. Huijser and Jan C. van Gemert  
International Conference on Computer Vision 2017 (ICCV): spotlight presentation.

Paper: https://jvgemert.github.io/pub/huijserICCV17ActiveBoundAnnoGAN.pdf  
arXiv preprint: https://arxiv.org/abs/1703.06971

<p align="center"><img src="https://user-images.githubusercontent.com/9445724/31823446-849b2b78-b5ac-11e7-9329-a7c56a6333ff.png" width="500" height="500"/></p>

We provide the code for the decision boundary annotation active learning algorithm.
If you find my code useful for your research, please cite:
```
@article{huijser2017active,
  title={Active Decision Boundary Annotation with Deep Generative Models},
  author={Huijser, Miriam W and van Gemert, Jan C},
  booktitle={International Conference on Computer Vision ({ICCV})},
  year={2017}
}
```

--------------------------------------

### Installation 
Clone this repository recursively because of the adapted ALI submodule:  
`git clone --recursive https://github.com/MiriamHu/ActiveBoundary.git`  

Then install requirements:
```
cd ActiveBoundary
pip install -r requirements.txt
```

### Running the code  
The code is compatible only with Python2.7 (`python`).  
The training code can be run with the following command:  
```
python train.py <query_strategy> [--iterations ITERATIONS] [--enable_gpu] [--oracle_type {line_labeler,noisy_line_labeler,human_line_labeler}] [--dataset {shoebag,mnist08,svhn08}] [--save_path SAVE_PATH] --percentage_labeled PERCENTAGE_LABELED] [--al_batch_size AL_BATCH_SIZE] 
```
`<query_strategy>` should be one of the following:
1. `random` - random sampling, also called "passive learning".
2. `uncertainty` - uncertainty sampling.
3. `uncertainty-dense` - uncertainty-dense sampling.
4. `clustercentroids` - 5-cluster centroids.  

If `--enable-gpu`, the query lines (and points) are generated and saved as *.pdf in the save path.  

For `--oracle_type` `human_line_labeler`, we require `--enable-gpu`. The user is shown an interface in which the decision boundary can be annotated.  

For more (hyper)parameters please refer to `options.py`.

The first time train.py is run for a certain dataset, the required data (and model if gpu is enabled) is downloaded automatically.

### Embedding
Any type of encoding-decoding network can be plugged into our system. See `generative_model.py` how to create your own class.

Our system currently uses the "Adversarially Learned Inference" model (Dumoulin et al., 2016). See https://ishmaelbelghazi.github.io/ALI/ for more information and the code.

--------------------------------------

### Datasets
We provide most encoded and non-encoded datasets from our paper:
- MNIST 0 vs. 8
- SVHN 0 vs. 8
- ShoeBag

To download a dataset either run `train.py --dataset [mnist08, svhn08, shoebag]` or run your own python script with the last lines of code in `options.py`.

--------------------------------------

Please do not hesitate to contact me (huijsermiriam@gmail.com) if you have any questions.  
https://www.aiir.nl/

