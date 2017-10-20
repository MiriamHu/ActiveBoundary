## ActiveBoundary
Miriam W. Huijser and Jan C. van Gemert  
International Conference on Computer Vision 2017 (ICCV): spotlight presentation.

Paper: https://jvgemert.github.io/pub/huijserICCV17ActiveBoundAnnoGAN.pdf  
arXiv preprint: https://arxiv.org/abs/1703.06971

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
Clone this repository recursively (this because of the adapted ALI submodule):  
`git clone --recursive https://github.com/MiriamHu/ActiveBoundary.git`  

Then install requirements:
```
cd ActiveBoundary
pip install -r requirements.txt
```

### Running the code  
The code is compatible only with both Python2.7 (`python`).  
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

--------------------------------------

Please do not hesitate to contact me (huijsermiriam@gmail.com) if you have any questions.
