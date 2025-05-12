# RePaint-reimplementation

## Introduction

This project aims to reimplement the paper RePaint: Inpainting using Denoising Diffusion Probabilistic Models. RePaint is a diffusion-based image inpainting method that excels at filling in large or complex missing regions by repeatedly resampling during the reverse diffusion process. Unlike traditional diffusion models that follow a fixed backward trajectory, RePaint introduces jumps-steps where the model intentionally moves backward in time before continuing forward-allowing it to better explore plausible completions and maintain global coherence. This makes it particularly effective for challenging masks, such as alternating lines or irregular holes.

The key contributions of the paper include: (1) a conditioning method that does not require retraining the DDPM, (2) a resampling schedule that improves semantic coherence, and (3) strong empirical results that outperform GAN and autoregressive baselines on multiple mask types.

## Chosen Result

In the original paper, the authors trained a DDPM on the CelebA-HQ dataset for 250,000 iterations, which takes 5 days even on 4×V100 GPUs. Due to resource constraints and the fact that the RePaint algorithm is adaptable to various DDPMs, we utilize pretrained models and focus on the CelebA-HQ dataset to reproduce the following:

- The visualization results using different masks is shown in the figure below:

  <img src="data/original_paper/figure4.png" alt="figure4" width="50%"/>

- The LPIPS results in the table below:
  ![figure4](data/original_paper/table1.png)

- Ablation results on the effect of resampling steps and jump length demonstrated in this image:
  ![figure3](data/original_paper/figure3.png)

This includes the reproduction of the paper's main contribution, the Repaint method, as well as the evaluation results that will serve as proof of the validity of our implementation.

## GitHub Contents

- `code/`: A directory containing the re-implementation code. `repaint.py` is the main file for the algorithm.

- `data/`: A directory containing the datasets used for evaluation. This includes some randomly selected images from the CelebA-HQ dataset and mask images. 

- `results/`: A directory containing the results of re-implementation, including generated images, tables, and the result of our ablation experiments.

- `poster/`: A directory containing a PDF of the poster used for in-class presentation.

- `report/`: A directory containing a PDF of the final report submitted.

- `LICENSE`: MIT License.

- `.gitignore`: A file specifying files or directories that should be ignored by Git.

## Re-implementation Details

We reimplemented the RePaint inpainting method, which modifies the unconditional DDPM's reverse process by conditioning on known regions and resampling (jumping forward in the reverse process). Our approach directly uses a pre-trained ddpm-celebahq-256 model from Google, trained on the CelebA-HQ-256 dataset. No model retraining or architecture modifications were required.

In our reimplementation, we focused on the following components:

* Condition on known area: At each denoising step, noise is added to the known (unmasked) region, preserving its content. The UNet model predicts noise for the masked region, and these are combined to form the inpainted image.

* Bidirectional Resampling: To enhance semantic consistency, we introduced a resampling step where the inpainted region is noised and denoised multiple times. For efficiency, we applied resampling every 10 reverse steps, experimenting with different forward jump lengths.

* Performance Evaluation: We evaluated the method using LPIPS scores on CelebA-HQ-256, comparing our results with the original paper.

## Reproduction Steps

**Prerequisites**: 

- GPU recommended. A CUDA-compatible GPU like A100 or T4. 
- Pre-trained diffusion model (eg. ddpm-celebahq-256). you can download pre-trained models from [Huggingface](https://huggingface.co/google/ddpm-celebahq-256).

**Step by step guide**:
- Step 1: Clone the repository.
```
git clone https://github.com/HongruiTang/RePaint-reimplementation
cd RePaint-reimplementation
```

- Step 2: Create and activate a virtual environment.
```
conda create -n repaint-env python=3.9 -y
conda activate repaint-env
```

- Step 3: Install dependencies
```
pip install -r requirements.txt
``` 

- Step 4: Go to [code/repaint.py](code/repaint.py) file and run the code using 
```
python code/repaint.py
```

In code/repaint.py, you can specify different `resample_steps` and `jump_length` when creating the RePaint scheduler. We have provide some example masks in the data/mask directory which you can use directly. You are encouraged to create your own mask to evaluate the RePaint algorithm.


## Results/Insights

We achieved results comparable to those of the original paper on the CelebA-HQ dataset. In this section, we present our results and compare them to the original paper. 

### Visual Results

The original paper conducted experiments over a wide range
of masks with different test images and compared their
results against several other state-of-the-art methods for
CelebA-HQ inpainting. We also generated images using
the same masks and test images, and visually compared our
results with the original paper's results. As can be seen
from the comparison, our output is similar to the output
from the original paper in terms of the level of detail andsemantic correctness.

![Visual Results](report/Visual_Results.png)



### Evaluation Result

In order to evaluate the performance of the model, we computed the LPIPS score of our model on different masks-lower LPIPS score are desirable as they indicate that image patches are perceptually similar. The following table shows the score from the original paper on the 2nd-to-last row, and our result on the last row.

![CelebA-HQ_Quantitative_Results](report/CelebA-HQ_Quantitative_Results.png)

### Ablation Study

We conducted an ablation study to observe the effect of
using different jump lengths and resample steps on the resulting image. We arrived at a similar conclusion as the
original paper-increasing the resampling steps and jump
length generates more harmonized images, but the benefits
saturate at approximately r = 10 and j = 10.

![Ablation Study](report/Ablation.png)

To reproduce our results, please follow the Reproduction Steps section, which will guide you through setting up the environment and running the code. For evaluating using the LPIPS score, we recommend referring to this [repository](https://github.com/richzhang/PerceptualSimilarity) for further details. This metric is to evaluate the distance between two images. Lower LPIPS scores indicate better perceptual similarity to the ground truth, and therefore, higher quality inpainted results. The expected reproduction result should closely resemble those shown above The generated images should be close to the original paper's result and natural to human observers across various mask types 

## Conclusion

**Key Takeaways**:

RePaint presents a novel conditioning method that enables pretrained unconditional diffusion models to perform inpainting across a wide variety of mask types. Its unified inference schedule ensures harmonization between known and generated regions, resulting in semantically coherent image completions. Our implementation successfully reproduced the original results on CelebA-HQ, achieving comparable LPIPS metrics across multiple mask types. Moreover, our ablation experiments revealed similar performance trends to those reported in the original paper, reinforcing the validity of RePaint’s design.

**Lessons Learned**:

One of the key lessons was the importance of assessing the computational feasibility of a project early on. While the RePaint method is elegant in theory, its iterative sampling procedure is resource-intensive, requiring thoughtful planning and resource management even when we are simply utilizing the base model at inference time instead of training deep models from scratch. Additionally, it was interesting to note that the intuition behind the RePaint method, including the effects of changing various hyperparameters, was actually quite straightforward, which is perhaps further testament to how simple ideas often work best, even in deep learning.

## References

- Lugmayr, A., Danelljan, M., Romero, A., Yu, F., Timofte, R., Van Gool, L., 2022. Repaint: Inpainting using denoising diffusion probabilistic models, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11461–11471.


## Acknowledgements

This project was conducted as the final project of **CS5782 Introduction to Deep Learning** at Cornell University. We would like to thank Professor Kilian Weinberger and Professor Jennifer Sun for providing valuable guidance and feedback. We would also like to thank for helpful comments and suggestions provided by our peers and TAs in the peer-review poster session.

We also acknowledge the authors of the original RePaint paper for making their work publicly available and inspiring our reimplementation efforts. The pre-trained models and datasets used in this project were sourced from publicly accessible repositories on Hugging Face. 
