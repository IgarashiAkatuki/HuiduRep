# HuiduRep
Repository of AAAI 2026 paper: HuiduRep: A Robust Self-Supervised Framework for Learning Neural Representations from Extracellular Recordings.

This paper is also accepted by NeurIPS 2025 Workshop: Foundation Models for the Brain and Body.
![AAAI-Poster](https://github.com/user-attachments/assets/d69793b7-4817-48a6-b007-46ebe48cd1f4)

This is Huidu, our ragdoll cat! ❤
![E519ED435DB70E9EEB1085CFFF7B8EB6](https://github.com/user-attachments/assets/9ee3624a-e004-4af5-a8e4-e520340fe7e3)

# Abstract
Extracellular recordings are transient voltage fluctuations in the vicinity of neurons, serving as a fundamental modality in neuroscience for decoding brain activity at single-neuron resolution. Spike sorting, the process of attributing each detected spike to its corresponding neuron, is a pivotal step in brain sensing pipelines. However, it remains challenging under low signal-to-noise ratio (SNR), electrode drift, and cross-session variability. In this paper, we propose \textbf{HuiduRep}, a robust self-supervised representation learning framework that extracts discriminative and generalizable features from extracellular recordings. By integrating contrastive learning with a denoising autoencoder, HuiduRep learns latent representations robust to noise and drift. With HuiduRep, we develop a spike sorting pipeline that clusters spike representations without ground truth labels. Experiments on hybrid and real-world datasets demonstrate that HuiduRep achieves strong robustness. Furthermore, the pipeline outperforms state-of-the-art tools such as kiloSort4 and MountainSort5. These findings demonstrate the potential of self-supervised spike representation learning as a foundational tool for robust and generalizable processing of extracellular recordings.

# Overall Architecture
<img width="8800" height="4643" alt="image" src="https://github.com/user-attachments/assets/6629a288-054d-4359-a0f6-34890ba4a2a4" />

```
.
│   README.md
│   train.ipynb
│   validate.ipynb
│
├───model
│       CMAES.py
│       FeatureDecoder.py
│       FeatureEncoder.py
│       Prediction.py
│       ProjectionHead.py
│
├───resources
│   └───checkpoint
│           HuiduRep.pt
│
├───train
│       CMAES_trainer.py
│       spike_dataset.py
│
└───utils
        CMAES_utils.py
        monitor_utils.py
        scheduler_utils.py
        spike_augment.py
        test_utils.py
        spikeforest_utils.py
```
- **train.ipynb**: overall training code of HuiduRep.
- **validate.ipynb**: overall validation code of HuiduRep.
- **model**
  - **CMAES.py**: Model definition of Huiduirep.
- **resources**
  - **HuiduRep.pt**: Weights of HuiduRep with $\alpha = 0.2$ and representation dim  $=32$.
- **train**
  - **CMAES_trainer.py**: Define the trainer function for HuiduRep, including the optimiser, learning rate scheduler and other hyperparameters.
  - **spike_dataset.py**: Define the dataset loader used during training, including the use of view augmentation strategies.
- **utils**
  - **CMAES_utils.py**: Contains the `load_model` function to load HuiduRep from its HuiduRep.pt.
  - **monitor_utils.py**: Contains a series of monitor functions with different cluster methods like GMM, Kmeans, HDBSCAN to monitor the effect of model along with evaluate the performance of HuiduRep.
  - **scheduler_utils.py**: Define some learning rate schedulers like `WarmupCosineLR`.
  - **spike_augment.py**: Define whole view augmentation strategies announced in our paper, include jitter, noise, crop, and collision.
  - **test_utils.py**: Define the utils for testing the performance of HuiduRep.
  - **spikeforest_utils.py**: Define the noticed preprocessing steps in the supplementary material as well as the threshold based spike detection method used in the HuiduRep Pipeline.
    
# Cite
```
@article {Cao2025.07.22.666242,
	author = {Cao, Feng and Feng, Zishuo and Shi, Wei and Zhang, Jicong},
	title = {HuiduRep: A Robust Self-Supervised Framework for Learning Neural Representations from Extracellular Recordings},
	elocation-id = {2025.07.22.666242},
	year = {2025},
	doi = {10.1101/2025.07.22.666242},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Extracellular recordings are transient voltage fluctuations in the vicinity of neurons, serving as a fundamental modality in neuroscience for decoding brain activity at single-neuron resolution. Spike sorting, the process of attributing each detected spike to its corresponding neuron, is a pivotal step in brain sensing pipelines. However, it remains challenging under low signal-to-noise ratio (SNR), electrode drift, and cross-session variability. In this paper, we propose HuiduRep, a robust self-supervised representation learning framework that extracts discriminative and generalizable features from extra-cellular recordings. By integrating contrastive learning with a denoising autoencoder, HuiduRep learns latent representations robust to noise and drift. With HuiduRep, we develop a spike sorting pipeline that clusters spike representations without ground truth labels. Experiments on hybrid and real-world datasets demonstrate that HuiduRep achieves strong robustness. Furthermore, the pipeline outperforms state-of-the-art tools such as KiloSort4 and MountainSort5. These findings demonstrate the potential of self-supervised spike representation learning as a foundational tool for robust and generalizable processing of extracellular recordings.Competing Interest StatementThe authors have declared no competing interest.Beihang University, https://ror.org/00wk2mp56, S202510006278},
	URL = {https://www.biorxiv.org/content/early/2025/08/02/2025.07.22.666242},
	eprint = {https://www.biorxiv.org/content/early/2025/08/02/2025.07.22.666242.full.pdf},
	journal = {bioRxiv}
}

```

or

```
@misc{cao2025huidureprobustselfsupervisedframework,
      title={HuiduRep: A Robust Self-Supervised Framework for Learning Neural Representations from Extracellular Recordings}, 
      author={Feng Cao and Zishuo Feng and Wei Shi and Jicong Zhang},
      year={2025},
      eprint={2507.17224},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2507.17224}, 
}
```
