# HuiduRep
Representation Learning Framework for Extracellular Recordings

This is Huidu, our ragdoll cat! ❤
![E519ED435DB70E9EEB1085CFFF7B8EB6](https://github.com/user-attachments/assets/9ee3624a-e004-4af5-a8e4-e520340fe7e3)

# Overall Architecture
![Spike Identification Workflow Using Transformer Encoding](https://github.com/user-attachments/assets/f10c9931-fd89-4fc4-bbc8-e7574c24870b)

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
