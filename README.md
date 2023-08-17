# Deformable Model-Driven Neural Rendering for High-Fidelity 3D Reconstruction of Human Heads Under Low-View Settings
Code for "Deformable Model-Driven Neural Rendering for High-Fidelity 3D Reconstruction of Human Heads Under Low-View Settings"

[Baixin Xu](https://xubaixinxbx.github.io/), [Jiarui Zhang](https://github.com/xubaixinxbx/3dheads), [Kwan-Yee Lin](https://kwanyeelin.github.io/), [Chen Qian](https://scholar.google.com/citations?user=AerkT0YAAAAJ&hl=zh-CN), [Ying He](https://personal.ntu.edu.sg/yhe/).

### Project | [Paper](https://arxiv.org/abs/2303.13855)

Reconstructing 3D human heads in low-view settings presents  technical challenges, mainly due to the pronounced risk of overfitting with limited views and high-frequency signals. To address this, we propose geometry decomposition and adopt a two-stage, coarse-to-fine training strategy, allowing for progressively capturing high-frequency geometric details. We represent 3D human heads using the zero level-set of a combined signed distance field, comprising a smooth template, a non-rigid deformation, and a high-frequency displacement field. The template captures features that are independent of both identity and expression and is co-trained with the deformation network across multiple individuals with sparse and randomly selected views. The displacement field, capturing individual-specific details, undergoes separate training for each person. Our network training does not require 3D supervision or object masks. Experimental results demonstrate the effectiveness and robustness of our geometry decomposition and two-stage training strategy. Our method outperforms existing neural rendering approaches in terms of reconstruction accuracy and novel view synthesis under low-view settings. Moreover, the pre-trained template serves a good initialization for our model when encountering unseen individuals. 

<img src='./misc/arch_pipeline.png' width=800>

## Video

<!-- <video controls src="./misc/video_compress.mp4"></video> -->



https://user-images.githubusercontent.com/113180125/226843465-432415fb-6ee3-44eb-930f-8d8a185f8a6d.mp4





Note: We upload the compressed video for convenience, please download and see the original [video](https://github.com/xubaixinxbx/High-fidelity-3D-Reconstruction-of-Human-Heads/tree/main/misc).

## Training
```
python training/exp_runner.py --conf ./conf/st1.conf --scan_id 399 --gpu 0 
```

## Citation

If you find our work useful, please kindly cite as:
```
@article{xu2023deformable,
  title={Deformable Model Driven Neural Rendering for High-fidelity 3D Reconstruction of Human Heads Under Low-View Settings},
  author={Xu, Baixin and Zhang, Jiarui and Lin, Kwan-Yee and Qian, Chen and He, Ying},
  journal={arXiv preprint arXiv:2303.13855},
  year={2023}
}
```

## Acknowledgement
* The codebase is developed based on [VolSDF](https://github.com/lioryariv/volsdf) of Yariv et al. Many thanks to their great contributions!
