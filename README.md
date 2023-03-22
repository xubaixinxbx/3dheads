# Deformable Model Driven Neural Rendering for High-fidelity 3D Reconstruction of Human Heads Under Low-View Settings
Code for "Deformable Model Driven Neural Rendering for High-fidelity 3D Reconstruction of Human Heads Under Low-View Settings"

[Baixin Xu](https://github.com/xubaixinxbx/High-fidelity-3D-Reconstruction-of-Human-Heads), [Jiarui Zhang](https://github.com/xubaixinxbx/High-fidelity-3D-Reconstruction-of-Human-Heads), [Kwan-Yee Lin](https://kwanyeelin.github.io/), [Chen Qian](https://scholar.google.com/citations?user=AerkT0YAAAAJ&hl=zh-CN), [Ying He](https://personal.ntu.edu.sg/yhe/).

### [Project](https://github.com/xubaixinxbx/High-fidelity-3D-Reconstruction-of-Human-Heads) | [Paper](https://github.com/xubaixinxbx/High-fidelity-3D-Reconstruction-of-Human-Heads)

We propose a robust method for learning neural implicit functions that can reconstruct 3D human heads with high-fidelity geometry from low-view inputs. We represent 3D human heads as the zero level-set of a composed signed distance field that consists of a smooth template, a non-rigid deformation, and a high-frequency displacement field. The template represents identity-independent and expression-neutral features, which is trained on multiple individuals, along with the deformation network. The displacement field encodes identity-dependent geometric details, trained for each specific individual. We train our network in two stages using a coarse-to-fine strategy without 3D supervision. Our experiments demonstrate that the geometry decomposition and two-stage training make our method robust and our model outperforms existing methods in terms of reconstruction accuracy and novel view synthesis under low-view settings. Additionally, the pre-trained template serves a good initialization for our model to adapt to unseen individuals.

<img src='./misc/arch_pipeline.png' width=800>

## Video

<!-- <video controls src="./misc/video_compress.mp4"></video> -->



https://user-images.githubusercontent.com/113180125/226843465-432415fb-6ee3-44eb-930f-8d8a185f8a6d.mp4





Note: We upload the compressed video for convenience, please download and see the original [video](https://github.com/xubaixinxbx/High-fidelity-3D-Reconstruction-of-Human-Heads/tree/main/misc).

## Training
This project is developed and tested on...
- You can create an anaconda environment called ... with:
```
to be
```

## Citation

If you find our work useful, please kindly cite as:
```
to be
```

## Acknowledgement
* The codebase is developed based on [VolSDF](https://github.com/lioryariv/volsdf) of Yariv et al. Many thanks to their great contributions!
