## GAT - Graph Attention Network (PyTorch) :computer: + graphs + :mega: = :heart:
This repo contains PyTorch implementation of the original GAT paper (:link: [Veličković et al.](https://arxiv.org/abs/1710.10903)). <br/>
It's aimed at making it **easy to start playing and learning** about GAT and GNNs in general. <br/>

## Table of Contents
* [What are graph neural networks and GAT?](#what-are-gnns)
* [Setup](#setup)
* [Usage](#usage)
* [Hardware requirements](#hardware-requirements)
* [Learning material](#video-learning-material)
    
## What are GNNs?

Graph neural networks are a family of neural networks that are dealing with signals defined over graphs!

Graphs can model many interesting natural phenomena so you'll see them used everywhere from:
* computational biology - predicting potent antibiotics like halicin
* computational pharmacology - predicting drug side effects
* traffic forecasting - used in Google Maps
* recommendation systems (used at Pintrest, Uber, Twitter, etc.) 

and all the way to particle physics in Large Hedron Collider (LHC).

GAT is a representative of spatial (convolutional) GNNs. Since CNNs had a tremendous success in the field of computer vision,
researchers decided to generalize it to graphs and here we are! :nerd_face:

Here is a schematic of GAT's structure:

<p align="center">
<img src="data/readme_pics/GAT_schematic.PNG" width="600"/>
</p>

## Cora visualized

You can't start talking about GNNs without mentioning the single most famous graph dataset - the Cora citation network.

Nodes in Cora represent research papers and the links are, you guessed it, citations between those papers.

I've added a utility for visualizing Cora and doing basic network analysis, here is how Cora looks like:

<p align="center">
<img src="data/readme_pics/cora_graph.PNG" width="700"/>
</p>

Node size corresponds to it's degree (i.e. the number of in/out going edges). Edge thickness roughly corresponds
to how "popular" or "connected" that edge is (betweenes similarity is the nerdy term check out the code.)

And here is the plot showing degree distribution on Cora:

<p align="center">
<img src="data/readme_pics/cora_degree_statistics.PNG" width="600"/>
</p>

## Attention visualized

## Analyzing the embedding space

## Setup

So we talked about what GNNs are, and what they can do for you (among other things). <br/>
Let's get this thing running! Follow the next steps:

1. `git clone https://github.com/gordicaleksa/pytorch-GAT`
2. Open Anaconda console and navigate into project directory `cd path_to_repo`
3. Run `conda env create` from project directory (this will create a brand new conda environment).
4. Run `activate pytorch-gat` (for running scripts from your console or setup the interpreter in your IDE)

That's it! It should work out-of-the-box executing environment.yml file which deals with dependencies. <br/>

-----

PyTorch pip package will come bundled with some version of CUDA/cuDNN with it,
but it is highly recommended that you install a system-wide CUDA beforehand, mostly because of the GPU drivers. 
I also recommend using Miniconda installer as a way to get conda on your system.
Follow through points 1 and 2 of [this setup](https://github.com/Petlja/PSIML/blob/master/docs/MachineSetup.md)
and use the most up-to-date versions of Miniconda and CUDA/cuDNN for your system.

## Usage
story on 3 imps

### Evaluating and profiling GAT
profiling info

### Tracking using Tensorboard

## Hardware requirements

GAT doesn't require super strong HW, especially not if you just want to play with Cora. With 2+ GBs GPU you're good to go.

It takes (on Cora citation network):
* ~10 seconds to train it on my RTX 2080 GPU
* During the training up to 1.5 GBs of VRAM memory is *reserved* (PyTorch's additional caching overhead)
* The model itself has only 365 KBs!

Compare this to hardware needed even for the smallest of [transformers](https://github.com/gordicaleksa/pytorch-original-transformer#hardware-requirements)!

### Future todos:

* Add jupyter notebook (coming soon)
* Add inductive example
* Potentially dd implementation leveraging PyTorch's sparse API

The repo already has everything it needs for learning purposes, these are just the bonus points. I've tested everything
from environment setup, to graph visualizations, etc.

If you have an idea of how to implement GAT using PyTorch's sparse API please feel free to submit a PR.
I personally had difficulties with their API, it's in beta, and it's questionable whether it's at all possible
to make an implementation as efficient as my implementation 3 using it.

## Video learning material

If you're having difficulties understanding GAT I did an in-depth overview of the paper [in this video:](https://www.youtube.com/watch?v=uFLeKkXWq2c&ab_channel=TheAIEpiphany)

<p align="left">
<a href="https://www.youtube.com/watch?v=uFLeKkXWq2c" target="_blank"><img src="https://img.youtube.com/vi/uFLeKkXWq2c/0.jpg" 
alt="The GAT paper explained" width="480" height="360" border="10" /></a>
</p>

I have some more videos which could further help you understand GNNs:
* [My overview of the GCN paper](https://www.youtube.com/watch?v=VyIOfIglrUM)
* [My overview of the GraphSAGE paper](https://www.youtube.com/watch?v=vinQCnizqDA)
* [My overview of the PinSage paper](https://www.youtube.com/watch?v=ed0NJdqwEyg)
* [My overview of Temporal Graph Networks (TGN)](https://www.youtube.com/watch?v=0tw66aTfWaI)

## Acknowledgements

I found these repos useful (while developing this one):

* [official GAT](https://github.com/PetarV-/GAT) and [GCN](https://github.com/tkipf/gcn)
* [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)
* [DeepInf](https://github.com/xptree/DeepInf) and [pyGAT](https://github.com/Diego999/pyGAT)

## Citation

If you find this code useful, please cite the following:

```
@misc{Gordić2020PyTorchGAT,
  author = {Gordić, Aleksa},
  title = {pytorch-GAT},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/gordicaleksa/pytorch-GAT}},
}
```

## Licence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/gordicaleksa/pytorch-GAT/blob/master/LICENCE)