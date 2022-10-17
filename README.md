<h1 align="center">
  <b>Hard Negative Mixing</b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-3.7-blue.svg" /></a>
       <a href= "https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-1.12-FF0000.svg" /></a>
       <a href= "https://github.com/davidsvy/hard-negative-mixing/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-white.svg" /></a>
</p>

An unofficial PyTorch implementation of the NeurIPS 2020 paper [Hard Negative Mixing for Contrastive Learning](https://proceedings.neurips.cc/paper/2020/file/f7cade80b7cc92b991cf4d2806d6bd78-Paper.pdf).



<p align="center">
  <image src="assets/method.png" />
  Image stolen from the paper.
</p>




Table of contents
===

<!--ts-->
  * [➤ Paper Summary](#paper-summary)
  * [➤ Installation](#installation)
  * [➤ Usage](#usage)
  * [➤ Citations](#citations)
<!--te-->


<a  id="paper-summary"></a>
Paper Summary
===
[Momentum Contrast (MoCo)](https://arxiv.org/abs/1911.05722) is a widely used unsupervised representation learning framework. Given an input sample, MoCo first generates 2 different views (query & key) by applying different augmentations. Then, the two views are fed into separate encoders resulting in a single representation vector for each one (q & k). Next, distance is computed between q & k (positives) as well as between q and a queue of previous samples (negatives). Finally, k is added to the queue. The goal is to minimize the distance between positives and maximize the distance between negatives.

[Hard Negative Mixing for Contrastive Learning](https://proceedings.neurips.cc/paper/2020/file/f7cade80b7cc92b991cf4d2806d6bd78-Paper.pdf) extends MoCo by creating more synthetic negative samples and thus increasing the difficulty. Specifically, for each query representation $q$, the authors propose to create the set $\tilde{Q}$ that contains the top $n$ most similar negatives from the queue. Two types of synthetic negatives are proposed:

- **Type 1**: Sample $x_1, x_2 \in \tilde{Q}$. Create new negative as $x = a x_1 + (1 - a) x_2$, where $a \in U(0, 1)$. Repeat $s_1$ times.
- **Type 2**: Sample $x_1 \in \tilde{Q}$. Create new negative as $x = \beta x_1 + (1 - \beta) q$, where $\beta \in U(0, 0.5)$. Repeat $s_2$ times.

The new synthetic $s_1 + s_2$ vectors serve as negatives along with the queue.


<a  id="installation"></a>
Installation
===
```
git clone https://github.com/davidsvy/hard-negative-mixing
cd hard-negative-mixing
pip install -r requirements.txt
```



<a  id="usage"></a>
Usage
===

```
python train_contrastive.py OPTIONS
```

## DATA:
    -dd, --dir_data         Path to directory where the Caltech256 dataset 
                            wil be stored.

    -do, --dir_out          Path to directory where logs & checkpoints wil 
                            be stored.    

    -i, --img_size          Image size for training.      


## TRAINING:
    -s, --steps             Number of training steps.

    -bs, --batch_size       Batch size for training.

    -ss, --steps_save       How often to save checkpoints.

    -sp, --steps_print      How often to print status. 

    -se, --seed             Seed for reproducibility.

    -d, --device            GPU id. If cuda is available default is gpu 0, 
                            else cpu.

    -lrb, --lr_base         Base learning rate.

    -lrw, --lr_warmup       Warmup learning rate.

    -lrm, --lr_min          Min learning rate.

    -cg, --clip_grad        Clip grad.



## MODEL:
    -a, --arch              Architecture of timm model.

    -r, --resume            Path to checkpoint.

    -dm, --dim_moco         Output dimension for MoCo.

    -km, --k_moco           Queue size for MoCo.

    -mm, --m_moco           Exponential moving average weight for MoCo.

    -tm, --t_moco           Temperature for MoCo.

    -nh, --n_hard           Sample from top n_hard most difficult 
                            negative samples from queue.

    -s1h, --s1_hard         Number of type 1 hard negatives created 
                            for each sample.

    -s2h, --s2_hard         Number of type 2 hard negatives created 
                            for each sample.

    -st1h, --start1_hard    When to begin type 1 hard negative mixing.

    -st2h, --start2_hard    When to begin type 2 hard negative mixing.



<a  id="citations"></a>
Citations
===

```bibtex
@misc{@inproceedings{NEURIPS2020_f7cade80,
    author = {Kalantidis, Yannis and Sariyildiz, Mert Bulent and Pion, Noe and Weinzaepfel, Philippe and Larlus, Diane},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and H. Lin},
    pages = {21798--21809},
    publisher = {Curran Associates, Inc.},
    title = {Hard Negative Mixing for Contrastive Learning},
    url = {https://proceedings.neurips.cc/paper/2020/file/f7cade80b7cc92b991cf4d2806d6bd78-Paper.pdf},
    volume = {33},
    year = {2020}
}
```

