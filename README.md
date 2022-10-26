# top-bottom-moseq


## Installation

Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Create and activate an environment called `top_bottom_moseq` with python=3.7:
```
conda create -n top_bottom_moseq python=3.7
conda activate top_bottom_moseq
```

Install opencv and jax
```
conda install -c conda-forge opencv
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Clone or download the [this github repo](https://github.com/calebweinreb/top-bottom-moseq) and pip install:
```
git clone https://github.com/calebweinreb/top-bottom-moseq.git
pip install -e top-bottom-moseq
```

Clone or download the [keypointMoseq repo](https://github.com/calebweinreb/keypointMoSeq) and pip install:
```
git clone https://github.com/calebweinreb/keypointMoSeq.git
pip install -e keypoint_moseq
```

Make the new environment accessible in jupyter 
```
python -m ipykernel install --user --name=top_bottom_moseq
```
