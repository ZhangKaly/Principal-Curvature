<div align="center">

# Principal Curvature Estimation 

</div>

<br>

## 📌  Introduction

**What?**

- This is the repository for the paper "Principal Curvatures Estimation with Applications to Single Cell Data" accepted at ICASSP 2025.
- It contains various of experiments including principal curvature estimation for toy datasets (e.g., sphere, torus, ellipsoid, hyperboloid) and single cell datasets (e.g., Embryonic stem cell data, PBMC data, IPSC data).
- Our model correctly identifies the initial branching point as having negative values of Gaussian curvature indicating saddle-like divergent paths out of the branching point.
- The principal directions on the diverging branch correctly identify the directions in which the cell lineages diverge

**How?**

- By dynamically adjusting neighborhood scales based on the explained variance ratio of local PCA, our model provides robust and accurate curvature estimates without requiring manual parameter tuning.
- This adaptability effectively handles variations in data density and the lack of prior curvature information, making it ideal for complex, diverse datasets.


<br>

## 🚀  Quickstart

```bash
# clone project
git clone git@github.com:ZhangKaly/Principal-Curvature.git
cd Principal-Curvature

# [OPTIONAL] create conda environment
conda create -n PrincipalCurvature python=3.10
conda activate PrincipalCurvature

# install requirements
pip install -r requirements.txt
```



```bash
# train on CPU
python train.py trainer=cpu

# train on 1 GPU
python train.py trainer=gpu

```
<br>

## ❤️  Contributions

This project exists thanks to all the people who contribute.

Have a question? Found a bug? Missing a specific feature? Feel free to file a new issue, discussion or PR with respective title and description.

Before making an issue, please verify that:

- The problem still exists on the current `main` branch.
- Your python dependencies are updated to recent versions.

Suggestions for improvements are always welcome!


