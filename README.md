# Sparse Coding (Olshausen & Field, 1996) Model
Sparse Coding (Olshausen & Field, 1996) Model with **LCA** (locally competitive algorithm) ([Rozell et al., *Neural Comput*. 2008](https://www.ece.rice.edu/~eld1/papers/Rozell08.pdf)).

Data is from <http://www.rctn.org/bruno/sparsenet/>.

## Requirement
- `Python >= 3.5`
- `numpy`, `matplotlib`, `scipy`,  `tqdm`, `sklearn`

## Usage
- Run `train.py` or `predictive-coding.ipynb` (written in Japanese).
- `ica.py` is implementation of ICA and PCA for Natural images.

## Results
### Loss function
<img src="https://raw.githubusercontent.com/takyamamoto/SparseCoding-OlshausenField-Model/master/results/error.png" width="500px"> 

### Receptive fields (using soft threshold function)
<img src="https://raw.githubusercontent.com/takyamamoto/SparseCoding-OlshausenField-Model/master/results/RF.png" width="500px"> 

### Receptive fields (using Cauchy threshold function (Mayo et al., 2020))
<img src="https://raw.githubusercontent.com/takyamamoto/SparseCoding-OlshausenField-Model/master/results/RF_cauchy_thresholding.png" width="500px"> 

### ICA
<img src="https://raw.githubusercontent.com/takyamamoto/SparseCoding-OlshausenField-Model/master/results/ICA.png" width="500px"> 

### PCA
<img src="https://raw.githubusercontent.com/takyamamoto/SparseCoding-OlshausenField-Model/master/results/PCA.png" width="500px"> 

## Reference
- Olshausen BA, Field DJ. [Emergence of simple-cell receptive field properties by learning a sparse code for natural images](https://www.nature.com/articles/381607a0). *Nature*. 1996;381(6583):607–609. [Data and Code](http://www.rctn.org/bruno/sparsenet/), [pdf](https://courses.cs.washington.edu/courses/cse528/11sp/Olshausen-nature-paper.pdf)
- Rozell CJ, Johnson DH, Baraniuk RG, Olshausen BA. [Sparse coding via thresholding and local competition in neural circuits](http://www.mit.edu/~9.54/fall14/Classes/class07/Palm.pdf). *Neural Comput*. 2008;20(10):2526‐2563.
- Mayo P, Holmes R, Achim A. [Iterative Cauchy Thresholding: Regularisation with a heavy-tailed prior](https://arxiv.org/abs/2003.12507). arXiv. 2020.