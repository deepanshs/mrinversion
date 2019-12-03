# coding: utf-8
# # Table of Contents
#
# 1. [Linear Algebra and least square problem on the subspace](#section2)<br />
#     - [Singular Value Decomposition (SVD)](#svd)<br />
#     - [Determining truncation point for singular values](#determine_r)<br />
#     - [Truncated SVD](#TSVD)<br />
#     - [Least square problem on the reduced subspace](#reducedsubspace)<br />
# In[5]:
import numpy as np

# <a id='section2'></a>
#
# # Linear Algebra and least square problem on the subspace
#
# A linear system when represented in the matrix form follows<br /><br />
# $${\bf K}~ {\bf f} = {\bf s}$$<br />
# where ${\bf K} \in \mathbb{R}^{m\times n}$ is a matrix called the 'kernel' and
# ${\bf s} \in \mathbb{R}^{m}$, ${\bf f} \in \mathbb{R}^{n}$ are column vectors of
# measured data and solution to be determined, respectively. Here it is assumed that
# both the kernel ${\bf K}$ and the data ${\bf s}$ is known. The data vector ${\bf s}$
# is often contaminated with errors from the measurement noise and therefore the above
# problem is more commonly presented as the least square (LS) problem <br /><br />
# $${\bf f}_{LS} = \underset{{\bf f}}{\text{argmin}} \| {\bf {K}~f - s} \|^2_2.$$
#
# When ${\bf K}$ is ill-conditioned, a situation that is commonly uncounted, the
# solution ${\bf f}_{LS}$ becomes highly unstable. In such cases regularization methods
# are implemented to stabilize the solution. Here, a hybrid singular value
# decomposition (SVD) and $l_1$ regularization is used to stabilize the solution.

# <a id='svd'></a>
# ##### Singular Value Decomposition (SVD)
# The Singular Value Decomposition (SVD) of a Kernel ${\bf K} \in
# \mathbb{R}^{m\times n}$ is defined as <br /><br />
# $${\bf {K}} = {\bf U}~ \pmb{S} ~{\bf V}^T$$
# <br />
# where ${\bf U} \in \mathbb{R}^{m\times m}$, ${\bf V} \in \mathbb{R}^{m\times m}$ are
# the left and the right unitary matrices with columns $u_i$ and $v_i$, respectively
# called left and right singular vectors. <br />
# $\pmb{S} \in \mathbb{R}^{m \times n}$ is a diagonal matrix with diagonal elements
# called the singular values. The singular values, ${\varsigma}_i$ are positive and
# arrange in decending order.

# In[6]:


# <a id='determine_r'></a>
# ##### Determining truncation point for singular values, ${\varsigma}_i$
# Singular values, ${\varsigma}_i$ that are zero or near zero are often harmful in
# linear inversion as it result in a unstable solution. The common approach to
# regularization the solution is the truncation of singular value, say at
# ${\varsigma}_r$. The following criterion to determine the truncation point is based
# on Ranking by SVD-Entropy as described
# [here](https://doi.org/10.1093/bioinformatics/btl214). The following is a brief
# summary.
#
# The normalized $j^\text{th}$ Eigen values of matrix ${\bf K K}^T$ is <br />
# $$\mathcal{S}_i =\frac{{\varsigma}_j^2}{ \sum_{k=1}^l \varsigma_k^2}$$<br /><br />
# where $l=\text{min}(m,n)$ and ${\varsigma}_j$'s are the singular values of the kernel
# ${\bf K}$.
#
# Let the dataset entropy be defined as
# $$E = -\frac{1}{\log(l)} \sum_{j=1}^l \mathcal{S}_j \log(\mathcal{S}_j)$$
#
# The contribution of the $i^\text{th}$ feature to the total Entropy $E$ is defined by
# leave one out comparision. Let $E_i$ be the entropy with the $i^\text{th}$
# contribution removed. <br /><br />
# $$ E_i = -\frac{1}{\log(l-1)} \left(\sum_{j=1}^l \mathcal{S}_j \log(\mathcal{S}_j)
# - \mathcal{S}_i \log(\mathcal{S}_i) \right)$$
#
# The contribution of the $i^\text{th}$ feature to the total Entropy $E$ is then
# defined as<br /><br />
# $$ \nabla E_i = E - E_i = E - \frac{E \log (l) + \mathcal{S}_i
# \log(\mathcal{S}_i)}{\log(l-1)}$$
#
# Defining $c$ and $d$ as the mean and standard deviation of $\nabla E_i$, the
# optimum truncation point, $r$, is given as<br /><br />
# $$r = \underset{i}{\text{argmin}} \left(\nabla E_i -c + d \right).$$

# In[7]:


def find_optimum_singular_value(s):
    length = s.size
    s2 = s ** 2.0
    sj = s2 / s2.sum()
    T = sj * np.log10(sj)
    logn = np.log10(length)
    lognm1 = np.log10(length - 1.0)
    entropy = (-1.0 / logn) * T.sum()

    deltaEntropy = entropy - (entropy * logn + T) / lognm1

    c = deltaEntropy.mean()
    d = deltaEntropy.std()

    r = np.argmin(deltaEntropy - c + d)
    return r


# <a id='TSVD'></a>
# ##### Truncated SVD
#
# The left and the right unitary matrix ${\bf U} \in \mathbb{R}^{m\times m}$ and
# ${\bf V} \in \mathbb{R}^{m\times m}$ are truncated to ${\bf U}_r \in
# \mathbb{R}^{m\times r}$ and ${\bf V}_r \in \mathbb{R}^{m\times r}$, respectively.
# The singular matrix $\pmb{S} \in \mathbb{R}^{m \times n}$ is truncated to $\pmb{S}
# \in \mathbb{R}^{r \times r}$. The truncated singular value decomposition result in a
# kernel ${\bf K}_\text{TSVD}$ that is an approximation of the original kernel
# ${\bf K}$.
#
# $$ {\bf K}_\text{TSVD} = {\bf U}_r~ \pmb{S}_r ~ {\bf V}_r^T $$

# In[8]:


def TSVD(K):
    U, S, VT = np.linalg.svd(K, full_matrices=False)
    r = find_optimum_singular_value(S)
    return U, S, VT, r


# <a id='reducedsubspace'></a>
# ##### Least square problem on the reduced subspace
#
# The approximate kernel, ${\bf K}_\text{TSVD}$, is a 'nearby' well conditioned and
# creates a new system of linear equations <br /><br />
# $$ {\bf K}_\text{SVD} {\bf f}_\dagger = {\bf s}$$<br />
# whose solution, ${\bf f}_\dagger$, is stable and 'close' to the solution of the
# original problem. This problem can further be reduced on a subspace as follows
# $$
# \begin{array}{rl}
# {\bf U}_r~ \pmb{S}_r ~ {\bf V}_r^T {\bf f}_\dagger &= {\bf s}\\
# \pmb{S}_r ~ {\bf V}_r^T {\bf f}_\dagger &= {\bf U}^T_r ~{\bf s}\\
# \tilde{\bf K} ~{\bf f}_\dagger &= \tilde{\bf s}\\
# \end{array}
# $$
# where $\tilde{\bf K} \in \mathbb{R}^{r\times n}$ is the 'nearby' well-conditioned
# kernel on the subspace and $\tilde{\bf s} \in \mathbb{R}^{r}$ is the compressed data.
# The least square problem (LS) similarly reformulates to<br /><br />
# $$\underset{\bf f}{\text{argmin}} \| \tilde{\bf K}  {\bf f}_\dagger - \tilde {\bf s}
# \|_2^2$$
#
# The projected or uncompressed data, ${\tilde{\tilde{\bf s}}} \in \mathbb{R}^{m}$
# follows
# $$ \tilde{\tilde{\bf s}} =  {\bf U}_r~ \tilde {\bf s} =  {\bf U}_r~ {\bf U}_r^T
# ~ {\bf s}$$
#
#

# In[9]:


def reduced_subspace_kernel_and_data(U, S, VT, signal):
    diagS = np.diag(S)
    K_tilde = np.dot(diagS, VT)
    s_tilde = np.dot(U.T, signal)
    projectedSignal = np.dot(U, s_tilde)

    K_tilde = np.asfortranarray(K_tilde)
    s_tilde = np.asfortranarray(s_tilde)
    projectedSignal = np.asfortranarray(projectedSignal)
    return K_tilde, s_tilde, projectedSignal
