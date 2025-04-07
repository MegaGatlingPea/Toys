# Understanding Diffusion Models: A Unified Perspective
给定一组数据$\{x_i\}$，找到这组数据的最大似然分布$p(x)$，考虑引入隐变量$z$进行建模，有以下两种方式（边缘化隐变量或者纯粹基于概率）计算$p(x)$
$$\tag{1} p(x) = \int p(x,z) \mathrm{d}z \cdot$$
$$\tag{2} p(x) = \frac{p(x,z)}{p(z|x)}$$

考虑编码器$\phi$将原始数据$x$编码到隐空间后得到$z$，即$z=\phi(x)$，对$p(x)$取对数后展开得到：

$$
\begin{aligned}
    \tag{3}
    \ln p(x) &= \ln p(x) \int q_{\phi}(z|x) \mathrm{d}z \\
    &= \int q_{\phi}(z|x) \ln p(x) \mathrm{d}z \\
    &= \mathbb{E}_{q_\phi(z|x)}\left[\ln p(x)\right] \\
    &= \mathbb{E}_{q_\phi(z|x)}\left[\ln \frac{p(x,z)q_\phi(z|x)}{p(z|x)q_\phi(z|x)}\right] \\
    &= \mathbb{E}_{q_\phi(z|x)}\left[\ln \frac{p(x,z)}{q_\phi(z|x)}\right] + \mathbb{E}_{q_\phi(z|x)}\left[\ln \frac{q_\phi(z|x)}{p(z|x)}\right]\\
    &= \mathbb{E}_{q_\phi(z|x)}\left[\ln \frac{p(x,z)}{q_\phi(z|x)}\right] + D_{\text{KL}}(q_\phi(z|x) || p(z|x)) \\
    &\geq \mathbb{E}_{q_\phi(z|x)}\left[\ln \frac{p(x,z)}{q_\phi(z|x)}\right]
\end{aligned}
$$

我们训练的目标显然是希望通过编码器$\phi$得到的隐变量分布与我们真实的后验分布尽可能一致，即最小化$D_{KL}(q_\phi(z|x) || p(z|x))$，直接做到这一点是十分困难的。注意到$\ln p(x)$并非是编码器$\phi$的函数，无论采用什么样的编码器$\ln p(x)$都可以视为常数，因此最小化$KL$散度等价于最大化$\mathbb{E}_{q_\phi(z|x)}\left[\ln \frac{p(x,z)}{q_\phi(z|x)}\right]$，即 $\mathrm{ELBO}$。（其实$\text{ELBO}$可以直接从公式1和琴生不等式得到，但是这种方法仅能得到最后的不等式而无法体现$\text{KL}$散度的影响）

接下来进一步考虑$\text{ELBO}$：

$$
\begin{aligned}
    \tag{4}
    \mathbb{E}_{q_\phi(z|x)}\left[\ln \frac{p(x,z)}{q_\phi(z|x)}\right] &= \mathbb{E}_{q_\phi(z|x)}\left[\ln \frac{p_\theta(x|z)p(z)}{q_\phi(z|x)}\right] \\
    &= \mathbb{E}_{q_\phi(z|x)}\left[\ln p_\theta(x|z) \right] + \mathbb{E}_{q_\phi(z|x)}\left[\ln \frac{p(z)}{q_\phi(z|x)}\right] \\
    &= \mathbb{E}_{q_\phi(z|x)}\left[\ln p_\theta(x|z) \right] - D_{\text{KL}}(q_\phi(z|x) || p(z)) \\
    &= \text{reconstruct term} - \text{prior matching term}
\end{aligned}
$$

至此目标已经很清晰了：最大化$\text{ELBO}$就是要最大化$\text{reconstruct}$项，这代表解码器$\theta$从隐变量生成符合原先样本分布的能力；最小化散度项，来让我们学习到的编码器得到的隐变量分布尽可能符合我们真实的隐变量分布（预定义的）。

## Variational AutoEncoder

在变分自编码器中，一般定义编码器和隐变量的分布如下：
$$
q_\phi(z|x) = \mathcal{N}(z;\mathbf{\mu}_\phi(\mathbf{x}),\mathbf{\sigma}_\phi^2(\mathbf{x}))$$
$$
p(z) = \mathcal{N}(z;0;\mathbf{I})
$$
据此，我们其实可以解析求得$\text{prior matching term}$，至于$\text{reconstruct term}$则可以通过$\text{Monte-Carlo}$打点的方法实现，重新写一下我们的目标：

$$
\begin{aligned}
&~~~~~\argmax\limits_{\phi,\theta} \mathbb{E}_{q_\phi(z|x)}\left[\ln p_\theta(x|z) \right] - D_{\text{KL}}(q_\phi(z|x) || p(z)) \\
&=\argmax\limits_{\phi,\theta}\int q_\phi(z|x) \ln p_\theta(x|z) \mathrm{d} z - D_{\text{KL}}(q_\phi(z|x) || p(z)) \\
&\approx\argmax\limits_{\phi,\theta}\sum_{l=1}^L \ln p_\theta(x|z^{(l)}) - D_{\text{KL}}(q_\phi(z|x) || p(z))
\end{aligned}
$$

这里存在一个问题，就是我们在训练的时候要优化编码器参数$\phi$，但是按照目前的反向传播路径，损失的反向传播会在传播到$\phi$的时候断掉，因为$z$是通过$x$编码得到分布后随机采样得到的，没有梯度，这里我们需要引入重参数化技巧：
$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot\epsilon, \epsilon \sim \mathcal{N}(\epsilon;0,\mathbf{I})
$$
这样子我们可以把随机性嫁接到$\epsilon$上，从而通过反向传播实现对编码器参数的优化。
