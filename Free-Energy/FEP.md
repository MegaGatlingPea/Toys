# Free Energy Perturbation Method (Original/Bennett Acceptance Ratio)

## 1. 背景

给定相同构象空间下的两个态$0$和$1$，态$0$的能量函数为$U_0(\vec{r})$，态$1$的能量函数为$U_1(\vec{r})$。态$0$和态$1$的亥姆霍兹自由能差可以写为：

$$
\Delta A = A_1 - A_0 = -k_B T \ln \left( \frac{Z_1}{Z_0} \right) = -k_B T \ln \left( \frac{Q_1}{Q_0} \right)
$$

其中$Z_0$和$Z_1$分别是态$0$和态$1$的配分函数，其中$Q_0$和$Q_1$分别是态$0$和态$1$的构象积分:（态$0$和态$1$的配分函数包含广义坐标项和广义动量项，对于该情况配分函数之比等于构象积分之比）

$$
Q_0 = \int d\vec{r} e^{-\beta U_0(\vec{r})}
$$

$$
Q_1 = \int d\vec{r} e^{-\beta U_1(\vec{r})}
$$

## 2. 计算方法

### 2.1 自由能差的计算

核心在计算两个构象积分$Q_0$和$Q_1$之比：

$$
\frac{Q_1}{Q_0} = \frac{Q_1}{Q_0} \frac{\int W(\vec{r}) d\vec{r} e^{-\beta \left(U_0(\vec{r}) + U_1(\vec{r})\right)}}{\int W(\vec{r}) d\vec{r} e^{-\beta \left(U_0(\vec{r}) + U_1(\vec{r})\right)}} = \frac{\langle W\left(\vec{r}\right) e^{-\beta U_1(\vec{r})} \rangle_0}{\langle W\left(\vec{r}\right) e^{-\beta U_0(\vec{r})} \rangle_1}
$$

我们可以通过对两个态的采样来实现构象积分的计算，但是为了获得更可信的结果，我们要选取合适的权重函数$W(\vec{r})$，使得我们通过此计算出来的自由能方差最小。考虑将结果带回自由能差的表达式中，我们得到：

$$
\Delta A = -k_B T \ln \left( \frac{Q_1}{Q_0} \right) = -k_B T \ln \left( \frac{\langle W\left(\vec{r}\right) e^{-\beta U_1(\vec{r})} \rangle_0}{\langle W\left(\vec{r}\right) e^{-\beta U_0(\vec{r})} \rangle_1} \right)
$$

考虑在$0$态采样$n_0$个构象，在$1$态采样$n_1$个构象，使得方差最小，有：
$$
\begin{aligned}
\text{Var}\left(\Delta A\right) &= \text{Var}\left(\ln\langle W\left(\vec{r}\right) e^{-\beta U_1(\vec{r})} \rangle_0\right) + \text{Var}\left(\ln\langle W\left(\vec{r}\right) e^{-\beta U_0(\vec{r})} \rangle_1\right) \\
&= \frac{\text{Var}\left(\langle W\left(\vec{r}\right) e^{-\beta U_1(\vec{r})} \rangle_0\right)}{\langle W\left(\vec{r}\right) e^{-\beta U_1(\vec{r})} \rangle_0^2} + \frac{\text{Var}\left(\langle W\left(\vec{r}\right) e^{-\beta U_0(\vec{r})} \rangle_1\right)}{\langle W\left(\vec{r}\right) e^{-\beta U_0(\vec{r})} \rangle_1^2} \\
\end{aligned}
$$

其中$\langle W\left(\vec{r}\right) e^{-\beta U(\vec{r})} \rangle$期望项可以通过采样+统计的手段得到：

$$
\langle W\left(\vec{r}\right) e^{-\beta U_1(\vec{r})} \rangle_0 = \frac{1}{n_0} \sum_{i=1}^{n_0} W\left(\vec{r}_i\right) e^{-\beta U_1(\vec{r}_i)}
$$

$$
\langle W\left(\vec{r}\right) e^{-\beta U_0(\vec{r})} \rangle_1 = \frac{1}{n_1} \sum_{i=1}^{n_1} W\left(\vec{r}_i\right) e^{-\beta U_0(\vec{r}_i)}
$$

代入原先的方差表达式，有：

$$
\begin{aligned}
\text{Var}\left(\Delta A\right) &= \frac{\text{Var}\left(\langle W\left(\vec{r}\right) e^{-\beta U_1(\vec{r})} \rangle_0\right)}{\langle W\left(\vec{r}\right) e^{-\beta U_1(\vec{r})} \rangle_0^2} + \frac{\text{Var}\left(\langle W\left(\vec{r}\right) e^{-\beta U_0(\vec{r})} \rangle_1\right)}{\langle W\left(\vec{r}\right) e^{-\beta U_0(\vec{r})} \rangle_1^2} \\
&= \frac{\text{Var}\left(\frac{1}{n_0} \sum_{i=1}^{n_0} W\left(\vec{r}_i\right) e^{-\beta U_1(\vec{r}_i)}\right)}{\langle W\left(\vec{r}\right) e^{-\beta U_1(\vec{r})} \rangle_0^2} + \frac{\text{Var}\left(\frac{1}{n_1} \sum_{i=1}^{n_1} W\left(\vec{r}_i\right) e^{-\beta U_0(\vec{r}_i)}\right)}{\langle W\left(\vec{r}\right) e^{-\beta U_0(\vec{r})} \rangle_1^2} \\
&= \frac{1}{n_0} \frac{\langle W\left(\vec{r}\right)^2 e^{-2\beta U_1(\vec{r})} \rangle_0 - \langle W\left(\vec{r}\right) e^{-\beta U_1(\vec{r})} \rangle_0^2}{\langle W\left(\vec{r}\right) e^{-\beta U_1(\vec{r})} \rangle_0^2} + \frac{1}{n_1} \frac{\langle W\left(\vec{r}\right)^2 e^{-2\beta U_0(\vec{r})} \rangle_1 - \langle W\left(\vec{r}\right) e^{-\beta U_0(\vec{r})} \rangle_1^2}{\langle W\left(\vec{r}\right) e^{-\beta U_0(\vec{r})} \rangle_1^2} \\
&= \frac{1}{n_0} \frac{\langle W\left(\vec{r}\right)^2 e^{-2\beta U_1(\vec{r})} \rangle_0}{\langle W\left(\vec{r}\right) e^{-\beta U_1(\vec{r})} \rangle_0^2} + \frac{1}{n_1} \frac{\langle W\left(\vec{r}\right)^2 e^{-2\beta U_0(\vec{r})} \rangle_1}{\langle W\left(\vec{r}\right) e^{-\beta U_0(\vec{r})} \rangle_1^2} - \left(\frac{1}{n_0} + \frac{1}{n_1}\right) \\
\end{aligned}
$$

将构型积分的积分表达式代入，有：

$$
\begin{aligned}
\text{Var}\left(\Delta A\right) &= \frac{\int \left(\frac{Q_0}{n_0}\exp\left(-\beta U_1(\vec{r})\right) + \frac{Q_1}{n_1}\exp\left(-\beta U_0(\vec{r})\right)\right) W\left(\vec{r}\right)^2 \exp\left(-\beta (U_1(\vec{r}) + U_0(\vec{r}))\right) d\vec{r}}{\left[\int W\left(\vec{r}\right) e^{-\beta (U_1(\vec{r}) + U_0(\vec{r}))} d\vec{r}\right]^2} - \left(\frac{1}{n_0} + \frac{1}{n_1}\right) \\
\end{aligned}
$$

现在我们要选取一个合适的权重函数$W(\vec{r})$，使得方差最小。这可以通过对权重函数进行变分来实现，简洁起见，我们定义$N$，$D$和$f$三个函数：

$$
f = \frac{Q_0}{n_0}\exp\left(-\beta U_1(\vec{r})\right) + \frac{Q_1}{n_1}\exp\left(-\beta U_0(\vec{r})\right)
$$

$$
N = \int f(\vec{r}) W\left(\vec{r}\right)^2 e^{-\beta (U_1(\vec{r}) + U_0(\vec{r}))} d\vec{r}
$$

$$
D = \int W\left(\vec{r}\right) e^{-\beta (U_1(\vec{r}) + U_0(\vec{r}))} d\vec{r}
$$

对方差函数求变分：

$$
\delta \text{Var}\left(\Delta A\right) = \frac{\delta N}{D^2} - 2 \frac{N \delta D}{D^3} = 0
$$

化简得到：

$$
\delta N = 2 \frac{N}{D} \delta D
$$

代入具体表达形式：

$$
\int f(\vec{r}) 2W\left(\vec{r}\right) \delta W\left(\vec{r}\right) e^{-\beta (U_1(\vec{r}) + U_0(\vec{r}))} d\vec{r} = 2 \frac{N}{D} \int  \delta W\left(\vec{r}\right) e^{-\beta (U_1(\vec{r}) + U_0(\vec{r}))} d\vec{r}
$$

$$
f(\vec{r}) W(\vec{r}) = \frac{N}{D} = \frac{\int f(\vec{r^\prime}) W(\vec{r^\prime})^2 e^{-\beta (U_1(\vec{r^\prime}) + U_0(\vec{r^\prime}))} d\vec{r^\prime}}{\int W(\vec{r^\prime}) e^{-\beta (U_1(\vec{r^\prime}) + U_0(\vec{r^\prime}))} d\vec{r^\prime}} = Const
$$

$$
W(\vec{r}) = Const \cdot \frac{1}{f(\vec{r})} = Const \cdot \frac{1}{\frac{Q_0}{n_0}\exp\left(-\beta U_1(\vec{r})\right) + \frac{Q_1}{n_1}\exp\left(-\beta U_0(\vec{r})\right)}
$$

这样我们就得到了使得自由能方差最小的权重函数$W(\vec{r})$，代入原先的方差表达式，有：











