---
title: "Day 1｜信息论：熵、KL 散度与互信息"
date: 2026-08-11
summary: "概率机器学习 Day 1 学习笔记：熵、交叉熵、联合熵、条件熵、KL 散度、互信息、最大信息系数、数据处理不等式与充分统计量。"
categories:
  - Machine Learning
tags:
  - Probability
  - Information Theory
  - Entropy
  - KL Divergence
  - Mutual Information
  - MIC
math: true
draft: false
---

## 1.1 熵

> 一种混乱程度
> 具有较高的熵，认为具有较高的信息含量

### 1.2 离散变量
#### 1.2.1 离散变量的熵
$K$ 个状态上分布为 $p$ 的离散随机变量 $X$ 的熵被定义为
$$
\mathbb{H}(x)\triangleq-\sum_{k=1}^{K}p(X=k)\log_2p(X=k)=-\mathbb{E}_X\left[\log p(X)\right]
$$
#### 1.2.2 重要的性质

- 具有最大熵的离散分布是均匀分布
- 具有最小熵的分布是任何将其所有概率质量放在一个状态上的$\delta$函数

证明在介绍完KL散度后证明
### 1.3 交叉熵
分布 $p$ 和 $q$ 的交叉熵被定义为
$$
\mathbb{H}(p,q)\triangleq-\sum_{k=1}^{K}p_k\log q_k
$$
特别的，对于固定分布 $p$，当 $q=p$ 时，交叉熵取得最小值
### 1.4 联合熵
两个随机变量 $X$ 和 $Y$ 的联合熵可以被定义为
$$
\mathbb{H}(X,Y)=-\sum_{x,y}p(x,y)\log_2p(x,y)
$$
#### 1.5 条件熵
当给定 $X$ 时， $Y$ 的条件熵是在观察到 $X$ 后，对 $Y$ 的不确定性在 $X$ 的所有可能值熵取平均值
$$
\begin{align*}
\mathbb{H}(Y|X)&\triangleq\mathbb{E}_{p(x)}\left[\mathbb{H}(p(Y|X))\right]\\
&=\sum_{x}p(x)\mathbb{H}(p(Y|X=x))=-\sum_{x}p(x)\sum_{y}p(y|x)\log p(y|x)\\
&=-\sum_{x,y}p(x,y)\log p(y|x)=-\sum_{x,y}p(x,y)\log \frac{p(x,y)}{p(x)}\\
&=-\sum_{x,y}p(x,y)\log p(x,y)-\sum_{x}p(x)\log \frac{1}{p(x)}\\
&=\mathbb{H}(X,Y)-\mathbb{H}(X)
\end{align*}
$$
我们可以令 $X=X_1$ 、$Y=X_2$，则有$\mathbb{H}(X_1,X_2)=\mathbb{H}(X_1)+\mathbb{H}(X_2|X_1)$，进而有熵的链式法则
$$
\mathbb{H}(X_1,X_2,...,X_n)=\sum_{i=1}^{n}\mathbb{H}(X_i|X_1,...,X_{i-1})
$$
#### 1.6 困惑度
离散概率分布 $p$ 的困惑度定义为
$$
\text{perplexity}\triangleq2^{\mathbb{H}(p)}
$$
### 1.7 连续变量的微分熵
若 $X$ 是一个具有概率密度函数 $p(X)$ 的连续随机变量，我们将微分熵定义为
$$
h(x)\triangleq-\int_X p(x)\log p(x) dx
$$

### 1.8 相对熵
对于离散分布，我们使用 $\mathbb{KL}$ 散度进行定义
$$
\mathbb{KL}(p\|q)\triangleq\sum_{k=1}^{K}p_k \log \frac{p_k}{q_k}
$$
对于连续分布
$$
\mathbb{KL}(p\|q)\triangleq\int p(x) \log \frac{p(x)}{q(x)}dx
$$
进一步可以解释为：
$$
\mathbb{KL}(p\|q)=\underbrace{\sum_{k=1}^{K}p_k \log p_k}_{-\mathbb{H}(p)}-\underbrace{\sum_{k=1}^{K}p_k \log q_k}_{-\mathbb{H}(p,q)}
$$
$$
\mathbb{KL}(p\|q)=-\mathbb{H}(p)+\mathbb{H}(p,q)
$$
 $\mathbb{KL}$ 散度用于测量两个分布相似程度的方法
#### 1.8.1 重要证明
 - 证明1：$\mathbb{KL}$ 散度非负
	$$
	\begin{align*}
	\mathbb{KL}(p\|q)&=\sum_{i}p_i \log \frac{p_i}{q_i}\\
	&=-\sum_{i}p_i \log \frac{q_i}{p_i}
	\end{align*}
	$$
	 由于 $\log x\le x−1$ ，我们可以得到 $\log\frac{q_i}{​p_i}​​\le \frac{q_i}{p_i}−1$，然后化简有 $p_i\log\frac{q_i}{​p_i}​​\le q_i−p_i$ 
	 $$
	\begin{align*}
    -\mathbb{KL}(p\|q)&\le \sum_{i}(q_i-p_i)\\
    &\le 0
	\end{align*}
	 $$
	 由于概率和为1，则有 $-\mathbb{KL}(p\|q) \le 0$，所以 $\mathbb{KL}(p\|q) \ge 0$。
 - 证明2：具有最大熵的离散分布是均匀分布
	 首先，令 $u_i=\frac{1}{K}$，于是均匀分布为 $u=\big(\frac{1}{K},...,\frac{1}{K}\big)$。
	 考虑概率分布 $p$ 与均匀分布 $u$ 之间的 KL 散度：
	 $$
	 \begin{align*}
	 \mathbb{KL}(p\|u)&=\sum_{i=1}^{K}p_i\log\frac{p_i}{u_i}\\
	 &=\sum_{i=1}^{K}p_i\log\frac{p_i}{1/K}\\
	 &=\sum_{i=1}^{K}p_i\big(\log p_i+\log K\big)\\
	 &=\sum_{i=1}^{K}p_i\log p_i+\log K\sum_{i=1}^{K}p_i
	 \end{align*}
	 $$
	 由于$\sum_{i=1}^{K}p_i=1$,
	 $$
	 \begin{align*}
	 \mathbb{KL}(p\|u)&=\sum_{i=1}^{K}p_i\log p_i+\log K\\
	 &=\log K-\mathbb{H}(p)
	 \end{align*}
	 $$
	 由于 $\mathbb{KL}$ 散度非负，$\log K\ge\mathbb{H}(p)$，当且仅当两个分布相等时，取等号使得熵最大，所以有限离散状态空间上，均匀分布拥有最大熵
	 $Q.E.D.$
 - 证明3：具有最小熵的分布是任何将其所有概率质量放在一个状态上的 $\delta$ 函数
	对于离散随机变量，$\delta$ 分布表示所有概率质量都集中在某一个状态 $x_0$ 上：
	$$  
	p(x)=\left\{\begin{matrix}
			  1,&x=x_0 \\
			  0,&x\ne x_0
			\end{matrix}\right.
	$$
	离散随机变量的熵为
	$$  
	H(p)=-\sum_x p(x)\log p(x)
	$$
	由于对任意概率 $0\le p(x)\le1$，都有
	$$  
	\log p(x)\le0
	$$
	因此
	
	$$  
	-p(x)\log p(x)\ge0
	$$
	所以
	$$  
	H(p)\ge0
	$$
	也就是说，离散熵的最小可能值为 $0$。
	要使
	$$  
	H(p)=0
	$$
	
	由于熵中的每一项都非负，因此必须对所有 $x$ 都有
	
	$$  
	-p(x)\log p(x)=0
	$$
	
	而当 $0<p(x)<1$ 时，
	
	$$  
	-p(x)\log p(x)>0
	$$
	
	所以只有
	$$  
	p(x)=0  
	\quad\text{或}\quad  
	p(x)=1  
	$$
	才能使该项为零。
	再结合概率归一化条件
	$$  
	\sum_x p(x)=1
	$$
	只能有一个状态的概率为 $1$，其余状态的概率全部为 $0$。
	因此，
	$$  
	\boxed{H_{\min}=0}  
	$$
	且最小值恰好由任意 $\delta$ 分布取得。
	$Q.E.D.$

#### 1.9 互信息
衡量两个随机变量的相关性
随机变量 $X$ 和 $Y$ 之间的互信息定义为：
$$
\mathbb{I}(X;Y)\triangleq\mathbb{KL}(p(x,y)\|p(x)p(y))=\sum_{y\in Y}\sum_{x\in X}p(x,y)\log \frac{p(x,y)}{p(x)p(y)} \ge 0
$$
当且仅当 $p(x,y)=p(x)p(y)$ 时，达到边界0
进一步可以使用联合熵和条件熵来表示
$$
\mathbb{I}(X;Y)=\mathbb{H}(X)-\mathbb{H}(X|Y)=\mathbb{H}(Y)-\mathbb{H}(Y|X)
$$
进一步化简合并可以得到
$$
\mathbb{I}(X;Y)=\mathbb{H}(X) + \mathbb{H}(Y)-\mathbb{H}(X,Y)
$$
然后从韦恩图上我们可以更加明显的理解这几个概念
<div style="width:100%; overflow-x:auto;">
<svg viewBox="0 0 1000 760"
     xmlns="http://www.w3.org/2000/svg"
     style="max-width:1000px; width:100%; height:auto; display:block; margin:auto;">
  <defs>
    <!-- 斜线填充 -->
    <pattern id="hatch"
             patternUnits="userSpaceOnUse"
             width="12"
             height="12"
             patternTransform="rotate(45)">
      <line x1="0" y1="0"
            x2="0" y2="12"
            stroke="currentColor"
            stroke-width="2"
            opacity="0.4"/>
    </pattern>
    <!-- ============================== -->
    <!-- c) 互信息：严格裁剪为 x ∩ y -->
    <!-- ============================== -->
    <!-- 左圆作为裁剪区域 -->
    <clipPath id="clipLeftC">
      <circle cx="180" cy="505" r="120"/>
    </clipPath>
    <!-- ============================== -->
    <!-- d) 条件熵：集合差 mask -->
    <!-- ============================== -->
    <!-- x - y -->
    <mask id="maskXminusY"
          maskUnits="userSpaceOnUse"
          x="500" y="360"
          width="450" height="300">
      <!-- 默认全部保留 -->
      <rect x="500" y="360"
            width="450" height="300"
            fill="white"/>
      <!-- 删除 y 圆区域 -->
      <circle cx="790" cy="505" r="120"
              fill="black"/>
    </mask>
    <!-- y - x -->
    <mask id="maskYminusX"
          maskUnits="userSpaceOnUse"
          x="500" y="360"
          width="450" height="300">
      <!-- 默认全部保留 -->
      <rect x="500" y="360"
            width="450" height="300"
            fill="white"/>
      <!-- 删除 x 圆区域 -->
      <circle cx="650" cy="505" r="120"
              fill="black"/>
    </mask>
  </defs>
  <!-- ================================================= -->
  <!-- a) 熵 -->
  <!-- ================================================= -->
  <g>
    <circle cx="180" cy="180" r="120"
            fill="url(#hatch)"
            stroke="currentColor"
            stroke-width="3"/>
    <circle cx="320" cy="180" r="120"
            fill="url(#hatch)"
            stroke="currentColor"
            stroke-width="3"/>
    <!-- 小写 x、y -->
    <text x="105" y="55"
          font-size="25"
          font-style="italic"
          text-anchor="middle">
      x
    </text>
    <text x="395" y="55"
          font-size="25"
          font-style="italic"
          text-anchor="middle">
      y
    </text>
    <!-- 熵 -->
    <text x="135" y="185"
          font-size="25"
          font-style="italic"
          text-anchor="middle">
      H(x)
    </text>
    <text x="365" y="185"
          font-size="25"
          font-style="italic"
          text-anchor="middle">
      H(y)
    </text>
    <!-- 单向箭头 -->
    <text x="180" y="330"
          font-size="21"
          text-anchor="middle">
      H(x) → x
    </text>
    <text x="320" y="330"
          font-size="21"
          text-anchor="middle">
      H(y) → y
    </text>
    <text x="250" y="365"
          font-size="24"
          font-weight="600"
          text-anchor="middle">
      a) 熵
    </text>
  </g>
  <!-- ================================================= -->
  <!-- b) 联合熵 -->
  <!-- ================================================= -->
  <g>
    <circle cx="650" cy="180" r="120"
            fill="url(#hatch)"
            stroke="currentColor"
            stroke-width="3"/>
    <circle cx="790" cy="180" r="120"
            fill="url(#hatch)"
            stroke="currentColor"
            stroke-width="3"/>
    <text x="575" y="55"
          font-size="25"
          font-style="italic"
          text-anchor="middle">
      x
    </text>
    <text x="865" y="55"
          font-size="25"
          font-style="italic"
          text-anchor="middle">
      y
    </text>
    <text x="720" y="185"
          font-size="27"
          font-style="italic"
          text-anchor="middle">
      H(x,y)
    </text>
    <text x="720" y="330"
          font-size="21"
          text-anchor="middle">
      H(x,y) → x ∪ y
    </text>
    <text x="720" y="365"
          font-size="24"
          font-weight="600"
          text-anchor="middle">
      b) 联合熵
    </text>
  </g>
  <!-- ================================================= -->
  <!-- c) 互信息 -->
  <!-- ================================================= -->
  <g>
    <!-- 两个圆的边界 -->
    <circle cx="180" cy="505" r="120"
            fill="none"
            stroke="currentColor"
            stroke-width="3"/>
    <circle cx="320" cy="505" r="120"
            fill="none"
            stroke="currentColor"
            stroke-width="3"/>
    <!--
      关键：
      画右圆，但是用左圆裁剪。
      最终留下的区域严格等于：
      left circle ∩ right circle
    -->
    <circle cx="320" cy="505" r="120"
            fill="url(#hatch)"
            clip-path="url(#clipLeftC)"/>
    <!-- 再画一次边界，保证边缘清楚 -->
    <circle cx="180" cy="505" r="120"
            fill="none"
            stroke="currentColor"
            stroke-width="3"/>
    <circle cx="320" cy="505" r="120"
            fill="none"
            stroke="currentColor"
            stroke-width="3"/>
    <text x="105" y="380"
          font-size="25"
          font-style="italic"
          text-anchor="middle">
      x
    </text>
    <text x="395" y="380"
          font-size="25"
          font-style="italic"
          text-anchor="middle">
      y
    </text>
    <text x="250" y="510"
          font-size="26"
          font-style="italic"
          text-anchor="middle">
      I(x;y)
    </text>
    <text x="250" y="655"
          font-size="21"
          text-anchor="middle">
      I(x;y) → x ∩ y
    </text>
    <text x="250" y="695"
          font-size="24"
          font-weight="600"
          text-anchor="middle">
      c) 互信息
    </text>
  </g>
  <!-- ================================================= -->
  <!-- d) 条件熵 -->
  <!-- ================================================= -->
  <g>
    <!-- x - y -->
    <circle cx="650" cy="505" r="120"
            fill="url(#hatch)"
            mask="url(#maskXminusY)"/>
    <!-- y - x -->
    <circle cx="790" cy="505" r="120"
            fill="url(#hatch)"
            mask="url(#maskYminusX)"/>
    <!-- 圆边界 -->
    <circle cx="650" cy="505" r="120"
            fill="none"
            stroke="currentColor"
            stroke-width="3"/>
    <circle cx="790" cy="505" r="120"
            fill="none"
            stroke="currentColor"
            stroke-width="3"/>
    <text x="575" y="380"
          font-size="25"
          font-style="italic"
          text-anchor="middle">
      x
    </text>
    <text x="865" y="380"
          font-size="25"
          font-style="italic"
          text-anchor="middle">
      y
    </text>
    <text x="605" y="510"
          font-size="23"
          font-style="italic"
          text-anchor="middle">
      H(x|y)
    </text>
    <text x="835" y="510"
          font-size="23"
          font-style="italic"
          text-anchor="middle">
      H(y|x)
    </text>
    <text x="625" y="655"
          font-size="19"
          text-anchor="middle">
      H(x|y) → x − y
    </text>
    <text x="815" y="655"
          font-size="19"
          text-anchor="middle">
      H(y|x) → y − x
    </text>
    <text x="720" y="695"
          font-size="24"
          font-weight="600"
          text-anchor="middle">
      d) 条件熵
    </text>
  </g>
</svg>
</div>

##### 1.9.1 条件互信息
当然我们还可以定义条件互信息
$$
\begin{align*}
\mathbb{I}(X; Y|Z) &\triangleq \mathbb{E}_{p(z)}\left[\mathbb{I}(X; Y|Z=z)\right]\\
&= \mathbb{E}_{p(x, y, z)} \left[ \log \frac{p(x, y|z)}{p(x|z)p(y|z)} \right]\\
&= \mathbb{H}(X|Z) + \mathbb{H}(Y|Z) - \mathbb{H}(X, Y|Z)\\
&= \mathbb{H}(X|Z) - \mathbb{H}(X|Y, Z) = \mathbb{H}(Y|Z) - \mathbb{H}(Y|X, Z)\\
&= \mathbb{H}(X, Z) + \mathbb{H}(Y, Z) - \mathbb{H}(Z) - \mathbb{H}(X, Y, Z)\\
&= \mathbb{I}(Y; X, Z) - \mathbb{I}(Y; Z)
\end{align*}
$$
上式也可以写为 $\mathbb{I}(Z, Y; X) = \mathbb{I}(Z; X) + \mathbb{I}(Y; X|Z)$，进而有**互信息的链式法则**
$$
\mathbb{I}(Z_1, \cdots, Z_N; X) = \sum_{n=1}^N \mathbb{I}(Z_n; X | Z_1, \cdots, Z_{n-1})
$$
##### 1.9.2 归一化互信息
当分母非零时，我们可以定义一个介于 $0$ 和 $1$ 之间的归一化相关性度量
$$
\text{NMI}(X,Y)=\frac{\mathbb{I}(X;Y)}{\text{min}(\mathbb{H}(X),\mathbb{H}(Y))}
$$
##### 1.9.3 最大信息系数

是一种用于衡量两个变量之间相关性的统计量，可以用于发现线性关系以及多种非线性关系。

MIC 的基本思想是：**将二维样本空间划分成不同大小的网格，在各种网格划分中寻找能够产生最大互信息的划分，并对互信息进行归一化。**

对于数据集 $D$，给定一个 $x\times y$ 的网格，定义

$$  
M(D)_{x,y}=\frac{\max_{G\in\mathcal{G}_{x,y}} I(D|G)}{\log \min{\{x,y\}}}
$$

其中，$\mathcal{G}_{x,y}$ 表示所有可能的 $x\times y$ 网格划分，$I(D|G)$ 表示数据 $D$ 在网格 $G$ 下离散化后得到的互信息。

分母

$$  
\log \min{\{x,y\}}
$$
用于对互信息进行归一化，因为对于一个 $x\times y$ 的网格，
$$  
I(X;Y) \le \min{H(X),H(Y)}\le \log \min\{{x,y\}}  
$$

因此 $M(D)_{x,y}$ 可以理解为：**当前网格能够捕获的信息量占该网格理论最大信息量的比例。**
最后，在所有满足复杂度约束的网格中取最大值：
$$  
\operatorname{MIC}(D)=\max_{xy<B(n)}M(D)_{x,y}.  
$$
其中 $n$ 为样本数量，$B(n)$ 用于限制网格复杂度，防止网格划分过细而产生过拟合。
因此，MIC 可以简单理解为**不同网格划分下的最大归一化互信息**

MIC 的取值通常位于 $0$ 到 $1$ 之间。MIC 越接近 $1$，说明两个变量之间存在越强的依赖关系；越接近 $0$，说明能够检测到的依赖关系越弱。
##### 1.9.4 数据处理不等式
假设有一个位置变量 $X$，我们观察到该未知变量的一个噪声函数，称之为 $Y$ 。如果我们以某种方式处理有噪声的观测结果，以创建一个新的变量 $Z$ ，那么显然，我们无法增加关于未知变量 $X$ 的信息量，这个就被成为数据处理不等式
形式化表达：
假设 $X\rightarrow Y\rightarrow Z$ 形成一个马尔可夫链，那么在给定 $Y$ 的条件下 $X$ 、$Z$ 相互独立，则$\mathbb{I}(X;Y)\ge\mathbb{I}(X;Z)$ 。
证明：由于互信息的链式法则，我们可以使用两种不同的方式扩展互信息
$$
\mathbb{I}(X;Y,Z)=\mathbb{I}(X;Z)+\mathbb{I}(X;Y|Z)=\mathbb{I}(X;Y)+\mathbb{I}(X;Z|Y)
$$
由于在给定 $Y$ 的条件下 $X$ 、$Z$ 相互独立，因此 $\mathbb{I}(X;Z|Y)=0$，于是 
$$
\mathbb{I}(X;Z)+\mathbb{I}(X;Y|Z)=\mathbb{I}(X;Y)
$$
由于 $\mathbb{I}(X;Y|Z)\ge0$ ,因此 $\mathbb{I}(X;Y)\ge\mathbb{I}(X;Z)$，同理可以证明 $\mathbb{I}(Y;Z)\ge\mathbb{I}(X;Z)$。

##### 1.9.5 充分统计量
从数据处理不等式中我们可以得到重要结论，假设 $\theta\rightarrow\mathcal{D}\rightarrow s(\mathcal{D})$，于是 
$$
\mathbb{I}(\theta;s(\mathcal{D}))\le\mathbb{I}(\theta;\mathcal{D})
$$
若等式成立，那么我们称 $s(\mathcal{D})$ 是数据 $\mathcal{D}$ 用于推断 $\theta$ 的充分统计量，一个例子，$s(\mathcal{D}) =\mathcal{D}$，就是数据本身。
进而我们可以定义一个**最小充分统计量**：如果对于所有充分统计量 $s'(\mathcal{D})$ ，存在某个函数 $f$ ，使得 $s(\mathcal{D})=f(s'(\mathcal{D}))$ ，那么我们称 s 是 D 的最小充分统计量