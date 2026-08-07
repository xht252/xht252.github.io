---
title: "RT-1 精读：Transformer 如何学会控制真实机器人"
date: 2026-08-07
summary: "用一篇短文快速读懂 RT-1：大规模真实机器人数据、FiLM-EfficientNet、TokenLearner、Transformer 与 Action Tokenization，以及它为什么成为后来 VLA 路线的重要起点。"
tags:
  - Robotics
  - Embodied AI
  - Transformer
  - RT-1
authors:
  - me
featured: false
draft: false
---

> **Paper:** Anthony Brohan et al., *RT-1: Robotics Transformer for Real-World Control at Scale*  
> **arXiv:** [2212.06817](https://arxiv.org/abs/2212.06817)  
> **PDF:** [arXiv PDF](https://arxiv.org/pdf/2212.06817)  
> **Project:** [RT-1 Project Page](https://robotics-transformer1.github.io/)  
> **Code:** [google-research/robotics_transformer](https://github.com/google-research/robotics_transformer)

RT-1 是一篇非常适合作为 **Robot Foundation Model / Vision-Language-Action（VLA）** 入门的论文。

如果只用一句话概括：

> **RT-1 用大规模、多任务的真实机器人示范数据训练一个语言条件 Transformer，让同一个策略直接从图像和自然语言指令预测机器人动作。**

它真正重要的地方并不是“Transformer 也可以控制机器人”，而是开始验证一个更大的问题：

> **机器人策略能不能像大语言模型一样，通过更大的数据规模、更高的任务多样性和足够强的模型容量，获得更好的泛化能力？**

![RT-1 Overview](https://robotics-transformer1.github.io/img/rt1_teaser.png)

*RT-1 的整体目标：从大规模真实机器人数据中学习一个可以泛化到新任务、新物体和新场景的统一策略。图片来自 RT-1 官方项目页。*

---

## 1. RT-1 到底在做什么？

RT-1 学习的是一个端到端机器人策略。

它的输入只有两类信息：

- 最近 **6 帧 RGB 图像**；
- 一条自然语言指令，例如 `pick apple`、`open the drawer`。

输出则是机器人当前时刻应该执行的动作：机械臂位姿变化、夹爪开合、移动底盘动作，以及结束任务的指令。

可以把整个过程压缩成：

```text
Language Instruction + Image History
                 ↓
        FiLM-EfficientNet
                 ↓
          TokenLearner
                 ↓
           Transformer
                 ↓
          Action Tokens
                 ↓
               Robot
```

RT-1 没有显式地图，没有单独的路径规划器，也不像 Dreamer 一样先学习一个世界模型再进行 imagined rollout。

它做的事情很直接：

> **根据当前看到的图像和任务指令，直接预测下一步动作。**

---

## 2. RT-1 最核心的东西其实是数据

RT-1 使用 **13 台真实机器人**，持续约 **17 个月**收集数据，最终得到约 **13 万条真实机器人 episode**，覆盖 **700+ 个任务**。

这些任务包括：

```text
pick object
place object
open / close drawer
move object near another object
put object into drawer
knock object over
...
```

这里最值得关注的不是“700”这个数字本身，而是 **任务之间存在大量可复用的结构**。

例如训练数据里可能出现过：

```text
pick coke can
pick apple
move apple near bottle
```

模型在训练过程中会反复看到：

```text
skill   +   object   +   scene
```

这些因素的不同组合。

因此测试时，即使某条完整指令没有在训练集中出现过，模型仍然有机会把已经学过的 skill 和 object 重新组合起来。

这也是 RT-1 所强调的一个重要能力：**compositional generalization**。

换句话说，RT-1 并不是希望机器人死记 700 个任务，而是希望一个统一策略能够从大量任务中学习到可以复用的行为结构。

---

## 3. 模型结构：其实只需要理解四步

RT-1 约有 **35M 参数**，同时还需要在真实机器人上以大约 **3 Hz** 的频率进行闭环控制。

因此它的模型设计并不是单纯“堆一个更大的 Transformer”，而是在 **模型容量和实时推理速度之间做平衡**。

完整结构如下：

![RT-1 Architecture](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEj11ho9tm4Td7ByTigAgDxFWsxbsZ6tQsAng3AtwuufHRuoLaLOV9YN7FUMyyAhefzuFOVCrbwTLsEaRYidOOToS__KRrotot-6aBxTliZxYz-B2jiJG-4myq5NB3vRKaY86nr5y1-13dBv_H_XyfnDijphCM3UBalczim0PeGJ63Z0Ok6k9zvKQ2D55A/s16000/image1.png)

*RT-1 的模型结构：语言指令通过 USE 编码，并通过 FiLM 调制 EfficientNet；视觉特征经 TokenLearner 压缩后送入 Transformer，最终输出离散动作 token。图片来自 Google Research 官方博客。*

### 3.1 Language → USE

自然语言指令首先经过预训练的 **Universal Sentence Encoder（USE）**，得到一个语言 embedding。

RT-1 并没有依靠机器人数据从零学习语言，而是直接利用已经训练好的语言语义表示。

### 3.2 Image → FiLM-EfficientNet

每帧 RGB 图像经过 ImageNet 预训练的 **EfficientNet-B3**。

这里真正关键的是 **FiLM**。

语言 embedding 会直接调制 EfficientNet 中的视觉特征，因此视觉编码器从比较早的阶段就能够根据当前任务选择需要关注的信息。

例如：

```text
Instruction: pick apple
                ↓
Language Embedding
                ↓
        FiLM modulation
                ↓
Vision features focus on task-relevant information
```

因此 RT-1 并不是“先看图，再在最后加入语言”，而是一种比较明显的 **early language fusion**。

### 3.3 TokenLearner：81 → 8

EfficientNet 最终得到一个 $9\times9\times512$ 的空间特征图。

展平以后，每一帧对应：

$$
9\times9=81
$$

个视觉 token。

RT-1 一次输入 6 帧，如果全部送入 Transformer，就会产生：

$$
6\times81=486
$$

个 token。

为了满足机器人实时控制要求，RT-1 使用 **TokenLearner**，把每帧的 81 个视觉 token 压缩到 **8 个 token**：

$$
81\rightarrow8.
$$

于是 6 帧最终只剩：

$$
6\times8=48
$$

个视觉 token。

这一步非常值得记住。

TokenLearner 并不是简单 resize 或 pooling，而是学习当前任务中哪些视觉信息更值得保留。它让 RT-1 可以在保留较强视觉表达能力的同时，大幅降低 Transformer 的计算量。

### 3.4 Transformer → Action

最后，48 个视觉—语言 token 被送入一个 **8 层 decoder-only Transformer**。

Transformer 不预测文本，而是预测机器人动作。

所以从结构上看，RT-1 实际上把机器人控制重新表述成了一个序列建模问题：

```text
Visual / Language Tokens
          ↓
      Transformer
          ↓
      Action Tokens
```

这也是 RT-1 这个名字中 **Robotics Transformer** 的核心。

---

## 4. 为什么要把机器人动作也变成 Token？

这是 RT-1 中我认为最值得注意的设计之一。

机器人机械臂和移动底盘原本都是连续控制量，但 RT-1 没有直接做连续值回归，而是把每一个连续动作维度划分为 **256 个 bins**。

也就是说：

```text
continuous action
       ↓
256 discrete bins
       ↓
 categorical prediction
       ↓
   action token
```

为什么这么做？

一个很重要的原因是机器人示范数据通常具有明显的 **multi-modality**。

例如抓取同一个物体：

- 可以从左边接近；
- 可以从右边接近；
- 也可能采用不同的末端执行器姿态。

这些动作都可能是合理的。

如果只用一个简单的 Gaussian 去拟合连续动作，模型可能会预测多个合理动作之间的“平均值”，而这个平均动作反而可能不合理。

离散的 categorical distribution 则能够更自然地表达多个可能的动作模式。

论文消融也表明，换成连续动作输出后性能明显下降。

所以 RT-1 给出了一个很重要的经验：

> **Action representation 不是一个无关紧要的实现细节，它会直接影响多任务机器人策略能够表达多复杂的行为分布。**

---

## 5. 实验结果：重点看泛化，而不是 Seen Task

论文在真实机器人上进行了超过 **3000 次 rollout**，主要测试四种情况：

- **Seen Tasks**：训练中出现过的任务；
- **Unseen Tasks**：没有完整出现过的新任务组合；
- **Distractors**：场景中加入额外干扰物；
- **Backgrounds**：改变背景和环境。

| Model | Seen | Unseen | Distractors | Backgrounds |
| --- | ---: | ---: | ---: | ---: |
| Gato | 65% | 52% | 43% | 35% |
| BC-Z | 72% | 19% | 47% | 41% |
| BC-Z XL | 56% | 43% | 23% | 35% |
| **RT-1** | **97%** | **76%** | **83%** | **59%** |

论文给出的主要对比结果如下图所示，图片同样直接引用 RT-1 官方项目页：

![RT-1 Main Results](https://robotics-transformer1.github.io/img/main_baselines.png)

*RT-1 与 Gato、BC-Z 和 BC-Z XL 在 Seen Tasks、Unseen Tasks、Distractors 与 Backgrounds 四类场景中的成功率对比。图片来自 RT-1 官方项目页。*

我认为真正值得看的并不是 **Seen = 97%**，而是另外三个指标。

### Unseen Tasks：76%

这说明 RT-1 确实可以把训练阶段学习到的技能、物体和任务结构重新组合，而不是只记住训练指令。

### Distractors：83%

说明场景中出现额外无关物体以后，策略仍然具有较好的鲁棒性。

### Backgrounds：59%

这个结果一方面优于基线，另一方面也暴露了 RT-1 的局限：

> **视觉环境发生较大变化以后，机器人策略仍然很容易受到 distribution shift 的影响。**

因此 RT-1 的“泛化”并不是今天大模型语境下的完全开放世界泛化，而更接近于：

> **在大规模但仍有限的机器人数据分布附近，实现更好的组合泛化和场景鲁棒性。**

---

## 6. RT-1 真正重要在哪里？

如果只记 RT-1 的 EfficientNet、TokenLearner 或 Transformer 层数，很容易错过这篇论文真正重要的地方。

我认为 RT-1 最值得带走的是下面这条逻辑：

```text
More Diverse Robot Data
          ↓
One Shared Multi-task Policy
          ↓
Knowledge Sharing Across Tasks
          ↓
Better Generalization
          ↓
Absorb Even More Data Sources
```

论文后面还进一步测试了 simulation data、另一种 Kuka 机器人数据以及与 SayCan 的组合。

这些实验虽然不是理解 RT-1 架构所必需的，但共同说明了一个重要现象：

> **一个足够强的统一策略，有可能从不同任务、不同数据源，甚至不同机器人 embodiment 中吸收可迁移的知识。**

这也解释了为什么 RT-1 后面的研究会逐渐走向：

```text
RT-1
真实机器人多任务数据
        ↓
统一 Vision + Language → Action 策略

RT-2
互联网规模 Vision-Language 知识
        +
机器人动作数据
        ↓
Vision-Language-Action Model

RT-X / Open X-Embodiment
更多机器人
更多 embodiment
更多任务
更多机构的数据
```

所以 RT-1 可以被看成后来 **VLA / Generalist Robot Policy** 路线的一个重要起点。

---

## 最后，用一句话重新理解 RT-1

读完整篇论文以后，我会把 RT-1 概括为：

> **RT-1 的核心不是提出一个特别复杂的机器人网络，而是证明：当真实机器人数据的规模和多样性足够大，并用一个能够有效吸收这些数据的统一 Transformer 策略进行训练时，机器人控制也开始出现类似基础模型的 scaling 与泛化趋势。**

如果继续沿着这条路线读下去，下一篇最自然的就是 **RT-2**：RT-1 解决的是“如何从大量机器人经验中学习统一策略”，而 RT-2 则进一步追问——**能不能把互联网规模的视觉—语言知识直接带进机器人动作空间？**
