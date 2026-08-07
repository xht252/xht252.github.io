---
title: "RT-2 精读：从 Vision-Language Model 到 Vision-Language-Action"
date: 2026-08-07
summary: "快速读懂 RT-2：如何把机器人动作表示成语言 Token，将 PaLI-X / PaLM-E 从视觉语言模型转化为可以直接闭环控制机器人的 Vision-Language-Action 模型，以及为什么这一步成为后来 VLA 路线的重要起点。"
tags:
  - Robotics
  - Embodied AI
  - VLA
  - Vision-Language Model
  - RT-2
authors:
  - me
featured: false
draft: false
---

> **Paper:** Anthony Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*  
> **arXiv:** [2307.15818](https://arxiv.org/abs/2307.15818)  
> **PDF:** [arXiv PDF](https://arxiv.org/pdf/2307.15818)  
> **Project:** [RT-2 Project Page](https://robotics-transformer2.github.io/)  
> **Blog:** [Google DeepMind: RT-2](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/)

如果 RT-1 解决的问题是：

> **能不能用一个统一的 Transformer，从大规模、多任务机器人数据中直接学习机器人策略？**

那么 RT-2 继续追问：

> **能不能把互联网规模的视觉—语言知识，直接迁移到机器人动作空间？**

RT-2 给出的答案非常直接：

> **把机器人动作也表示成 Token，然后像训练语言模型一样，让 Vision-Language Model 同时学习“回答问题”和“输出机器人动作”。**

这就是论文中提出的 **Vision-Language-Action Model，VLA**。

![RT-2 Overview](https://robotics-transformer2.github.io/img/fig1.png)

*RT-2 的核心思路：将互联网视觉语言数据与机器人动作数据共同训练，使一个预训练 VLM 直接变成可以闭环控制机器人的 VLA。图片来自 RT-2 官方项目页。*

---

## 1. 从 RT-1 到 RT-2，真正发生了什么变化？

RT-1 已经证明，大规模多任务机器人数据可以训练出一个具有一定泛化能力的统一策略。

但是 RT-1 有一个很明显的边界：

> **它知道的世界，基本仍然来自机器人自己见过的数据。**

即使收集了 13 台机器人、17 个月、约 13 万条示范，这个规模与互联网数据相比仍然非常小。

机器人很难通过真实交互穷举：

```text
所有物体
所有环境
所有语言表达
所有语义关系
所有常识
```

而大型 Vision-Language Model 已经在互联网规模的数据中学到了大量这些知识。

所以 RT-2 的关键问题变成：

```text
Internet-scale Vision-Language Knowledge
                  +
           Robot Experience
                  ↓
      Can they become one policy?
```

RT-2 不再从头设计一个机器人网络，而是直接从已有的大型 VLM 出发。

论文主要使用两种预训练模型：

- **PaLI-X**：RT-2-PaLI-X，实验包含 5B 和 55B 参数版本；
- **PaLM-E**：RT-2-PaLM-E，主要使用 12B 参数版本。

于是 RT-1 与 RT-2 的关系可以非常简单地理解成：

| | RT-1 | RT-2 |
| --- | --- | --- |
| 主要知识来源 | Robot Data | Web Data + Robot Data |
| 模型 | Robot Transformer | Pre-trained VLM |
| 输入 | Image + Language | Image + Language |
| 输出 | Action Tokens | **Text-form Action Tokens** |
| 重点 | 多任务机器人策略 | **把 Web Knowledge 迁移到机器人控制** |
| 核心概念 | Robotics Transformer | **Vision-Language-Action Model** |

---

## 2. RT-2 最关键的一步：把 Action 当成 Language

理解 RT-2，其实只需要真正理解这一件事。

传统 Vision-Language Model 做的是：

```text
Image + Text
     ↓
    VLM
     ↓
Text Tokens
```

例如：

```text
Q: What is in the image?
A: A red apple.
```

机器人策略则需要：

```text
Image + Instruction
        ↓
      Policy
        ↓
Robot Action
```

问题就在于：

> VLM 的输出空间是 **Language Token**，而机器人需要的是连续控制量。

RT-2 没有为此增加一个复杂的新 Action Head。

它选择了一个非常简单的方案：

> **直接把机器人动作编码成文本 Token。**

![RT-2 Action Representation](https://robotics-transformer2.github.io/img/fig3.png)

*RT-2 的动作表示。一个机器人动作由终止标志、三维位置变化、三维旋转变化和夹爪状态组成。图片来自 RT-2 官方项目页。*

RT-2 延续 RT-1 的动作空间：

```text
Terminate / Continue
Δ Position X
Δ Position Y
Δ Position Z
Δ Rotation X
Δ Rotation Y
Δ Rotation Z
Gripper
```

连续动作维度仍然被均匀离散为 **256 bins**。

于是一个动作就可以写成类似：

```text
1 128 91 241 5 101 127 217
```

这样的 Token 序列。

这时机器人控制问题就被重新写成了：

```text
Q: What action should the robot take to <task>?
A: 1 128 91 241 5 101 127 217
```

于是对于模型来说：

```text
"The object is an apple."
```

和

```text
"1 128 91 241 5 101 127 217"
```

本质上都只是：

> **预测下一个 Token。**

这一步看起来很简单，但非常关键。

因为一旦 Action 进入和 Language 相同的 Token Space，原本用于视觉问答、图像理解和语言推理的模型，就不需要重新设计整个网络结构，而可以直接被训练成机器人策略。

这也是 **Vision-Language-Action** 这个概念最核心的含义。

---

## 3. Co-Fine-Tuning：不要为了学动作，把原来的知识忘掉

如果直接拿一个 VLM，只使用机器人数据进行 Fine-tuning，会出现一个问题：

> 模型可能逐渐忘掉预训练阶段从 Web 学到的视觉和语言知识。

而 RT-2 的目的恰恰就是利用这些知识。

因此论文采用 **Co-Fine-Tuning**：

```text
Internet Vision-Language Data
             +
       Robot Action Data
             ↓
       Co-Fine-Tuning
             ↓
            RT-2
```

训练过程中同时保留两类任务。

### 普通 VLM 数据

例如：

```text
Image + Question
       ↓
Natural Language Answer
```

### Robot 数据

例如：

```text
Robot Image + Instruction
          ↓
      Action Tokens
```

于是同一个模型同时学习：

```text
Vision → Language
Language → Semantic Reasoning
Vision + Language → Action
```

这也是 RT-2 与“拿一个视觉预训练 Encoder 接到机器人策略前面”最大的不同。

RT-2 希望保留的是整个 VLM 中已经形成的 **视觉—语言语义能力**，而不只是一个视觉特征提取器。

论文的消融实验也很清楚：

| RT-2-PaLI-X | Size | Training | Unseen Average |
| --- | ---: | --- | ---: |
| From scratch | 5B | Robot data | 9% |
| Fine-tuning | 5B | Robot data | 42% |
| **Co-Fine-Tuning** | **5B** | **Web + Robot** | **44%** |
| Fine-tuning | 55B | Robot data | 52% |
| **Co-Fine-Tuning** | **55B** | **Web + Robot** | **63%** |

最值得记住的不是某一个具体数字，而是这个趋势：

```text
Pre-training matters
        +
Co-Fine-Tuning matters
        +
Model Scale matters
```

RT-2 的泛化能力并不是单纯来自“机器人数据更多”，而是在尝试利用 **Foundation Model 已经学到的世界知识**。

---

## 4. 实验到底证明了什么？

RT-2 做了超过 **6000 次真实机器人 evaluation trials**。

但快速精读时不需要看完所有实验，只需要关注两个问题：

### 4.1 对普通机器人任务，RT-2 有没有变差？

没有。

在 Seen Tasks 上：

| Model | Seen Tasks |
| --- | ---: |
| RT-1 | 92% |
| RT-2-PaLI-X-55B | 91% |
| RT-2-PaLM-E-12B | 93% |

也就是说，引入大型 VLM 后，RT-2 仍然保持了与 RT-1 相近的基本机器人控制能力。

真正的区别出现在 **Unseen** 场景。

### 4.2 Web Knowledge 能不能真的迁移到机器人？

论文测试了：

```text
Unseen Objects
Unseen Backgrounds
Unseen Environments
```

最终的平均成功率：

| Model | Unseen Average |
| --- | ---: |
| R3M | 12% |
| VC-1 | 10% |
| RT-1 | 32% |
| MOO | 35% |
| **RT-2-PaLI-X-55B** | **62%** |
| **RT-2-PaLM-E-12B** | **62%** |

这里可以看到 RT-2 最直接的效果：

> **Seen Task 基本没有损失，但 Unseen Average 从 RT-1 的 32% 提升到了 62%。**

因此 RT-2 的价值不是“把机器人熟悉任务做得更准”，而是：

> **让机器人能够利用 VLM 中已经存在的视觉和语义知识，在从未见过的物体、背景和环境中做出更合理的动作。**

---

## 5. 什么叫 RT-2 的 Emergent Capability？

这是这篇论文最有意思的实验。

研究者故意给机器人一些 **机器人训练数据中从未出现过的语义任务**。

例如：

```text
move apple to 3
```

模型需要先理解数字符号，再把这种理解转化成动作。

或者：

```text
move banana to the sum of two plus one
```

模型不仅要看到 banana，还要理解：

```text
2 + 1 = 3
```

然后把 banana 移动到数字 3。

还有：

```text
pick up the bag about to fall off the table
```

这里不是简单识别 `bag`，而需要理解场景中的空间状态：

> 哪个包处于“快掉下桌子”的状态？

论文把这些能力大致分为：

- **Symbol Understanding**
- **Reasoning**
- **Person Recognition**

结果中：

| Model | Emergent Task Average |
| --- | ---: |
| VC-1 | 11% |
| RT-1 | 17% |
| RT-2-PaLM-E-12B | 40% |
| **RT-2-PaLI-X-55B** | **60%** |

RT-2-PaLI-X-55B 的平均成功率达到 RT-1 的三倍以上。

但这里有一个非常重要的边界。

RT-2 论文自己也强调：

> **Web pre-training 并没有让机器人凭空学会新的运动技能。**

模型真正迁移的是：

```text
semantic concepts
visual concepts
relations
reasoning
world knowledge
```

机器人的低层运动能力仍然主要来自 Robot Data。

所以 RT-2 的“Emergent Skill”更准确地说是：

> **把已有的机器人运动技能，用在以前机器人数据没有覆盖过的语义条件上。**

这一区别非常重要。

---

## 6. Chain-of-Thought 也可以进入机器人控制

RT-2 还做了一个很有意思的探索。

普通 RT-2 是：

```text
Instruction
    ↓
Action
```

研究者又尝试加入一个 `Plan`：

```text
Instruction
    ↓
Plan
    ↓
Action
```

例如：

```text
Instruction:
I need to hammer a nail,
what object from the scene might be useful?

Plan:
rock

Action:
1 129 138 ...
```

![RT-2 Chain-of-Thought](https://robotics-transformer2.github.io/img/CoT5.png)

*RT-2 将自然语言 Plan 与 Action Token 放在同一个生成序列中，展示了把语义推理和低层机器人控制统一在一个模型中的可能性。图片来自 RT-2 官方项目页。*

这个实验并不是 RT-2 最成熟的部分，但它展示了一件很重要的事情：

> 当 Language 和 Action 都被表示成 Token 后，**Reasoning → Planning → Action** 可以开始出现在同一个自回归模型中。

这与以前的 SayCan 路线很不一样。

SayCan 更接近：

```text
LLM
 ↓
High-level skill selection
 ↓
Separate low-level policy
```

而 RT-2 尝试的是：

```text
Vision + Language
       ↓
      VLA
       ↓
Reasoning Tokens
       ↓
 Action Tokens
```

也就是把 **高层语义推理和低层控制逐渐统一到一个模型内部**。

---

## 7. RT-2 真正重要在哪里？

如果只看网络结构，RT-2 甚至会显得有点“简单”。

它没有提出特别复杂的新 Transformer Block。

最核心的操作实际上就是：

```text
Robot Action
    ↓
Discrete Tokens
    ↓
Use the existing VLM vocabulary
    ↓
Co-Fine-Tune Web + Robot Data
```

但正是这个简单的接口，把两个以前相对分离的世界连接起来：

```text
Foundation Models
Vision
Language
World Knowledge
Reasoning
        │
        │ same token interface
        ↓
Robot Action
Physical Interaction
Closed-loop Control
```

所以我认为 RT-2 最重要的贡献不是一个具体的网络，而是一种 **范式转换**：

> **机器人策略不一定要从头学习整个世界。可以先让 Foundation Model 从互联网中学习视觉、语言与常识，再通过少量机器人数据把这些知识 Ground 到 Action Space。**

这也是为什么 RT-2 之后，**VLA** 很快成为具身智能中最重要的路线之一。

后面的很多工作，本质上都在继续回答几个问题：

```text
如何获得更多 Robot Data？
如何支持更多 Robot Embodiments？
如何让 Action Representation 更统一？
如何加入更强的 Spatial / Temporal Reasoning？
如何让 VLA 具备更长时序的 Planning？
如何把 World Model 与 VLA 结合？
```

---

## 8. RT-2 也有非常明显的局限

RT-2 并没有解决通用机器人的所有问题。

### 第一，模型非常大

最大的 RT-2-PaLI-X 有 **55B 参数**。

论文需要把模型运行在多 TPU 的 Cloud Service 上，再通过网络给机器人提供推理结果。

55B 版本实际控制频率大约只有：

```text
1–3 Hz
```

而 5B 版本约为：

```text
5 Hz
```

这与后来追求更高频率、更低延迟的 VLA 形成了一个非常现实的矛盾：

> **Foundation Model 越强，实时机器人控制往往越困难。**

### 第二，Physical Skill 仍然受 Robot Data 限制

RT-2 能理解：

```text
rock can be used as a hammer
```

并不意味着它会突然学会一套完全没有示范过的复杂 hammering motion。

VLM 提供的主要是：

```text
What?
Which?
Where?
Why?
```

而真正的：

```text
How to physically execute it?
```

仍然高度依赖机器人轨迹数据。

这也是后来 VLA 研究仍然需要大规模 Robot Dataset 的原因。

---

## 最后，用一句话理解 RT-2

如果 RT-1 的核心是：

> **让 Transformer 从大量机器人经验中学习一个统一的多任务策略。**

那么 RT-2 的核心就是：

> **把机器人动作也变成一种语言，使互联网规模视觉—语言预训练得到的语义知识和推理能力，可以直接进入端到端机器人控制。**

所以从研究发展的角度看：

```text
RT-1
Robot Data
   ↓
Generalist Robot Policy

        ↓

RT-2
Web Vision-Language Data
          +
      Robot Data
          ↓
Vision-Language-Action

        ↓

RT-X / Open X-Embodiment
More Robots
More Embodiments
More Robot Data

        ↓

Modern VLA
General-purpose Embodied Foundation Models
```

RT-1 让我们看到机器人策略也可以开始 **scale with data**。

RT-2 则进一步证明：

> **机器人真正需要扩大的，不只是 Robot Data，而是能够被 Ground 到 Action 上的整个 Foundation Model Knowledge。**

这也是我认为 RT-2 比 RT-1 更重要的一步：它基本确定了今天很多具身智能系统仍然在沿用的 **Vision + Language → Action** 主线。
