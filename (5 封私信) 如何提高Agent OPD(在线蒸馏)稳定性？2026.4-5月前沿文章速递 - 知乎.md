# (5 封私信) 如何提高Agent OPD(在线蒸馏)稳定性？2026.4-5月前沿文章速递 - 知乎
**On Policy Distillation (OPD，在线蒸馏)**最近在大模型后训练中炙手可热，在[DeepSeek V4](https://zhida.zhihu.com/search?content_id=274635250&content_type=Article&match_order=1&q=DeepSeek+V4&zhida_source=entity)的后训练中也大范围应用了此技术。这里就不多赘述技术细节和优势了，不太了解的朋友们可以把它当作是[SFT](https://zhida.zhihu.com/search?content_id=274635250&content_type=Article&match_order=1&q=SFT&zhida_source=entity)和RL之间的中间步骤，相较于SFT能够提供更“符合student体质”的奖励信号，且比RL的各种Outcome Reward更加稠密，是token级别的信号。

OPD在纯文字（推理、数学等等）的各种任务中表现很好，但是**如果涉及到Agent场景，有工具调用，那训练就会变得非常不稳定**。正确率在低位徘徊，在某些情况甚至越训练正确率越低（简称训崩了）。

涉及多轮对话的Agent任务时，传统的OPD面临的问题：

**误差累积**：在多轮工具调用中，错误会沿着轨迹放大，这种放大是突然的（比如调用了错误的工具），教师模型很难拉回来，最终将学生模型推入一个教师模型从未见过的状态空间，导致教师的监督信号彻底失效。

本文聚焦于这个问题，给大家提供两篇产业界的观察和解决方案，非常的新，都是两周内发布的。

一、[阿里通义TCOD](https://zhida.zhihu.com/search?content_id=274635250&content_type=Article&match_order=1&q=%E9%98%BF%E9%87%8C%E9%80%9A%E4%B9%89TCOD&zhida_source=entity)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------

首先是**阿里通义**团队的TCOD: Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents

[https://arxiv.org/abs/2604.24005](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2604.24005)

![](https://pica.zhimg.com/v2-19df2bb32e2589402811923947a6f454_1440w.jpg)

文章在进行Agent OPD时观察到两个现象：

1.  **训练过程中KL散度会上升，正确率下降（对应下图a、b）**
2.  **即使最终KL散度收敛了，但初始值过高，难以接受（对应下图c）**

（上面的KL都是**反向KL**，student的策略在分母上：）

![](https://picx.zhimg.com/v2-9201b236da39932bf9eb24937e788d39_1440w.jpg)

这两个现象的背后机制都是误差随轨迹累计放大导致的（上图d）。

### **本文方法：采用课程学习的方法，让学生模型从短到长逐步产生轨迹，而不是一口气生成整条轨迹。这样能逐步提升学生能力，避免长轨迹导致的错误累计。** 

![](https://pic2.zhimg.com/v2-57e66b4e10ca503fd02039646cd4e4bd_1440w.jpg)

文章提供了两种方法变体，对OPD原始代码的更改都不多：

**F2B**（前向到后向）：先让学生负责前几步，再逐步接管后续步骤（上图中）

**B2F**（后向到前向）：先让老师引导到接近终点的状态，学生只负责最后几步，再逐渐向前延伸（上图右）

**实验结果：** F2B和B2F都缓解了正确率下降，增加了KL稳定性，在某些情况也能让模型性能提升

![](https://pic2.zhimg.com/v2-4abf114fae5977a44c82150fbcb91177_1440w.jpg)

二、[腾讯混元SOD](https://zhida.zhihu.com/search?content_id=274635250&content_type=Article&match_order=1&q=%E8%85%BE%E8%AE%AF%E6%B7%B7%E5%85%83SOD&zhida_source=entity)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------

**腾讯混元**团队也在5月8号推出了SOD: Step-wise On-policy Distillation for Small Language Model Agents。

[https://arxiv.org/abs/2605.07725](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2605.07725)

腾讯在把OPD应用于Agent这种Tool use的场景时发现训练非常不稳定，这是因为在工具调用场景下轨迹是不连续的，如果学生模型选了错误的工具，有了错误的Observation，就会迅速扩大和教师的差距。而**在分布偏差较大时，老师token级别的信号就不准确了，甚至可能放大差异，导致越训越崩**。

![](https://pic2.zhimg.com/v2-72e9891eb82269102da8a95717a822cd_1440w.jpg)

### **本文方法：设计了一个step-level的指标来衡量学生和老师之间的差异，然后根据差异来调整“蒸馏强度”（就是某个step的权重）。定性来说是在差异小的地方提高权重，在差异大的部分减少权重，防止越训越崩**

![](https://pica.zhimg.com/v2-a336b68fbd56b7002989767265bd4860_1440w.jpg)

用下面的公式来衡量第k个step**学生和老师之间的分布差异**：

![](https://picx.zhimg.com/v2-8edd1d2ce66ff3c8319448d268a819c5_1440w.jpg)

然后用下面的公式算出来每个step的**蒸馏权重**（设第一步的蒸馏强度是最大的，设为1）：

![](https://pic4.zhimg.com/v2-d6c34aa37353aa3a8a39c1a030a14613_1440w.jpg)

最终的loss也不止OPD Loss，还有**GRPO的结果奖励**来做控制：

![](https://pic3.zhimg.com/v2-1b0cbeff067946399c2283c686b852e0_1440w.jpg)

实验结果：

![](https://pic4.zhimg.com/v2-d6b002424e66dab9a300836eba999fe1_1440w.jpg)