# AugBlur-ISTD
### 摘要
<font style="color:#000000;">针对红外小目标检测中面临</font>**<font style="color:#000000;">测试时跨域分布偏移</font>**<font style="color:#000000;">（</font>_<font style="color:#000000;">Test-time Cross-domain Distribution Shift</font>_<font style="color:#000000;">）和</font>**<font style="color:#000000;">测试时异方差噪声扰动</font>**<font style="color:#000000;">（</font>_<font style="color:#000000;">Test-time Heteroscedastic Noise Perturbations</font>_<font style="color:#000000;">）的双重挑战。我们提出一种两阶段优化框架，首先，通过</font><font style="color:#000000;">Test-time Driven Fusion</font><font style="color:#000000;">生成与测试环境对齐的训练样本，利用小波多频带滤波精准分离测试背景与双指标筛选训练目标，并基于Re-Generation策略实现背景-目标解耦增强；其次，设计</font><font style="color:#000000;">Noise-guided Test-time Adaption</font><font style="color:#000000;">从测试图像中提取真实噪声特征构建动态噪声库，通过混合训练与自信息最小化损失约束模型学习噪声不变性，突破传统人工噪声建模的分布偏差限制。最后，我们构建包含多场景的适应测试环境数据集AugBlur-ISTD，覆盖校园、停车场等复杂环境。实验表明，本文方法在检测精度(PixAcc)、检测概率（Pd）、误报率（Fa）和交并集（IoU）及F1上均优于其他先进的方法。本文的代码可以在https://github.com/xxx上找到。</font>

<font style="color:#000000;">关键词：红外小目标检测（ISTD）、TTA、自信息、域偏移</font>

## Introduction
<font style="color:#000000;">红外小目标检测（ISTD）在无人机监控、军事侦察等领域具有重要应用价值，但复杂环境下的模型泛化能力仍面临严峻挑战。然而，在复杂测试环境下保持稳健的检测性能仍然是一个持续的挑战。现有的深度学习模型通常受到域偏移（domain shift）【】的影响，导致训练数据与测试数据之间的性能下降，特别是在涉及异构背景、传感器噪声变化以及运动伪影的场景中。传统方法如合成噪声增强【】和静态数据预处理【】无法有效应对真实世界测试环境中复杂的噪声分布和背景动态变化，从而在实际部署中导致泛化性能不足。 </font>

<font style="color:#000000;">近年来，深度学习技术在目标表征方面展现出了强大的能力。研究者们不断探索构建层级特征表达与跨层特征复用的方法，例如DNA-Net通过跨层特征复用来增强小目标响应，HCF-Net利用层次化上下文融合策略解决多尺度背景干扰问题，而SCTransNet则创新性地融合了空间注意力与通道重校准机制，在复杂热噪声干扰下依然保持了良好的检测鲁棒性。</font>**<font style="color:#000000;">然而，当前的研究对固定训练数据的依赖一定程度上限制了其对未知测试域的适应能力。</font>**

<font style="color:#000000;">为此，测试时适应（Test-Time Adaptation, TTA）通过在线参数调整（如BN Adapt、TENT）或离线数据增强（如SHOT、AdaAugment）缓解测试域偏移，但面临关键瓶颈：前者通常会带来额外的计算开销，后者受限于人工规则难以建模复杂域偏移。与现有方法不同，</font>**<font style="color:#000000;">本文方法无需修改模型架构或调整推理流程，通</font>****<font style="color:#000000;">过利用测试数据的背景信息来优化训练数据</font>****<font style="color:#000000;">，同时保持部署效率与特征鲁棒性。</font>**

<font style="color:#000000;">因此，我们提出一种两阶段优化框架，首先，我们提出 </font>**<font style="color:#000000;">Test-time Driven Fusion</font>**<font style="color:#000000;">，一种在训练前执行的针对测试时优化（Test-time Optimization）的数据增强策略，通过模拟测试环境、挖掘困难目标并优化目标嵌入方式，从而提升模型在测试时对小目标的检测能力。该策略的核心优势在于在训练阶段即对数据进行优化，使模型在测试时具备更强的自适应能力，无需额外的推理调整，即可在不同测试环境下保持优异的小目标检测性能。 </font><font style="color:#000000;">其次，设</font>**<font style="color:#000000;">计</font>****<font style="color:#000000;">Noise-guided Test-time Adaption</font>**<font style="color:#000000;">从测试图像中提取真实噪声特征构建动态噪声库，通过混合训练与自信息最小化损失约束模型学习噪声不变性。</font><font style="color:#000000;">这一方法能够模拟更为复杂的噪声场景，提升噪声建模的准确性，并为后续的图像去噪及噪声建模任务提供更加真实的测试数据。  最后，我们构建了一个全新的红外小目标训练集，</font><font style="color:#000000;">通过动态退化注入与目标-背景域对齐策略，为复杂域偏移场景下的红外小目标检测提供了高泛化性训练基准。</font>

<font style="color:#000000;">总而言之，本文的主要贡献如下：</font>

1. <font style="color:#000000;">我们</font><font style="color:#000000;">构建了一个全新的面向无人机红外的</font>**<font style="color:#000000;">跨域动态增强数据集</font>**<font style="color:#000000;">AugBlur-ISTD，涵盖运动模糊及复杂测试背景，增强数据集对真实测试场景的覆盖能力。数据集包含206张高分辨率图像，标注行人、自行车与汽车三类小目标。</font>
2. <font style="color:#000000;">我们提出了</font>**<font style="color:#000000;">Test-time Driven Fusion策略，</font>**<font style="color:#000000;">将困难目标自然嵌入测试背景，生成域对齐的合成样本。测试域先验信息主动融入训练数据，使模型在训练阶段即适应测试域特征，无需额外推理调整即实现跨域泛化能力提升。</font>
3. <font style="color:#000000;">我们提出</font>**<font style="color:#000000;">Noise-guided Test-time Adaptation策略，</font>**<font style="color:#000000;">通过直接提取测试集真实噪声的统计特性构建动态噪声库，摒弃人工噪声建模，实现传感器噪声特征的自适应增强和生成域对齐样本，显著提升模型在复杂噪声环境中的泛化能力。</font>



## Related Work
**<font style="color:rgb(38, 38, 38);">A.Single-frame Infrared Small Target Detection</font>**

<font style="color:#000000;">目前，复杂背景中红外弱小目标的精确检测与分割仍面临严峻挑战。受成像机理限制，红外图像常伴随低信噪比、弱对比度及非均匀背景干扰等问题，导致传统算法在跨场景应用时性能显著衰减。以局部对比度分析为主的传统框架（如Tophat、IPI）因缺乏多尺度特征建模能力，难以适应动态变化的背景纹理。深度学习技术通过构建层级特征表达，在目标表征学习方面展现出显著优势：DNA-Net通过跨层特征复用增强小目标响应，HCF-Net采用层次化上下文融合策略解决多尺度背景干扰问题，SCTransNet则创新性地融合空间注意力与通道重校准机制，在复杂热噪声干扰下仍能保持优异的检测鲁棒性。尽管这些方法显著提升了性能，但红外成像的域偏移问题以及标注数据稀缺导致的模型过拟合问题，仍然是制约实际应用的核心瓶颈。</font>



**<font style="color:rgb(38, 38, 38);">B. </font>****<font style="color:rgb(64, 64, 64);">Test-time Adaptation</font>**

<font style="color:#000000;">测试时适应（Test-Time Adaptation, TTA）旨在解决模型部署时因测试数据分布偏移（如光照变化、传感器噪声等）导致的性能下降问题。其核心挑战在于利用有限的测试数据信息提升模型泛化能力。训练与测试数据分布的不一致性导致模型在训练时学到的特征往往无法直接迁移到实际测试环境中，这一现象在小物体识别中尤为显著。传统方法通常依赖在线调整模型参数以缓解分布差异，例如经典工作BN Adapt通过直接替换测试批次的统计量更新批归一化层，仅需微秒级计算即可适应简单偏移，但其对复杂域差异（如跨模态变化）的局限性推动了更鲁棒方法的研究。</font>

<font style="color:#000000;">近年来，研究者开始探索通过测试数据的信息增强或生成合成样本，在训练阶段预适应目标域特征，从而减少实时推理时的计算负担。例如，Test-Time Training (TTT)通过自监督任务微调模型，间接实现分布对齐；SHOT利用伪标签迭代重构目标域特征，推动离线适应发展；AdaAugment直接从测试数据提取噪声特性构建增强策略，而MEMO通过多增强一致性约束模型稳定性。这些方法表明，测试数据驱动的特征增强与合成可有效降低对实时参数调整的依赖，为轻量化部署提供了新方向。</font>

<font style="color:#000000;">为了解决现有方法对在线参数调整的依赖及复杂噪声建模的不足，我们提出了Test-time Driven Fusion策略和Noise-guided Test-time Adaptation策略：前者通过域对齐的合成样本生成，将测试环境特征主动嵌入训练阶段，消除推理时额外调整需求；后者基于测试集真实噪声统计构建动态增强库，突破人工噪声假设限制，显著提升模型在未知噪声环境中的泛化鲁棒性。</font>

## Method
<font style="color:#000000;">为提升ISTD模型在真实环境下的泛化性能，本文提出一种新颖的双阶段优化框架。在数据处理阶段，通过Test-time Driven Fusion策略，整合测试域的背景信息。合成更具真实背景多样性的训练样本，从而扩充数据集并引入背景先验。在模型训练阶段，引入Noise-guided Test-time Adaptation策略，在模型特征空间中融入深层噪声统计信息，有效抑制模型在测试阶段受到的噪声干扰，显著提升检测性能。该框架通过数据-特征协同优化，分别从</font>**<font style="color:#000000;">数据空间</font>**<font style="color:#000000;">实现域对齐与</font>**<font style="color:#000000;">特征空间</font>**<font style="color:#000000;">降低域差异及噪声敏感性，最终实现模型在未知复杂环境中的稳健泛化。</font>

![图1.我们方法的整体框架。Stage1展示的是由(a)(b)部分构成的整体Test-time Driven Fusion框架。其中(a)展示的是提取训练集困难小目标（右图）和提取测试集的背景（左图）的过程。（b)是基于SSIM排名和泊松融合技术将困难目标自然地嵌入测试背景，创建新的训练样本。 Stage2展示的是Noise-guided Test-time Adaption的过程， 通过从测试图像提取真实噪声特征，与训练图像混合，同时使用自信息最小化损失提高模型对噪声的鲁棒性，其中的（c）详细展示了提取测试 集中的噪声的过程。](https://cdn.nlark.com/yuque/0/2025/png/50405538/1743501373598-da046c33-178b-4f6d-9cb4-c4f67d2832d8.png)

### Test-time Driven Fusion
<font style="color:#000000;background-color:rgb(252, 252, 252);">为增强模型对测试域的背景</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">先验适应性</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">，本节提出了Test-time Driven Fusion策略，如图1所示。我们利用Background Region Detection（BRD）和 双指标筛选 以筛选背景和难检目标，再通过Re-Generation生成与</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">测试域对齐的训练样本</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">，使模型在训练阶段能学习并补偿潜在的</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">跨域分布偏移</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">。</font>

**Background Region Detection(BRD). **<font style="color:#000000;background-color:rgb(252, 252, 252);">针对复杂场景下的背景区域检测问题，我们提出基于多频带特征融合的Background Region Detection策略。首先，为了抑制随机噪声干扰并增强图像结构特征，我们选择</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">小波多频带滤波（WMF）</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">对测试集</font>$ X^{test}\in \mathbb R^{640×512} $<font style="color:#000000;background-color:rgb(252, 252, 252);">进行预处理。具体而言，通过离散小波变换将图像分解成多个不同频率的子带，其中低频基带</font>$ L^{low} $<font style="color:#000000;background-color:rgb(252, 252, 252);">反映图像的</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">主要结构</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">，高频细节带</font>$ \{H_k^{high}\}^3_{k=1} $<font style="color:#000000;background-color:rgb(252, 252, 252);">则包含图像的</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">边缘细节</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">和</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">噪声成分</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">，WMF以保留了主要结构信息的低频子带为</font><font style="color:#000000;">引导图像</font><font style="color:#000000;background-color:rgb(252, 252, 252);">，对高频子带进行边缘感知滤波。具体公式如下：</font>

$ \begin{equation}
\tilde{H}_k^{high}  = H_k^{high} \cdot \frac{|\nabla L^{low}|}{\max(|\nabla L^{low}|) + \epsilon}
\end{equation} $

<font style="color:#000000;background-color:rgb(252, 252, 252);">式中，</font>$ \tilde H_k^{high} $<font style="color:#000000;background-color:rgb(252, 252, 252);">表达滤波后的高频细节带，</font>$ H_k^{high} $<font style="color:#000000;background-color:rgb(252, 252, 252);">是滤波前的高频细节带，</font>$ L^{low} $<font style="color:#000000;background-color:rgb(252, 252, 252);">是低频基带，</font>$ \epsilon $是<font style="color:#000000;background-color:rgb(252, 252, 252);">防止数值溢出的正则化项。进一步地，将处理后的图像分割成多个非重叠子块</font>$ B_{i,j} $<font style="color:rgb(6, 6, 7);">，接着在子块级别，</font><font style="color:#000000;background-color:rgb(252, 252, 252);">计算其边缘密度</font>$ d_{i,j} $<font style="color:#000000;background-color:rgb(252, 252, 252);">和拉普拉斯响应</font>$ l_{i,j} $<font style="color:#000000;background-color:rgb(252, 252, 252);">：</font>

$ \begin{equation}
d_{i,j} = \frac{1}{S^2} \sum_{(u,v) \in B_{i,j}} \sqrt{G_u^2 + G_v^2}
\end{equation} $

$ 
\begin{equation}
l_{i,j} = \sum_{(u,v) \in B_{i,j}} \left| \nabla^2 I(u,v) \right|
\end{equation} $

<font style="color:#000000;background-color:rgb(252, 252, 252);">其中，</font>$ (i,j) $<font style="color:#000000;background-color:rgb(252, 252, 252);">为空间坐标索引，</font>$ S $<font style="color:#000000;background-color:rgb(252, 252, 252);">为子块尺寸，</font>$ G_u $<font style="color:#000000;background-color:rgb(252, 252, 252);">,</font>$ G_v $<font style="color:#000000;background-color:rgb(252, 252, 252);">分别是子块的水平和垂直梯度场。</font>$ \nabla^2 $<font style="color:#000000;background-color:rgb(252, 252, 252);">为拉普拉斯算子。最终将特征归一化至</font>$ [0,1] $<font style="color:#000000;background-color:rgb(252, 252, 252);">并融合为背景倾向区域的概率</font>$ p_{i,j} $<font style="color:#000000;background-color:rgb(252, 252, 252);">：</font>

$ \begin{equation}
p_{i,j} = 0.5 d_{i,j} + 0.5 l_{i,j}
\end{equation} $

<font style="color:#000000;background-color:rgb(252, 252, 252);">最后，基于输入的</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">原始</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">图像</font>$ X_{}^{test} $<font style="color:#000000;background-color:rgb(252, 252, 252);">的每个子块</font>$ B_{i,j} $<font style="color:#000000;background-color:rgb(252, 252, 252);">，若其背景概率满足</font>$ p_{i,j} < \tau_b $<font style="color:#000000;background-color:rgb(252, 252, 252);">（</font>$ \tau_b $<font style="color:#000000;background-color:rgb(252, 252, 252);">为预设阈值），则判定为候选背景区域。聚合所有满足条件的子块坐标，裁剪出低纹理特征的背景信息的矩形区域</font>$ P_{}^{test} \in R^{126×126} $<font style="color:#000000;background-color:rgb(252, 252, 252);">。进一步地，通过双线性插值将裁剪区域 </font>$ P_{}^{test} $<font style="color:#000000;background-color:rgb(252, 252, 252);">上采样至原始分辨率</font>$ 640×512 $<font style="color:#000000;background-color:rgb(252, 252, 252);"> ，以保持与输入图像的几何一致性。如图2所示，WMF后的图像信噪比要更高，表明WMF可引导</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">区分</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">高频子带中的</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">真实边缘</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">与</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">随机噪声</font>**<font style="color:#000000;background-color:rgb(252, 252, 252);">。</font>

![图2. (a)是所提出的Background Region Detection框架说明。经过WMF(小波多频带滤波方法)和基于边缘密度和拉普拉斯响应的小目标概率计算，我们最后裁剪出合适的背景区域。经过滤波方法后图像的信噪比更均匀，背景裁剪的成功率提高。其中(b)是WMF的具体实现，L是小波分解出的低频子带，H是高频子带。](https://cdn.nlark.com/yuque/0/2025/png/50405538/1742726802971-8c9a0659-c3bb-4b28-a39f-82f68fa2e96e.png)

<font style="color:#000000;">本研究提出的基于小波多频带滤波（WMF）的背景区域检测方法，通过融合多频带特征显著提升了背景与目标的区分能力，通过图3</font><font style="color:#000000;background-color:rgb(252, 252, 252);">不同背景裁剪方法的对比分析可以发现</font><font style="color:#000000;">。相较于传统随机裁剪及基于边缘密度的方法，本方法通过低频结构引导的高频子带滤波技术，有效抑制了随机噪声干扰，并精准识别小目标与背景纹理的细微差异。实验表明，该方法在复杂场景下能够避免误判小目标为背景区域，生成高置信度的背景裁剪结果，为跨域数据优化提供了更可靠的先验信息。</font>

![图3. 不同Background Region Detection方法在test数据集上裁剪结果的可视化。左栏是背景区域倾向图，右栏是裁剪区域可视化图。红色框是不同方法的裁剪结果，黄色虚线圈出了其他方法对背景的误判，其识别的背景中出现了不需要的小目标，黄色框是为了对小目标进行清晰的放大展示](https://cdn.nlark.com/yuque/0/2025/png/50405538/1742659209572-f8e6a401-31e2-4651-8a3f-007fbaaa21c6.png)



**<font style="color:rgb(64, 64, 64);">双指标筛选Dual-indicator Selection.</font>**<font style="color:#000000;">为了确保了在训练过程中更加关注那些定位较为困难的目标，从而提高模型对小目标的学习能力，我们提出了一种双指标筛选策略。具体而言，首先基于初始模型权重 </font>$ f_𝜃
 $<font style="color:#000000;">对训练集</font>$  X^{\text{train}} $<font style="color:#000000;">生成预测掩膜</font>$ \hat Y $<font style="color:#000000;">，再结合真实标注掩膜</font>$ Y $<font style="color:#000000;">计算PixAcc和IoU。其中，像素准确率通过统计正确识别、误判和漏检的像素比例获得，重叠度则通过比较预测区域与真实标注区域的重合程度计算。进一步地，我们构建困难目标合集</font>$ P^{train} $<font style="color:#000000;">:</font>

$ \begin{equation}
P^{\text{train}} =
\begin{cases} 
\text {Choose} & \text{if } \text{PixAcc}(\hat Y， Y) < \tau_p \wedge \text{IoU}(\hat Y， Y) < \tau_i
\\
\varnothing & \text{otherwise}
\end{cases}

\end{equation}
 $

<font style="color:#000000;">式中，</font>$ \tau_p $<font style="color:#000000;">和</font>$ \tau_i $<font style="color:#000000;">分别是PixAcc和IoU的设定阈值。该准则确保同时覆盖分类错误率高与定位偏差显著的挑战性样本。</font>

<font style="color:#000000;"></font>

**Re-Generation**<font style="color:#000000;">.</font><font style="color:#000000;">该部分提出一种Re-Generation策略，整体框架如图4所示。首先对目标区域</font>$ P_{\text{}}^{\text{test}} $<font style="color:#000000;">进行局部窗口划分成</font>$ A^{test} $<font style="color:#000000;">，再通过计算候选区域集</font>$ P_{\text{}}^{\text{train}} $<font style="color:#000000;">与目标区域</font>$ A^{test} $<font style="color:#000000;">的 SSIM 评分，能够量化它们在颜色、局部对比度以及结构模式上的相似性，从而筛选出最适合融合的区域。在候选区域筛选阶段，所有区域的 SSIM 评分被计算并排序，以选择得分最高的区域作为最佳匹配区域。</font>

$ \begin{equation} 
\mathbf{A}^{top}= \arg \max_{{A^{test}}} \text {SSIM}\mathbf{(A_{\text{}}^{\text{test}}}, \mathbf{P_{\text{}}^{\text{train}}})
\end{equation}
 $

<font style="color:#000000;">式中</font>$ A^{test} $<font style="color:#000000;">是图像中的候选区域，通过使用Ranking-based Selection来选择最优的匹配区域</font>$ {A}^{top} $<font style="color:#000000;">，即最大化的 SSIM 值来决定最佳匹配。 同时，为了减小目标图像与背景图像在边界处的色差与拼接痕迹，我们采用 Poisson </font>~~<font style="color:#DF2A3F;"></font>~~<font style="color:#000000;">Fusion，通过调整边界区域的梯度场，使合成图像的过渡更加自然，其目标函数和边界条件如下：</font>

$ \begin{equation}


\min_{{X}^{p}} \iint_{A^{top}} \left| \nabla  {{X}^{p}} - \nabla{P^{\text{train}}} \right|^2 \, du \, dv





\end{equation}

 $

$ \begin{equation}
\displaystyle
{I}^{a}|_{\partial A^{top}} = {I}^{a*}|_{\partial A^{top}}
\end{equation}
 $

<font style="color:#000000;">式中，</font>$ {X}^{p} $<font style="color:#000000;">表示合成后的图像，</font>$ \nabla $<font style="color:#000000;">表示的梯度，</font>$ A^{top} $<font style="color:#000000;">是合成图像中被前景图像覆盖的区域。合并后图像在</font>$ A^{top} $<font style="color:#000000;">内的像素表示函数是</font>$ {I}^{a} $<font style="color:#000000;">，在</font>$ A^{top} $<font style="color:#000000;">外的像素值表示函数是</font>$ {I}^{a*} $<font style="color:#000000;">，</font>$ \partial A^{top} $<font style="color:#000000;">表示</font>$ A^{top} $<font style="color:#000000;">区域的边界。</font>

![图4.该流程图展示了基于结构相似性（SSIM）的候选区域选择流程：首先通过计算测试图像块与训练图像块的颜色和纹理相似性，生成候选区域的SSIM评分（如0.84、0.77、0.71、0.60）；随后按评分从高到低排序，选择最优匹配区域（如SSIM=0.84）进行目标嵌入](https://cdn.nlark.com/yuque/0/2025/png/50405538/1743346258633-f207fd70-be56-43b0-b494-b0ece84f15bb.png)

![](https://cdn.nlark.com/yuque/0/2025/png/50405538/1741448070332-a9463dfb-310f-4fcc-838f-da8e4af8af0a.png)![](https://cdn.nlark.com/yuque/0/2025/png/50405538/1741448098371-05b29e90-f7e9-4679-a7aa-aa8670c8504c.png)![](https://cdn.nlark.com/yuque/0/2025/png/50405538/1741448115157-b6b95038-1285-43b2-901b-c71f260f1f3d.png)



### Noise-guided Test-time Adaption
<font style="color:#000000;"> 为了引导模型学习从训练域到测试域的噪声不变性特征，本节提出了一种Noise-guided Test-time Adaption策略，整体框架如图5所示。通过噪声特征引导的测试时自适应机制，显著提升了模型在复杂噪声场景下的泛化能力，有效解决了传统静态模型对测试域分布漂移敏感的问题。</font>

![图5.Noise-guided Test-time Adaption的整体流程，上半部分是Noise Mixing的过程，先从测试集中提取合适的噪声再放大并mixup到训练集中，形成新的数据集。下半部分是baseline以及损失的计算方式（由监督损失和自信息最小化构成的损失）协同优化模型对噪声干扰的鲁棒性，最终提升复杂场景下的显著性目标检测精度。其中，（a）(b)是CAF里面的模块](https://cdn.nlark.com/yuque/0/2025/png/50405538/1743408308351-8db96670-6d1d-4d3b-9220-d6a7e0131e8a.png)

**<font style="color:#000000;">Network Structure.</font>**<font style="color:#000000;">本研究采用了一种基于U-Net架构的红外目标检测网络模型，其核心设计包含编码器-解码器结构与多模态特征融合机制。编码器采用五层残差下采样模块</font>$ \{E_i\}^5_{i=1} $<font style="color:#000000;">提取多尺度高阶特征，其中</font>$ E_i \in \mathbb{R}^{H/2^i \times W/2^i\times C_i},
 $<font style="color:#000000;">，并通过块嵌入生成低维语义表示；解码器则通过上采样</font>$ \{D_i\}^4_{i=1} $<font style="color:#000000;">与跳跃连接逐步恢复空间分辨率，对</font>**<font style="color:#000000;">每个</font>**<font style="color:#000000;">解码层输出施加二元交叉熵损失，并通过  Context-aware Attention Fusion Module（CAF）聚合不同层级的上下文信息生成最终预测并通过加权求和得到最终的总损失。具体来说，损失</font>$ L_{BCE} $<font style="color:#000000;">由所有尺度的损失组成：</font>

$ \begin{equation}
\mathcal{L}_{\text BCE} = \sum_{i=1}^{4} \lambda_i \cdot {BCE(Y, \hat{Y})}
\end{equation} $

<font style="color:#000000;">其中，</font>$ BEC(\cdot) $<font style="color:#000000;">是二元交叉熵损失函数。</font>$ Y $<font style="color:#000000;">是第</font>$ i $<font style="color:#000000;">个尺度的真实标签，</font>$ \hat{Y} $<font style="color:#000000;">是该尺度的模型预测输出，</font>$ L_{\text{BCE}} $<font style="color:#000000;">表示监督损失，</font>$ \lambda_i $<font style="color:#000000;">是每个尺度的权重系数。</font>



**<font style="color:rgb(64, 64, 64);">Real-world Noise Mixing.</font>**<font style="color:rgb(64, 64, 64);">为提升红外小目标检测模型对真实噪声的泛化能力，本研究提出了一种Real-world Noise Mixing策略。针对</font>**<font style="color:rgb(64, 64, 64);">红外小目标检测数据集</font>**<font style="color:rgb(64, 64, 64);">，其数据主要来源于单一传感器，导致噪声分布呈现</font>**<font style="color:rgb(64, 64, 64);">高度相似性</font>**<font style="color:rgb(64, 64, 64);">。本方法通过从真实测试数据中提取噪声特征块，构建动态噪声库，实现噪声-数据的自适应融合。</font>

<font style="color:rgb(64, 64, 64);">由于不同传感器工作状态存在差异，噪声分布呈现相似但不完全一致的特性，为捕捉噪声的动态特征，从测试集</font>$ X^{test} $<font style="color:rgb(64, 64, 64);">中选取</font>$ k
 $<font style="color:rgb(64, 64, 64);">组</font>**<font style="color:rgb(64, 64, 64);">噪声分布一致</font>**<font style="color:rgb(64, 64, 64);">的图像构建噪声样本库：</font>

$ \begin{equation}
X^{test}=[X^{test}_1,X^{test}_2,…,X^{test}_k]\in \mathbb R^{k×c×h×w}
\end{equation} $

进一步地，通过滑动窗口技术对每组噪声样本都划分180个采样区域$ \{A_n^{noise}\}_{n=1}^{180} \in \mathbb R^{n×c×\frac{h}{15}×\frac{w}{12}} $。 在噪声采样区域的选择上，<font style="color:rgb(64, 64, 64);">我们注意到，图像中纹理丰富的区域通常具有较高的方差，但这些纹理往往会掩盖噪声本身的特征。为了确保所提取的噪声序列具有均匀的纹理分布，我们依据图像的局部方差和梯度信息，自适应地选择噪声采样区域，构建噪声库 </font>$ P^{noise}\in \mathbb{R}^{k×c×h×w} $<font style="color:rgb(64, 64, 64);">：</font>

$ \begin{equation}
{P^{\text{noise}}} = 
\begin{cases} 
\text{Resize}({A_n^{noise}}), & \text{if }\sigma^2  <\sigma_{max}^2\text{ and } \mu_{\text{min}}  < \mu  \\
\varnothing
, & \text{otherwise}
\end{cases}
\end{equation}

 $

其中，为了确保噪声样本具有显著的噪声特征，设置了方差的阈值$ \sigma_{max}^2 $ 和均值的阈值$ \mu_{min} $。<font style="color:rgb(64, 64, 64);">为捕捉噪声的动态特征，</font>噪声图像与原数据集进行<font style="color:rgb(64, 64, 64);">通过线性插值生成混合样本</font>$ X^{noise} $

$ \begin{equation}
{X^{noise}} = \lambda·\text {R}(P^{noise}) + (1 - \lambda) X^{train}  ,  \quad  \lambda ∼U(0,1)
\end{equation} $

其中，$ R(\cdot) $指的是从噪声库中随机采样。$  X^{train}  $是制作的数据集。$ \lambda $控制噪声注入强度。经过实验$ \lambda=0.5 $的效果最佳。



**<font style="color:rgb(64, 64, 64);">Self-Information Minimization Test-Time Adaptation.</font>**<font style="color:rgb(64, 64, 64);">为了增强模型在测试时对真实噪声扰动的鲁棒性，我们提出一种基于特征预测一致性的自信息最小化测试时学习策略。通过离线构建测试域噪声特征库，并在训练阶段向输入数据注入测试噪声生成扰动样本，构建共享权重的双分支网络架构：其中主分支提取原始干净图像的全局特征，辅助分支提取测试噪声扰动样本的混合特征，通过最小化两分支特征间的自信息，实现模型特征空间与测试环境的隐式对齐。该机制强制模型在输出特征空间中对原始数据与含噪声数据保持预测一致性，从而抑制测试集异质噪声干扰对特征提取过程的影响，提升模型对显著性语义特征的捕获能力。</font>

<font style="color:rgb(64, 64, 64);">在特征提取层面，本文选取解码器路径末端（即经下采样及上采样重建后的最终输出层）作为关键特征提取点，其输出特征可表示为</font>$ D_{1}\in \mathbb{R}^{H \times W \times 1} $<font style="color:rgb(64, 64, 64);">层。通过共享特征提取空间，同时获取原始</font>$ X^{train}  $<font style="color:rgb(64, 64, 64);">与混合样本</font>$ {X^{noise}}  $<font style="color:rgb(64, 64, 64);">提取末端特征</font>$ \hat{Y}^{trian} $<font style="color:rgb(64, 64, 64);">和</font>$ \hat{Y}^{noise} $<font style="color:rgb(64, 64, 64);">，为约束特征分布一致性，设计自信息差异损失函数：	</font>

$ \begin{equation}
\mathcal{L}_{\text{SIM}} = \frac{1}{N} \sum_{i=1}^{N} \left( -\log(\hat Y^{\text{noise}} + \epsilon) + \log(\hat Y^{\text{train}} + \epsilon) \right)^2
\end{equation} $

<font style="color:rgb(64, 64, 64);">其中</font>$ \epsilon = 10^{-8}
 $<font style="color:rgb(64, 64, 64);">用于数值稳定性控制，</font>$ N $<font style="color:rgb(64, 64, 64);">为批次大小。 该损失函数通过负对数似然比对两特征空间进行相似性度量，其最小化过程迫使编码器提取与噪声无关的鲁棒特征表示。 </font>

<font style="color:rgb(64, 64, 64);">最后，模型通过联合优化目标分割监督损失与特征一致性损失实现性能提升：</font>

$ \begin{equation}
\mathcal{L}_{\text{total}} = 
\underbrace{\mathcal{L}_{\text{BCE}}}_{\text{分割损失}}
 \quad+ \underbrace{\mathcal{L}_{\text{SIM}}}_{\text{特征一致性损失}}
\end{equation}
 $

<font style="color:rgb(64, 64, 64);">式中二元交叉熵损失</font>$ \mathcal{L}_{BCE} $<font style="color:rgb(64, 64, 64);">提供像素级分割监督，而自信息最小化损失</font>$ {\mathcal{L}_{SIM}} $<font style="color:rgb(64, 64, 64);">则通过特征空间正则化增强模型对复杂背景的适应能力。</font>



## Dataest(AugBlur-ISTD)
<font style="color:rgb(64, 64, 64);">为解决无人机红外小目标检测中因运动模糊导致的性能退化问题，图6展示我们构建的动态增强数据集AugBlur-ISTD。从原始数据中筛选出103张具有代表性的红外图像，经过严格的筛选与标注流程，精确地定位图像中的小目标并进行了像素级标注。为模拟真实飞行状态，本研究提出Dynamic Degradation Injection（</font>**动态退化注入策略**<font style="color:rgb(64, 64, 64);">）：将70%原始数据保留为自然样本，剩余30%通过运动模糊退化模型处理，生成退化样本。进一步结合Test-time Driven Fusion，生成103张</font>**<font style="color:rgb(64, 64, 64);">目标-背景</font>**<font style="color:rgb(64, 64, 64);">混合增强样本，最终构建包含206张图像的</font>**<font style="color:rgb(64, 64, 64);">增强数据集</font>**$ D^{train}=\{X^{trian}_k,Y_k\}^{206}_{k=1} $<font style="color:rgb(64, 64, 64);">。所有图像分辨率为640×512，如图6所示，覆盖校园、停车场等多场景，包含行人、自行车及汽车三类关键目标，并在不同飞行高度与视角下采集，以模拟无人机实际观测条件。</font>

<font style="color:rgb(64, 64, 64);">该框架在数据处理阶段引入Dynamic Degradation（</font>**<font style="color:rgb(64, 64, 64);">动态退化）</font>**<font style="color:rgb(64, 64, 64);">与</font>**<font style="color:rgb(64, 64, 64);">混合增强机制</font>**<font style="color:rgb(64, 64, 64);">，通过模拟测试环境背景扰动及边缘困难目标强化，有效缓解小目标检测中的域偏移问题。实验表明，该方法显著提升了模型在复杂场景下的泛化能力，尤其在图像质量退化与背景异质性场景中表现突出。与传统的测试时适应技术（Test-Time Adaptation）相比，所提出的预训练优化策略形成功能互补，为小目标检测任务提供了一种兼顾鲁棒性与效率的增强范式。</font><font style="color:#DF2A3F;"></font>

![图6.(a)图表示数据集的组成分布，31张运动模糊组，72张原始采集组，103张增强组，(b)图呈现了多场景下的背景与目标示例](https://cdn.nlark.com/yuque/0/2025/png/50405538/1743410007549-acffb095-3d81-4cca-9539-b05e71f69835.png)

## Experiment
### <font style="color:rgb(0,0,0);">Experimental Settings</font>
**<font style="color:rgb(0,0,0);">Metrics </font>**<font style="color:rgb(64, 64, 64);"> 我们采用五种评估指标对模型性能进行全面评估：像素精度(PixAcc)计算正确分类像素的比例，直观反映分类准确性；平均交并比(mIoU)通过计算预测区域与真实区域交集除以并集的均值，全面衡量分割质量；归一化交并比(nIoU)对标准IoU进行改进，通过归一化处理更好地应对类别不平衡问题，特别适合小目标分割评估；检测概率(Pd)计算正确检测目标占所有真实目标的比例，反映模型的完整性；F1分数作为精确率与召回率的调和平均，提供了模型性能的综合评估，平衡了准确性与完整性。这些指标从不同角度评估了模型在像素分类、区域分割及目标检测方面的表现，能够全面客观地分析所提方法的有效性。  </font>

**<font style="color:rgb(0,0,0);">Implementation Details </font>**<font style="color:rgb(64, 64, 64);"> 我们基于PyTorch，在配备NVIDIA GPU的环境下进行训练与测试。训练采用batch size为8的Adam优化器，初始学习率0.001，结合余弦退火策略逐步降至</font>$ 10^{-5} $<font style="color:rgb(64, 64, 64);">，总训练轮数为1000轮。网络参数使用Kaiming方法初始化，采用二元交叉熵损失函数指导训练过程。  </font>

### <font style="color:rgb(0,0,0);">Comparison with State-of-the-Art Methods  </font>
**<font style="color:rgb(64, 64, 64);">定量分析</font>**<font style="color:rgb(64, 64, 64);"> 为了全面评估本研究所提出方法的有效性，我们将其与当前红外小目标检测领域的多种先进方法进行了对比，包括ACM-Net、ALC-Net、DNA-Net、RDIAN、ISTDU-Net、UIU-Net、HCF-Net和SCTransNet。如表1所示，本研究方法在mIoU（75.44）、nloU(62.42)、Pd(69.75)、F1（84.98）指标上均优于现有方法。</font>

<font style="color:#8A8F8D;">表1.目标检测性能对比。展示了最先进的方法与我们的方法在五个评估指标（像素精度(PixAcc)、平均交并比(mIoU)、归一化交并比(nIoU)、检测概率(Pd)、F1）上的性能表现。最好的结果用红色表示</font>

| <font style="color:rgb(64, 64, 64);">Model</font> | <font style="color:rgb(64, 64, 64);">Performance</font> | | | | |
| :---: | :---: | --- | --- | --- | --- |
| | <font style="color:rgb(64, 64, 64);">PixAcc</font> | <font style="color:rgb(64, 64, 64);">mIoU</font> | <font style="color:rgb(64, 64, 64);">nIoU</font> | <font style="color:rgb(64, 64, 64);">Pd</font> | <font style="color:rgb(64, 64, 64);">F1</font> |
| ACM-Net  | <font style="color:rgb(64, 64, 64);">84.59</font> | <font style="color:rgb(64, 64, 64);">46.80</font> | <font style="color:rgb(64, 64, 64);">37.84</font> | <font style="color:rgb(64, 64, 64);">38.48</font> | <font style="color:rgb(64, 64, 64);">60.25</font> |
| ALC-Net | <font style="color:rgb(64, 64, 64);">85.27</font> | <font style="color:rgb(64, 64, 64);">36.18</font> | <font style="color:rgb(64, 64, 64);">30.85</font> | <font style="color:rgb(64, 64, 64);">30.25</font> | <font style="color:rgb(64, 64, 64);">50.81</font> |
| DNA-Net | <font style="color:rgb(64, 64, 64);">79.91</font> | <font style="color:rgb(64, 64, 64);">71.30</font> | <font style="color:rgb(64, 64, 64);">57.69</font> | <font style="color:rgb(64, 64, 64);">60.00</font> | <font style="color:rgb(64, 64, 64);">75.36</font> |
| RDIAN | <font style="color:rgb(64, 64, 64);">73.95</font> | <font style="color:rgb(64, 64, 64);">62.00</font> | <font style="color:rgb(64, 64, 64);">51.20</font> | <font style="color:rgb(64, 64, 64);">59.49</font> | <font style="color:rgb(64, 64, 64);">67.45</font> |
| ISTDU-Net | <font style="color:rgb(64, 64, 64);">78.27</font> | <font style="color:rgb(64, 64, 64);">71.25</font> | <font style="color:rgb(64, 64, 64);">58.01</font> | <font style="color:rgb(64, 64, 64);">62.15</font> | <font style="color:rgb(64, 64, 64);">74.60</font> |
| UIU-Net | <font style="color:rgb(64, 64, 64);">78.28</font> | <font style="color:rgb(64, 64, 64);">69.50</font> | <font style="color:rgb(64, 64, 64);">50.43</font> | <font style="color:rgb(64, 64, 64);">59.11</font> | <font style="color:rgb(64, 64, 64);">68.43</font> |
| HCF-Net | <font style="color:rgb(64, 64, 64);">47.95</font> | <font style="color:rgb(64, 64, 64);">35.45</font> | <font style="color:rgb(64, 64, 64);">37.42</font> | <font style="color:rgb(64, 64, 64);">54.31</font> | <font style="color:rgb(64, 64, 64);">40.76</font> |
| SCTransNet | <font style="color:#000000;">78.15</font> | <font style="color:#000000;">73.93</font> | <font style="color:#000000;">60.92</font> | <font style="color:#000000;">67.21</font> | <font style="color:#000000;">83.60</font> |
| **Ours** | <font style="color:rgb(64, 64, 64);">82.54</font> | **<font style="color:#DF2A3F;">75.44</font>** | **<font style="color:#DF2A3F;">62.42</font>** | **<font style="color:#DF2A3F;">69.75</font>** | **<font style="color:#DF2A3F;">84.98</font>** |


**可视化分析**<font style="color:rgb(64, 64, 64);"> 在原始数据集中的8种代表性算法的可视化结果如图7所示。其中HCF-Net存在对于相对较小的人形小目标大量漏检，其他的模型也存在不少漏检和错检的情况，对于背景往往容易产生误判，在这种情况下即使识别到了小目标，其轮廓也不清晰。但经过本研究的方法，模型对于背景区域的判断准确率有较大程度提升，目标监测更精确，也能很好的区分两个位置较近的目标。对比图片（1），SCT算法识别出人形小目标的清晰度与本研究的方法不相上下，但其在背景区域出现了错判，其他模型算法也对背景出现了不同程度的错判。对于图片（3），本文方法对于汽车小目标识别的精确度和清晰度也更加贴合真实的小目标，不会出现其他模型轮廓不清晰的情况。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/50405538/1741696277347-af1d62b3-e226-404e-b30d-638aa6b38dfa.png)

<font style="color:#8A8F8D;">图7.  不同模型在数据集上的可视化结果。蓝色、黄色和红色圈圈分别代表正确检测到的目标、漏检和误报，蓝色框是为了对小目标进行清晰的放大观察。</font>

**效率分析**<font style="color:rgb(64, 64, 64);"> 如表2所示，我们的方法在精度和效率的平衡上显著优于其他模型。与参数量相近的SCTransNet相比，IoU提升至75.37（+1.44），推理时间缩小了0.38；相较于高效率的DNA-Net，以略高的参数量和推理时间换取IoU提升4.07；对比参数量达50.54M的UIU-Net，仅用22%的参数量便实现更高IoU且速度翻倍。面对轻量级模型（如RDIAN、ALC-Net），其IoU分别领先13.37和39.19，兼顾实用性。同时避免了HCF-Net等高参数低效模型的缺陷，综合表现最优。</font>

<font style="color:#8A8F8D;">表2 .不同baseline性能对比：效率（Params/Inference times）与精度（IoU）</font>

| **Model** | **Performance** | | | |
| :---: | :---: | --- | --- | --- |
| |  Pub year | Params (M) | Inference times（10<sup>-2</sup>S) | <font style="color:rgb(64, 64, 64);">IoU</font> |
| ACM-Net  | <font style="color:#000000;">2021</font> | <font style="color:#000000;">0.398</font> | <font style="color:rgb(64, 64, 64);">2.41</font> | <font style="color:#000000;">46.80</font> |
| ALC-Net | <font style="color:#000000;">2021</font> | <font style="color:#000000;">0.427</font> | <font style="color:rgb(64, 64, 64);">1.06</font> | <font style="color:#000000;">36.18</font> |
| DNA-Net | <font style="color:#000000;">2022</font> | <font style="color:#000000;">4.697</font> | <font style="color:rgb(64, 64, 64);">4.58</font> | <font style="color:#000000;">71.30</font> |
| RDIAN | <font style="color:#000000;">2022</font> | <font style="color:#000000;">0.217</font> | <font style="color:rgb(64, 64, 64);">1.73</font> | <font style="color:#000000;">62.00</font> |
| ISTDU-Net | <font style="color:#000000;">2022</font> | <font style="color:#000000;">2.752</font> | <font style="color:rgb(64, 64, 64);">3.78</font> | <font style="color:#000000;">71.25</font> |
| UIU-Net | <font style="color:#000000;">2022</font> | <font style="color:#000000;">50.54</font> | <font style="color:rgb(64, 64, 64);">1.97</font> | <font style="color:#000000;">69.50</font> |
| HCF-Net | <font style="color:#000000;">2024</font> | <font style="color:#000000;">15.29</font> | <font style="color:rgb(64, 64, 64);"> 53.72</font> | <font style="color:#000000;">35.45</font> |
| SCTransNet | <font style="color:#000000;">2024</font> | <font style="color:#000000;">11.19</font> | <font style="color:rgb(64, 64, 64);">9.37</font> | <font style="color:#000000;">73.93</font> |
| **Ours** | <font style="color:#000000;">2025</font> | <font style="color:#000000;">11.33</font> | <font style="color:rgb(64, 64, 64);">8.99</font> | <font style="color:#DF2A3F;">75.37</font> |


### **<font style="color:rgb(0,0,0);">Ablation Studies </font>**
<font style="color:rgb(64, 64, 64);">本小节系统地评估了多策略协同优化对语义分割模型性能的影响，并且分别详细阐述了Background Region Detection块筛选的有效性</font>**、**<font style="color:rgb(64, 64, 64);">Noise-guided Test-time Adaption的参数α对实验的影响。</font>

**Background Region Detection的有效性  **<font style="color:rgb(64, 64, 64);">Background Region Detection的有效性源于小波滤波与概率裁剪的协同作用，它们减少了随机噪声对分割的干扰并保留了关键结构信息。小波滤波可能通过分离低频基带和高频细节提升了图像特征提取能力，而概率裁剪则精确定位了关键区域。表4展示的性能差异表明这两种技术存在互补效应，缺少任何一个都会导致性能下降。因此该背景区域检测组件通过多技术融合有效提高了图像分割精度，对提升后续处理效果具有实质性贡献，在实际应用中应优先采用完整的配置以获得最佳性能。  </font>

<font style="color:#8A8F8D;">表4. 不同背景区域检测方法的性能比较  </font>

| <font style="color:rgb(64, 64, 64);"></font> | <font style="color:rgb(64, 64, 64);">Performance</font> | | | | |
| :---: | :---: | --- | --- | --- | --- |
| | <font style="color:rgb(64, 64, 64);">PixAcc</font> | <font style="color:rgb(64, 64, 64);">mIoU</font> | <font style="color:rgb(64, 64, 64);">nIoU</font> | <font style="color:rgb(64, 64, 64);">Pd</font> | <font style="color:rgb(64, 64, 64);">F1</font> |
| <font style="color:rgb(64, 64, 64);">baseline</font> | <font style="color:#000000;"> 78.15</font> | <font style="color:#000000;">73.93</font> | <font style="color:#000000;">60.92</font> | <font style="color:#000000;"> 67.21</font> | <font style="color:#000000;">83.60</font> |
| <font style="color:rgb(64, 64, 64);">Random</font><br/><font style="color:rgb(64, 64, 64);">Crop</font> | <font style="color:rgb(64, 64, 64);">79.40</font> | <font style="color:rgb(64, 64, 64);">74.23</font> | <font style="color:rgb(64, 64, 64);">60.38</font> | <font style="color:rgb(64, 64, 64);">66.71</font> | <font style="color:rgb(64, 64, 64);">84.08</font> |
| <font style="color:rgb(64, 64, 64);">w/o WMF</font> | <font style="color:rgb(64, 64, 64);">79.10</font> | <font style="color:rgb(64, 64, 64);">74.27</font> | <font style="color:rgb(64, 64, 64);">60.51</font> | <font style="color:rgb(64, 64, 64);">64.43</font> | <font style="color:rgb(64, 64, 64);">84.00</font> |
| <font style="color:rgb(64, 64, 64);">all</font> | <font style="color:rgb(64, 64, 64);">80.66</font> | <font style="color:rgb(64, 64, 64);">75.37</font> | <font style="color:rgb(64, 64, 64);">61.52</font> | <font style="color:rgb(64, 64, 64);">65.70</font> | <font style="color:rgb(64, 64, 64);">84.75</font> |




**Noise-guided Test-time Adaption的参数  **<font style="color:rgb(64, 64, 64);">表5反映了噪声引导测试时适应机制中的一个重要平衡点。当α=0.5时达到最佳性能，这可能表明模型在学习过程中需要适当保留原始图像信息的同时引入足够的噪声变化。过低的α值（如0.1，0.3）可能导致引入的噪声不足以促使模型学习适应性特征；而过高的α值（如0.7，0.9）则可能引入过多噪声，破坏了原始图像中的关键信息结构。而在α=0.5时，模型能够更有效地从混合图像中提取与原始图像保持一致的特征表示，同时学习对噪声的鲁棒性，从而在各项性能指标上取得最佳综合效果。  </font>

<font style="color:#8A8F8D;"></font>

<font style="color:#8A8F8D;">表5.</font><font style="color:#8A8F8D;">Noise-guided Test-time Adaption</font><font style="color:#8A8F8D;">的Mixup参数性能对比（PixAcc/mIoU/nIoU/Pd/F1）</font>

|  | Performance | | | | |
| :---: | :---: | --- | --- | --- | --- |
| | <font style="color:rgb(64, 64, 64);">PixAcc</font> | <font style="color:rgb(64, 64, 64);">mIoU</font> | <font style="color:rgb(64, 64, 64);">nIoU</font> | <font style="color:rgb(64, 64, 64);">Pd</font> | <font style="color:rgb(64, 64, 64);">F1</font> |
| _<font style="color:rgb(0, 0, 0);">α=0.1</font>_ | <font style="color:rgb(64, 64, 64);">80.56</font> | <font style="color:rgb(64, 64, 64);">75.37</font> | <font style="color:rgb(64, 64, 64);">62.23</font> | <font style="color:rgb(64, 64, 64);">68.86</font> | <font style="color:rgb(64, 64, 64);">84.83</font> |
| _<font style="color:rgb(0, 0, 0);">α=0.3</font>_ | <font style="color:rgb(64, 64, 64);">79.97</font> | <font style="color:rgb(64, 64, 64);">75.37</font> | <font style="color:rgb(64, 64, 64);">61.03</font> | <font style="color:rgb(64, 64, 64);">65.95</font> | <font style="color:rgb(64, 64, 64);">84.67</font> |
| _<font style="color:rgb(0, 0, 0);">α=0.5</font>_ | <font style="color:#DF2A3F;">82.54</font> | <font style="color:#DF2A3F;">75.44</font> | <font style="color:#DF2A3F;">62.42</font> | <font style="color:#DF2A3F;">69.75</font> | <font style="color:#DF2A3F;">84.98</font> |
| _<font style="color:rgb(0, 0, 0);">α=0.7</font>_ | <font style="color:rgb(64, 64, 64);">79.56</font> | <font style="color:rgb(64, 64, 64);">74.91</font> | <font style="color:rgb(64, 64, 64);">61.11</font> | <font style="color:rgb(64, 64, 64);">68.61</font> | <font style="color:rgb(64, 64, 64);">84.38</font> |
| _<font style="color:rgb(0, 0, 0);">α=0.9</font>_ | <font style="color:rgb(64, 64, 64);">81.05</font> | <font style="color:rgb(64, 64, 64);">75.12</font> | <font style="color:rgb(64, 64, 64);">62.09</font> | <font style="color:rgb(64, 64, 64);">68.61</font> | <font style="color:rgb(64, 64, 64);">84.55</font> |




**<font style="color:rgb(64, 64, 64);">整体分析</font>**<font style="color:rgb(64, 64, 64);"> 如表3所示，仅基线模型综合性能最低表明未利用测试集增强（TTA）时模型泛化能力有限，其综合性能显著低于其他实验组（PixAcc: 78.15%, mIoU: 73.93%）。通过</font>**<font style="color:rgb(64, 64, 64);">融合数据集</font>**<font style="color:rgb(64, 64, 64);">的策略（Re-Generation）多样化小目标场景，显著提升PixAcc（+3.16%）和部分子指标。进一步引入</font>**<font style="color:rgb(64, 64, 64);">自信息最小化</font>**<font style="color:rgb(64, 64, 64);">策略后，模型在噪声干扰下的鲁棒性显著增强（nIoU: +0.82%, Pd: +4.18%），且综合指标趋于稳定（F1: 84.98），有效抑制了测试集噪声的负面影响。实验结果表明，多策略协同优化可突破单一模块的性能瓶颈。</font>

<font style="color:#8A8F8D;">表3 对比不同策略组合对语义分割性能的影响，包括基线模型（Baseline）、测试集增强生成融合数据（Fusion datasets）、自信息差异学习（Self-Information）的启用状态（✓/×）。</font>

| <font style="color:rgb(64, 64, 64);">baseline</font> | <font style="color:rgb(64, 64, 64);">Fusion datesets</font> | <font style="color:rgb(64, 64, 64);">Self-Information</font> | <font style="color:rgb(64, 64, 64);">Performance</font> | | | | |
| :---: | :---: | :---: | :---: | --- | --- | --- | --- |
| | | | <font style="color:rgb(64, 64, 64);">PixAcc</font> | <font style="color:rgb(64, 64, 64);">mIoU</font> | <font style="color:rgb(64, 64, 64);">nIoU</font> | <font style="color:rgb(64, 64, 64);">Pd</font> | <font style="color:rgb(64, 64, 64);">F1</font> |
| √ | <font style="color:rgb(64, 64, 64);">×</font> | <font style="color:rgb(64, 64, 64);">×</font> | <font style="color:#000000;">78.15</font> | <font style="color:#000000;"> 73.93</font> | <font style="color:#000000;"> 60.92</font> | <font style="color:#000000;">67.21</font> | <font style="color:#000000;"> 83.60</font> |
| √ | <font style="color:rgb(64, 64, 64);">√</font> | <font style="color:rgb(64, 64, 64);">×</font> | <font style="color:rgb(64, 64, 64);">81.31</font><font style="color:rgb(64, 64, 64);">（+3.16%）</font> | <font style="color:rgb(64, 64, 64);">74.90</font> | <font style="color:rgb(64, 64, 64);">61.60</font> | <font style="color:rgb(64, 64, 64);">65.57</font> | <font style="color:rgb(64, 64, 64);">84.59</font> |
| √ | <font style="color:rgb(64, 64, 64);">√</font> | <font style="color:rgb(64, 64, 64);">√</font> | <font style="color:#DF2A3F;">82.54</font> | <font style="color:#DF2A3F;">75.44</font> | <font style="color:#DF2A3F;">62.42</font> | <font style="color:#DF2A3F;">69.75</font> | <font style="color:#DF2A3F;">84.98</font> |




## Conclusion
<font style="color:#000000;">本文针对红外小目标检测中的跨域分布偏移问题，提出了一种域适应增强框架。Test-time Driven Fusion在无需额外推理调整的条件下，提升了模型对未知测试环境的适应性。Noise-guided Test-time Adaption突破了人工噪声假设的局限性，还增强了模型对异质噪声的鲁棒性。通过大量实验证明，我们的⽅法有效地提⾼了模型的性能。</font>





