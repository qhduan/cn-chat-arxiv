# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [HeadEvolver: Text to Head Avatars via Locally Learnable Mesh Deformation](https://arxiv.org/abs/2403.09326) | 通过可学习的局部网格变形技术，HeadEvolver框架可以通过文本引导生成高质量的头部头像，保留细节并支持编辑和动画。 |
| [^2] | [Neural Redshift: Random Networks are not Random Functions](https://arxiv.org/abs/2403.02241) | 本论文研究了未经训练的随机权重网络，发现即使简单的MLPs也具有强烈的归纳偏见，不同于传统观点的是，NNs并不具有固有的“简单偏见”，而是依赖于组件的作用。 |
| [^3] | [Round Trip Translation Defence against Large Language Model Jailbreaking Attacks](https://arxiv.org/abs/2402.13517) | 往返翻译（RTT）方法是第一个专门设计用于抵御大型语言模型（LLMs）社交工程攻击的算法，成功地减少了多种攻击形式的成功率。 |
| [^4] | [On Generating Explanations for Reinforcement Learning Policies: An Empirical Study.](http://arxiv.org/abs/2309.16960) | 本文通过引入一组线性时态逻辑（LTL）公式，介绍了一种生成强化学习策略解释的方法，并展示了其在模拟夺旗环境中的有效性。 |
| [^5] | [LEyes: A Lightweight Framework for Deep Learning-Based Eye Tracking using Synthetic Eye Images.](http://arxiv.org/abs/2309.06129) | 本研究提出了一种名为LEyes的轻量级深度学习眼动跟踪框架，利用合成眼部图像进行训练，解决了由于训练数据集不足和眼部图像变异导致的模型泛化问题。实验结果表明，LEyes训练的模型在瞳孔和CR定位方面优于其他算法。 |

# 详细

[^1]: HeadEvolver：通过本地可学习网格变形实现文本到头部头像的转换

    HeadEvolver: Text to Head Avatars via Locally Learnable Mesh Deformation

    [https://arxiv.org/abs/2403.09326](https://arxiv.org/abs/2403.09326)

    通过可学习的局部网格变形技术，HeadEvolver框架可以通过文本引导生成高质量的头部头像，保留细节并支持编辑和动画。

    

    我们提出了HeadEvolver，一个新颖的框架，可以通过文本引导生成风格化的头部头像。HeadEvolver使用模板头部网格的本地可学习网格变形，生成高质量的数字资产，以实现保留细节的编辑和动画。为了解决全局变形中缺乏细粒度和语义感知本地形状控制的挑战，我们引入了可训练参数作为每个三角形的Jacobi矩阵的加权因子，以自适应地改变本地形状同时保持全局对应和面部特征。此外，为了确保来自不同视角的结果形状和外观的连贯性，我们使用预训练的图像扩散模型进行可微分渲染，并添加正则化项以在文本引导下优化变形。大量实验证明，我们的方法可以生成具有关节网格的多样化头部头像，可无缝编辑。

    arXiv:2403.09326v1 Announce Type: cross  Abstract: We present HeadEvolver, a novel framework to generate stylized head avatars from text guidance. HeadEvolver uses locally learnable mesh deformation from a template head mesh, producing high-quality digital assets for detail-preserving editing and animation. To tackle the challenges of lacking fine-grained and semantic-aware local shape control in global deformation through Jacobians, we introduce a trainable parameter as a weighting factor for the Jacobian at each triangle to adaptively change local shapes while maintaining global correspondences and facial features. Moreover, to ensure the coherence of the resulting shape and appearance from different viewpoints, we use pretrained image diffusion models for differentiable rendering with regularization terms to refine the deformation under text guidance. Extensive experiments demonstrate that our method can generate diverse head avatars with an articulated mesh that can be edited seaml
    
[^2]: 神经红移：随机网络并非随机函数

    Neural Redshift: Random Networks are not Random Functions

    [https://arxiv.org/abs/2403.02241](https://arxiv.org/abs/2403.02241)

    本论文研究了未经训练的随机权重网络，发现即使简单的MLPs也具有强烈的归纳偏见，不同于传统观点的是，NNs并不具有固有的“简单偏见”，而是依赖于组件的作用。

    

    我们对神经网络（NNs）的泛化能力的理解仍不完整。目前的解释基于梯度下降（GD）的隐含偏见，但无法解释梯度自由方法中模型的能力，也无法解释最近观察到的未经训练网络的简单偏见。本文寻找NNs中的其他泛化源。为了独立于GD理解体系结构提供的归纳偏见，我们研究未经训练的随机权重网络。即使是简单的MLPs也表现出强烈的归纳偏见：在权重空间中进行均匀抽样会产生一个非常偏向于复杂性的函数分布。但与常规智慧不同，NNs并不具有固有的“简单偏见”。这一特性取决于组件，如ReLU、残差连接和层归一化。可利用替代体系结构构建偏向于任何复杂性水平的偏见。Transformers也具有这一特性。

    arXiv:2403.02241v1 Announce Type: cross  Abstract: Our understanding of the generalization capabilities of neural networks (NNs) is still incomplete. Prevailing explanations are based on implicit biases of gradient descent (GD) but they cannot account for the capabilities of models from gradient-free methods nor the simplicity bias recently observed in untrained networks. This paper seeks other sources of generalization in NNs.   Findings. To understand the inductive biases provided by architectures independently from GD, we examine untrained, random-weight networks. Even simple MLPs show strong inductive biases: uniform sampling in weight space yields a very biased distribution of functions in terms of complexity. But unlike common wisdom, NNs do not have an inherent "simplicity bias". This property depends on components such as ReLUs, residual connections, and layer normalizations. Alternative architectures can be built with a bias for any level of complexity. Transformers also inher
    
[^3]: 大型语言模型逆向翻译防御对抗攻击

    Round Trip Translation Defence against Large Language Model Jailbreaking Attacks

    [https://arxiv.org/abs/2402.13517](https://arxiv.org/abs/2402.13517)

    往返翻译（RTT）方法是第一个专门设计用于抵御大型语言模型（LLMs）社交工程攻击的算法，成功地减少了多种攻击形式的成功率。

    

    大型语言模型（LLMs）容易受到社交工程攻击，这些攻击对人类具有可解释性，但需要LLMs具有高水平的理解能力才能抵抗。现有的防御措施最多只能缓解这些攻击的不到一半。为解决这一问题，我们提出了往返翻译（RTT）方法，这是第一个专门设计用于抵御LLMs社交工程攻击的算法。RTT会改写对抗性提示并推广表达的思想，使LLMs更容易检测出诱发有害行为。这种方法灵活、轻量且可转移至不同的LLMs。我们的防御成功地缓解了超过70%的Prompt Automatic Iterative Refinement (PAIR)攻击，这是目前我们所知最有效的防御。我们也是首次尝试缓解MathsAttack，并将其攻击成功率降低了近40%。我们的代码已公开发布。

    arXiv:2402.13517v1 Announce Type: cross  Abstract: Large language models (LLMs) are susceptible to social-engineered attacks that are human-interpretable but require a high level of comprehension for LLMs to counteract. Existing defensive measures can only mitigate less than half of these attacks at most. To address this issue, we propose the Round Trip Translation (RTT) method, the first algorithm specifically designed to defend against social-engineered attacks on LLMs. RTT paraphrases the adversarial prompt and generalizes the idea conveyed, making it easier for LLMs to detect induced harmful behavior. This method is versatile, lightweight, and transferrable to different LLMs. Our defense successfully mitigated over 70% of Prompt Automatic Iterative Refinement (PAIR) attacks, which is currently the most effective defense to the best of our knowledge. We are also the first to attempt mitigating the MathsAttack and reduced its attack success rate by almost 40%. Our code is publicly av
    
[^4]: 生成强化学习策略解释的实证研究

    On Generating Explanations for Reinforcement Learning Policies: An Empirical Study. (arXiv:2309.16960v1 [cs.AI])

    [http://arxiv.org/abs/2309.16960](http://arxiv.org/abs/2309.16960)

    本文通过引入一组线性时态逻辑（LTL）公式，介绍了一种生成强化学习策略解释的方法，并展示了其在模拟夺旗环境中的有效性。

    

    本文引入了一组设计用于提供策略解释的线性时态逻辑（LTL）公式。我们的重点是构建既阐明策略所实现的最终目标又阐明其执行过程中所维持的前提条件的解释。这些基于LTL的解释具有结构化表示，特别适用于局部搜索技术。通过模拟的夺旗环境，证明了我们提出的方法的有效性。论文最后提出了未来研究的建议方向。

    In this paper, we introduce a set of \textit{Linear Temporal Logic} (LTL) formulae designed to provide explanations for policies. Our focus is on crafting explanations that elucidate both the ultimate objectives accomplished by the policy and the prerequisites it upholds throughout its execution. These LTL-based explanations feature a structured representation, which is particularly well-suited for local-search techniques. The effectiveness of our proposed approach is illustrated through a simulated capture the flag environment. The paper concludes with suggested directions for future research.
    
[^5]: LEyes：一种轻量级深度学习眼动跟踪框架，使用合成眼部图像

    LEyes: A Lightweight Framework for Deep Learning-Based Eye Tracking using Synthetic Eye Images. (arXiv:2309.06129v1 [cs.CV])

    [http://arxiv.org/abs/2309.06129](http://arxiv.org/abs/2309.06129)

    本研究提出了一种名为LEyes的轻量级深度学习眼动跟踪框架，利用合成眼部图像进行训练，解决了由于训练数据集不足和眼部图像变异导致的模型泛化问题。实验结果表明，LEyes训练的模型在瞳孔和CR定位方面优于其他算法。

    

    深度学习已经加强了凝视估计技术，但实际部署受到不足的训练数据集的限制。眼部图像的硬件引起的变异以及记录的参与者之间固有的生物差异会导致特征和像素级别的差异，阻碍了在特定数据集上训练的模型的泛化能力。虚拟数据集可以是一个解决方案，但创建虚拟数据集既需要时间又需要资源。为了解决这个问题，我们提出了一个名为Light Eyes or "LEyes"的框架，与传统的逼真方法不同，LEyes仅模拟视频眼动跟踪所需的关键图像特征。LEyes便于在多样化的凝视估计任务上训练神经网络。我们证明，使用LEyes训练的模型在眼睛瞳孔和CR定位方面优于其他最先进的算法。

    Deep learning has bolstered gaze estimation techniques, but real-world deployment has been impeded by inadequate training datasets. This problem is exacerbated by both hardware-induced variations in eye images and inherent biological differences across the recorded participants, leading to both feature and pixel-level variance that hinders the generalizability of models trained on specific datasets. While synthetic datasets can be a solution, their creation is both time and resource-intensive. To address this problem, we present a framework called Light Eyes or "LEyes" which, unlike conventional photorealistic methods, only models key image features required for video-based eye tracking using simple light distributions. LEyes facilitates easy configuration for training neural networks across diverse gaze-estimation tasks. We demonstrate that models trained using LEyes outperform other state-of-the-art algorithms in terms of pupil and CR localization across well-known datasets. In addit
    

