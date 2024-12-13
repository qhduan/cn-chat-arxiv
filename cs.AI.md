# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ProSwitch: Knowledge-Guided Language Model Fine-Tuning to Generate Professional and Non-Professional Styled Text](https://arxiv.org/abs/2403.09131) | ProSwitch通过知识引导的指令微调，在专业和非专业风格之间生成文本，并在专业性评估和质量评估方面表现出优越性。 |
| [^2] | [LLMs with Chain-of-Thought Are Non-Causal Reasoners](https://arxiv.org/abs/2402.16048) | 本文探讨了大型语言模型在推理过程中思维链条（CoT）的作用，发现LLMs在答案生成过程中与人类推理存在差异，相关因素包括语境学习、有监督微调以及对人类反馈的强化学习。 |
| [^3] | [Flexible Physical Camouflage Generation Based on a Differential Approach](https://arxiv.org/abs/2402.13575) | 该研究引入了一种新颖的神经渲染方法，名为FPA，通过学习对抗模式并结合特殊设计的对抗损失和隐蔽约束损失，可以生成物理世界中具有对抗性和隐蔽性质的伪装。 |
| [^4] | [ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models](https://arxiv.org/abs/2402.13516) | 本文介绍了一种名为"ProSparse"的有效稀疏化方法，以推动大型语言模型实现更高的激活稀疏性而不降低模型性能 |
| [^5] | [Learn To be Efficient: Build Structured Sparsity in Large Language Models](https://arxiv.org/abs/2402.06126) | 本文通过引入一种新的算法"Learn-To-be-Efficient(LTE)"，提出了在大型语言模型(LLM)中构建结构化稀疏性的方法。该方法通过训练高效意识的LLM学习激活更少的神经元，取得更好的稀疏性和性能折衷。 |
| [^6] | [Reinforcement Learning as a Catalyst for Robust and Fair Federated Learning: Deciphering the Dynamics of Client Contributions](https://arxiv.org/abs/2402.05541) | 本研究提出了一个新的框架——强化联邦学习（RFL），通过利用深度强化学习自适应地优化客户贡献的聚合过程，以增强模型鲁棒性和在非相同分布环境下参与者之间的公平性。 |
| [^7] | [Respect the model: Fine-grained and Robust Explanation with Sharing Ratio Decomposition](https://arxiv.org/abs/2402.03348) | 本论文提出了一种称为共享比例分解(SRD)的新颖解释方法，真实地反映了模型的推理过程，并在解释方面显著提高了鲁棒性。通过采用向量视角和考虑滤波器之间的复杂非线性交互，以及引入仅激活模式预测(APOP)方法，可以重新定义相关性并强调非活跃神经元的重要性。 |
| [^8] | [Hierarchical Fashion Design with Multi-stage Diffusion Models.](http://arxiv.org/abs/2401.07450) | 本论文提出了一种名为HieraFashDiff的新型时尚设计方法，它使用多级扩散模型实现了从高级设计概念到低级服装属性的分层设计和编辑，解决了当前在时尚设计中的挑战。 |
| [^9] | [OneAdapt: Fast Adaptation for Deep Learning Applications via Backpropagation.](http://arxiv.org/abs/2310.02422) | OneAdapt通过梯度上升策略来实现快速自适应，满足了深度学习应用在配置参数方面的三个要求。 |
| [^10] | [STARC: A General Framework For Quantifying Differences Between Reward Functions.](http://arxiv.org/abs/2309.15257) | 这篇论文提出了一个通用框架（STARC），用于评估奖励函数之间的差异，填补了奖励学习理论基础的空白。 |
| [^11] | [Provably Efficient Learning in Partially Observable Contextual Bandit.](http://arxiv.org/abs/2308.03572) | 本文研究了在部分可观察情境轮盘赌中的转移学习问题，提出了一种通过优化问题识别行为和奖励因果效应的方法，并利用因果约束来改进轮盘赌算法。 |

# 详细

[^1]: ProSwitch：知识引导的语言模型微调，生成专业和非专业风格的文本

    ProSwitch: Knowledge-Guided Language Model Fine-Tuning to Generate Professional and Non-Professional Styled Text

    [https://arxiv.org/abs/2403.09131](https://arxiv.org/abs/2403.09131)

    ProSwitch通过知识引导的指令微调，在专业和非专业风格之间生成文本，并在专业性评估和质量评估方面表现出优越性。

    

    大语言模型（LLMs）在各种语言应用中表现出有效性，包括文本摘要和可控文本生成。然而，关于它们通过微调在不同风格间切换的能力的研究仍未被充分探讨。本研究聚焦于文本专业性，并引入了一种新颖的方法，名为ProSwitch，通过知识引导的指令微调，使语言模型具备生成专业和非专业回复的能力。ProSwitch分为三个阶段：数据准备，用于收集领域知识和训练语料库；指令微调，用于优化带有多种指令格式的语言模型；全面评估，用于评估生成文本的专业性区分能力和基于参考的质量。 ProSwitch相对于通用和专门语言模型的比较分析显示了我们的方法的优越性。

    arXiv:2403.09131v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated efficacy in various linguistic applications, including text summarization and controlled text generation. However, studies into their capacity of switching between styles via fine-tuning remain underexplored. This study concentrates on textual professionalism and introduces a novel methodology, named ProSwitch, which equips a language model with the ability to produce both professional and non-professional responses through knowledge-guided instruction tuning. ProSwitch unfolds across three phases: data preparation for gathering domain knowledge and training corpus; instruction tuning for optimizing language models with multiple levels of instruction formats; and comprehensive evaluation for assessing the professionalism discrimination and reference-based quality of generated text. Comparative analysis of ProSwitch against both general and specialized language models reveals that our appro
    
[^2]: LLMs带有思维链条是非因果推理者

    LLMs with Chain-of-Thought Are Non-Causal Reasoners

    [https://arxiv.org/abs/2402.16048](https://arxiv.org/abs/2402.16048)

    本文探讨了大型语言模型在推理过程中思维链条（CoT）的作用，发现LLMs在答案生成过程中与人类推理存在差异，相关因素包括语境学习、有监督微调以及对人类反馈的强化学习。

    

    本文探讨了大型语言模型（LLMs）推理中思维链条（CoT）的作用。尽管它有改善任务性能的潜力，但我们的分析揭示了在LLMs中正确答案跟随不正确CoTs的频率及反之。我们采用因果分析来评估CoTs/指令与LLMs答案之间的因果关系，揭示LLMs近似的结构因果模型（SCM）。通过比较暗示SCM与人类推理的SCM，我们突显了LLM和人类推理过程之间的差异。我们进一步研究了影响暗示SCM因果结构的因素，揭示了语境学习、有监督微调以及对人类反馈的强化学习显著影响因果关系。我们在https://github.com/StevenZHB/CoT_Causal_Analysis发布了代码和结果。

    arXiv:2402.16048v1 Announce Type: cross  Abstract: This paper explores the role of the Chain of Thought (CoT) in Large Language Models (LLMs) reasoning. Despite its potential to improve task performance, our analysis reveals a surprising frequency of correct answers following incorrect CoTs and vice versa. We employ causal analysis to assess the cause-effect relationship between CoTs/instructions and answers in LLMs, uncovering the Structural Causal Model (SCM) that LLMs approximate. By comparing the implied SCM with that of human reasoning, we highlight discrepancies between LLM and human reasoning processes. We further examine the factors influencing the causal structure of the implied SCM, revealing that in-context learning, supervised fine-tuning, and reinforcement learning on human feedback significantly impact the causal relations. We release the code and results at https://github.com/StevenZHB/CoT_Causal_Analysis.
    
[^3]: 基于差异方法的灵活物理伪装生成

    Flexible Physical Camouflage Generation Based on a Differential Approach

    [https://arxiv.org/abs/2402.13575](https://arxiv.org/abs/2402.13575)

    该研究引入了一种新颖的神经渲染方法，名为FPA，通过学习对抗模式并结合特殊设计的对抗损失和隐蔽约束损失，可以生成物理世界中具有对抗性和隐蔽性质的伪装。

    

    这项研究介绍了一种新的神经渲染方法，专门针对对抗伪装，在广泛的三维渲染框架内进行了定制。我们的方法，名为FPA，通过忠实地模拟光照条件和材料变化，确保在三维目标上对纹理进行微妙而逼真的表现。为了实现这一目标，我们采用一种生成方法，从扩散模型中学习对抗模式。这涉及将一个特别设计的对抗损失和隐蔽约束损失结合在一起，以确保伪装在物理世界中的对抗性和隐蔽性质。此外，我们展示了所提出的伪装在贴纸模式下的有效性，展示了其覆盖目标而不影响对抗信息的能力。通过实证和物理实验，FPA在攻击成功率和可转移性方面表现出很强的性能。

    arXiv:2402.13575v1 Announce Type: cross  Abstract: This study introduces a novel approach to neural rendering, specifically tailored for adversarial camouflage, within an extensive 3D rendering framework. Our method, named FPA, goes beyond traditional techniques by faithfully simulating lighting conditions and material variations, ensuring a nuanced and realistic representation of textures on a 3D target. To achieve this, we employ a generative approach that learns adversarial patterns from a diffusion model. This involves incorporating a specially designed adversarial loss and covert constraint loss to guarantee the adversarial and covert nature of the camouflage in the physical world. Furthermore, we showcase the effectiveness of the proposed camouflage in sticker mode, demonstrating its ability to cover the target without compromising adversarial information. Through empirical and physical experiments, FPA exhibits strong performance in terms of attack success rate and transferabili
    
[^4]: ProSparse: 引入和增强大型语言模型内部激活稀疏性

    ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models

    [https://arxiv.org/abs/2402.13516](https://arxiv.org/abs/2402.13516)

    本文介绍了一种名为"ProSparse"的有效稀疏化方法，以推动大型语言模型实现更高的激活稀疏性而不降低模型性能

    

    Activation sparsity指的是激活输出中存在许多弱贡献元素。作为使用ReLU激活函数的模型的普遍属性，已被证明是提高模型推理效率的一种有前途的范例。然而，大多数大型语言模型（LLMs）采用了没有内在激活稀疏性的激活函数（例如GELU和Swish）。一些最近的努力尝试引入ReLU或其变体作为替代激活函数，以帮助LLMs实现激活稀疏性和推理加速，但很少能同时获得高稀疏度和可比较的模型性能。本文介绍了一种名为"ProSparse"的有效稀疏化方法，以推动LLMs实现更高的激活稀疏性而不降低模型性能。具体来说，将LLMs的激活函数替换为ReLU后，ProSparse采用渐进稀疏正则化

    arXiv:2402.13516v1 Announce Type: cross  Abstract: Activation sparsity refers to the existence of considerable weakly-contributed elements among activation outputs. As a prevalent property of the models using the ReLU activation function, it has been proven a promising paradigm to boost model inference efficiency. Nevertheless, most large language models (LLMs) adopt activation functions without intrinsic activation sparsity (e.g., GELU and Swish). Some recent efforts have explored introducing ReLU or its variants as the substitutive activation function to help LLMs achieve activation sparsity and inference acceleration, but few can simultaneously obtain high sparsity and comparable model performance. This paper introduces an effective sparsification method named "ProSparse" to push LLMs for higher activation sparsity without decreasing model performance. Specifically, after substituting the activation function of LLMs with ReLU, ProSparse adopts progressive sparsity regularization wit
    
[^5]: 学习变得高效：在大型语言模型中构建结构化稀疏性

    Learn To be Efficient: Build Structured Sparsity in Large Language Models

    [https://arxiv.org/abs/2402.06126](https://arxiv.org/abs/2402.06126)

    本文通过引入一种新的算法"Learn-To-be-Efficient(LTE)"，提出了在大型语言模型(LLM)中构建结构化稀疏性的方法。该方法通过训练高效意识的LLM学习激活更少的神经元，取得更好的稀疏性和性能折衷。

    

    大型语言模型(LLM)以其十亿级参数取得了显著的成功，但它们产生了高昂的推理开销。在LLM中出现的激活稀疏性为通过仅涉及部分参数进行推理提供了一种自然的方法来减少这种成本。现有方法只关注利用这种自然形成的激活稀疏性，忽视了进一步放大这种固有稀疏性的潜力。本文中，我们假设LLM可以通过实现更结构化的激活稀疏性来学习高效。为实现这一目标，我们引入了一种新颖的算法"Learn-To-be-Efficient(LTE)", 旨在训练高效意识的LLM学习激活更少的神经元，并在稀疏性和性能之间取得更好的折衷。此外，与主要关注基于ReLU模型的SOTA MoEfication方法不同，LTE还可以应用于像GPT和LLaMA这样具有软激活函数的LLM。我们在四个模型和十一个数据集上评估了LTE。

    Large Language Models (LLMs) have achieved remarkable success with their billion-level parameters, yet they incur high inference overheads. The emergence of activation sparsity in LLMs provides a natural approach to reduce this cost by involving only parts of the parameters for inference. Existing methods only focus on utilizing this naturally formed activation sparsity, overlooking the potential for further amplifying this inherent sparsity. In this paper, we hypothesize that LLMs can learn to be efficient by achieving more structured activation sparsity.To achieve this, we introduce a novel algorithm, Learn-To-be-Efficient (LTE), designed to train efficiency-aware LLMs to learn to activate fewer neurons and achieve a better trade-off between sparsity and performance. Furthermore, unlike SOTA MoEfication methods, which mainly focus on ReLU-based models, LTE can also be applied to LLMs like GPT and LLaMA with soft activation functions. We evaluate LTE on four models and eleven datasets
    
[^6]: 强化学习作为鲁棒和公平联邦学习的催化剂：解密客户贡献动力学

    Reinforcement Learning as a Catalyst for Robust and Fair Federated Learning: Deciphering the Dynamics of Client Contributions

    [https://arxiv.org/abs/2402.05541](https://arxiv.org/abs/2402.05541)

    本研究提出了一个新的框架——强化联邦学习（RFL），通过利用深度强化学习自适应地优化客户贡献的聚合过程，以增强模型鲁棒性和在非相同分布环境下参与者之间的公平性。

    

    最近在联邦学习（FL）方面的进展产生了模型，通过在多个分散的设备或系统上训练来保护用户隐私并保留本地数据样本。然而，这些策略经常忽视统计异质性和对敌对攻击的脆弱性所带来的困难，这些因素会降低模型的鲁棒性和公平性。个性化的FL策略可以通过调整模型来适应个别客户的特点，但往往忽视了服务器端聚合的脆弱性。为了解决这些问题，我们提出了强化联邦学习（RFL），这是一个利用深度强化学习来自适应优化聚合过程中客户贡献的新框架，从而增强恶意客户下的模型鲁棒性和参与者之间的公平性在非相同分布环境下。为了实现这一目标，我们提出了一种细致的方法，其中包括基于深度确定性策略梯度算法的协同训练，以优化客户贡献的过程。

    Recent advancements in federated learning (FL) have produced models that retain user privacy by training across multiple decentralized devices or systems holding local data samples. However, these strategies often neglect the inherent challenges of statistical heterogeneity and vulnerability to adversarial attacks, which can degrade model robustness and fairness. Personalized FL strategies offer some respite by adjusting models to fit individual client profiles, yet they tend to neglect server-side aggregation vulnerabilities. To address these issues, we propose Reinforcement Federated Learning (RFL), a novel framework that leverages deep reinforcement learning to adaptively optimize client contribution during aggregation, thereby enhancing both model robustness against malicious clients and fairness across participants under non-identically distributed settings. To achieve this goal, we propose a meticulous approach involving a Deep Deterministic Policy Gradient-based algorithm for co
    
[^7]: 尊重模型: 细粒度且鲁棒的解释与共享比例分解

    Respect the model: Fine-grained and Robust Explanation with Sharing Ratio Decomposition

    [https://arxiv.org/abs/2402.03348](https://arxiv.org/abs/2402.03348)

    本论文提出了一种称为共享比例分解(SRD)的新颖解释方法，真实地反映了模型的推理过程，并在解释方面显著提高了鲁棒性。通过采用向量视角和考虑滤波器之间的复杂非线性交互，以及引入仅激活模式预测(APOP)方法，可以重新定义相关性并强调非活跃神经元的重要性。

    

    对现有的解释方法能否真实阐明模型决策过程的真实性提出了质疑。现有方法偏离了对模型的忠实表达，因此容易受到对抗性攻击的影响。为了解决这个问题，我们提出了一种新颖的可解释性人工智能(XAI)方法，称为SRD(共享比例分解)，它真实地反映了模型的推理过程，从而显著提高了解释的鲁棒性。与传统的神经元级别强调不同，我们采用向量视角来考虑滤波器之间复杂的非线性交互。我们还引入了一个有趣的观察，称为仅激活模式预测(APOP)，让我们强调非活跃神经元的重要性，并重新定义相关性，包括活跃和非活跃神经元的所有相关信息。我们的方法SRD允许递归分解一个点特征向量(PFV)。

    The truthfulness of existing explanation methods in authentically elucidating the underlying model's decision-making process has been questioned. Existing methods have deviated from faithfully representing the model, thus susceptible to adversarial attacks. To address this, we propose a novel eXplainable AI (XAI) method called SRD (Sharing Ratio Decomposition), which sincerely reflects the model's inference process, resulting in significantly enhanced robustness in our explanations. Different from the conventional emphasis on the neuronal level, we adopt a vector perspective to consider the intricate nonlinear interactions between filters. We also introduce an interesting observation termed Activation-Pattern-Only Prediction (APOP), letting us emphasize the importance of inactive neurons and redefine relevance encapsulating all relevant information including both active and inactive neurons. Our method, SRD, allows for the recursive decomposition of a Pointwise Feature Vector (PFV), pr
    
[^8]: 带有多级扩散模型的分层时尚设计

    Hierarchical Fashion Design with Multi-stage Diffusion Models. (arXiv:2401.07450v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2401.07450](http://arxiv.org/abs/2401.07450)

    本论文提出了一种名为HieraFashDiff的新型时尚设计方法，它使用多级扩散模型实现了从高级设计概念到低级服装属性的分层设计和编辑，解决了当前在时尚设计中的挑战。

    

    跨模态时尚合成和编辑通过自动生成和局部修改设计草图，为时尚设计师提供智能支持。尽管当前的扩散模型在图像合成方面表现出了可靠的稳定性和可控性，但在从抽象的设计元素中生成时尚设计和精细编辑方面仍面临重大挑战。高级设计概念，例如办公室、商务和派对，形成了抽象的感官表达方式，而袖长、领型和裤长等可衡量的方面被视为服装的低级属性。使用冗长的文字描述来控制和编辑时尚图像存在困难。在本文中，我们提出了一种名为HieraFashDiff的新型时尚设计方法，它使用共享的多级扩散模型，将高级设计概念和低级服装属性融入到分层结构中。具体而言，我们将输入文本分为不同的层次，并将其输入到多级扩散模型中。

    Cross-modal fashion synthesis and editing offer intelligent support to fashion designers by enabling the automatic generation and local modification of design drafts.While current diffusion models demonstrate commendable stability and controllability in image synthesis,they still face significant challenges in generating fashion design from abstract design elements and fine-grained editing.Abstract sensory expressions, \eg office, business, and party, form the high-level design concepts, while measurable aspects like sleeve length, collar type, and pant length are considered the low-level attributes of clothing.Controlling and editing fashion images using lengthy text descriptions poses a difficulty.In this paper, we propose HieraFashDiff,a novel fashion design method using the shared multi-stage diffusion model encompassing high-level design concepts and low-level clothing attributes in a hierarchical structure.Specifically, we categorized the input text into different levels and fed 
    
[^9]: OneAdapt：通过反向传播实现深度学习应用的快速自适应

    OneAdapt: Fast Adaptation for Deep Learning Applications via Backpropagation. (arXiv:2310.02422v1 [cs.LG])

    [http://arxiv.org/abs/2310.02422](http://arxiv.org/abs/2310.02422)

    OneAdapt通过梯度上升策略来实现快速自适应，满足了深度学习应用在配置参数方面的三个要求。

    

    深度学习在流媒体数据的推断方面已经普及，如视频中的目标检测、LiDAR数据和音频波形中的文本提取。为了实现高推断准确性，这些应用通常需要大量的网络带宽来收集高保真数据，并且需要广泛的GPU资源来运行深度神经网络(DNN)。尽管通过优化配置参数（如视频分辨率和帧率）可以大大减少对网络带宽和GPU资源的需求，但目前的自适应技术无法同时满足三个要求：（i）以最小的额外GPU或带宽开销来自适应配置；（ii）基于数据对最终DNN的准确性的影响来达到接近最优的决策；（iii）针对一系列配置参数进行自适应。本文提出了OneAdapt，通过利用梯度上升策略来自适应配置参数，满足了这些要求。关键思想是充分利用DNN的不同

    Deep learning inference on streaming media data, such as object detection in video or LiDAR feeds and text extraction from audio waves, is now ubiquitous. To achieve high inference accuracy, these applications typically require significant network bandwidth to gather high-fidelity data and extensive GPU resources to run deep neural networks (DNNs). While the high demand for network bandwidth and GPU resources could be substantially reduced by optimally adapting the configuration knobs, such as video resolution and frame rate, current adaptation techniques fail to meet three requirements simultaneously: adapt configurations (i) with minimum extra GPU or bandwidth overhead; (ii) to reach near-optimal decisions based on how the data affects the final DNN's accuracy, and (iii) do so for a range of configuration knobs. This paper presents OneAdapt, which meets these requirements by leveraging a gradient-ascent strategy to adapt configuration knobs. The key idea is to embrace DNNs' different
    
[^10]: STARC:评估奖励函数之间差异的通用框架

    STARC: A General Framework For Quantifying Differences Between Reward Functions. (arXiv:2309.15257v1 [cs.LG])

    [http://arxiv.org/abs/2309.15257](http://arxiv.org/abs/2309.15257)

    这篇论文提出了一个通用框架（STARC），用于评估奖励函数之间的差异，填补了奖励学习理论基础的空白。

    

    为了使用强化学习解决任务，需要将任务的目标形式化为奖励函数。然而，对于许多现实世界的任务来说，手动指定一个永不激励不良行为的奖励函数非常困难。因此，使用奖励学习算法来从数据中学习奖励函数变得越来越流行。然而，奖励学习的理论基础尚未完善。特别地，通常不知道给定的奖励学习算法在高概率下是否会学习到一个安全优化的奖励函数。这意味着奖励学习算法通常必须经过经验评估，这是昂贵的，并且很难预测其失效模式。其中一个阻碍获得更好理论保证的障碍是缺乏较好的方法来量化奖励函数之间的差异。在本文中，我们提供了一种解决方案。

    In order to solve a task using reinforcement learning, it is necessary to first formalise the goal of that task as a reward function. However, for many real-world tasks, it is very difficult to manually specify a reward function that never incentivises undesirable behaviour. As a result, it is increasingly popular to use reward learning algorithms, which attempt to learn a reward function from data. However, the theoretical foundations of reward learning are not yet well-developed. In particular, it is typically not known when a given reward learning algorithm with high probability will learn a reward function that is safe to optimise. This means that reward learning algorithms generally must be evaluated empirically, which is expensive, and that their failure modes are difficult to predict in advance. One of the roadblocks to deriving better theoretical guarantees is the lack of good methods for quantifying the difference between reward functions. In this paper we provide a solution t
    
[^11]: 在部分可观察情境轮盘赌中的可证效率学习

    Provably Efficient Learning in Partially Observable Contextual Bandit. (arXiv:2308.03572v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2308.03572](http://arxiv.org/abs/2308.03572)

    本文研究了在部分可观察情境轮盘赌中的转移学习问题，提出了一种通过优化问题识别行为和奖励因果效应的方法，并利用因果约束来改进轮盘赌算法。

    

    本文研究了在部分可观察情境轮盘赌中的转移学习问题，其中代理人仅有来自其他代理人的有限知识，并且对隐藏的混淆因素只有部分信息。我们将该问题转化为通过优化问题来识别或部分识别行为和奖励之间的因果效应。为了解决这些优化问题，我们将未知分布的原始功能约束离散化为线性约束，并通过顺序解线性规划来采样兼容的因果模型，以考虑估计误差得到因果约束。我们的采样算法为适当的采样分布提供了理想的收敛结果。然后，我们展示了如何将因果约束应用于改进经典的轮盘赌算法，并以行动集和函数空间规模为参考改变了遗憾值。值得注意的是，在允许我们处理一般情境分布的函数逼近任务中

    In this paper, we investigate transfer learning in partially observable contextual bandits, where agents have limited knowledge from other agents and partial information about hidden confounders. We first convert the problem to identifying or partially identifying causal effects between actions and rewards through optimization problems. To solve these optimization problems, we discretize the original functional constraints of unknown distributions into linear constraints, and sample compatible causal models via sequentially solving linear programmings to obtain causal bounds with the consideration of estimation error. Our sampling algorithms provide desirable convergence results for suitable sampling distributions. We then show how causal bounds can be applied to improving classical bandit algorithms and affect the regrets with respect to the size of action sets and function spaces. Notably, in the task with function approximation which allows us to handle general context distributions
    

