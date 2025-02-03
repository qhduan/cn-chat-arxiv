# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Can LLMs Separate Instructions From Data? And What Do We Even Mean By That?](https://arxiv.org/abs/2403.06833) | 本研究提出了一种形式化的度量来量化指令与数据分离现象，以及一种可以从模型黑盒输出计算的经验变量，并引入了新数据集SEP，用于评估 |
| [^2] | [SecGPT: An Execution Isolation Architecture for LLM-Based Systems](https://arxiv.org/abs/2403.04960) | 提出了一种面向LLM系统的执行隔离架构SecGPT，旨在解决第三方应用程序执行所引发的安全和隐私问题 |
| [^3] | [Private Fine-tuning of Large Language Models with Zeroth-order Optimization.](http://arxiv.org/abs/2401.04343) | 引入了DP-ZO，一种通过私有化零阶优化来保护大型语言模型训练数据隐私的方法。 |
| [^4] | [Theoretical guarantees on the best-of-n alignment policy.](http://arxiv.org/abs/2401.01879) | 该论文研究了对齐生成模型的最佳n对齐策略，并证明了之前文献中的某个分析表达式是错误的。研究者们提出了一个新的KL散度估计方法，并通过实验证明其有效性。 |

# 详细

[^1]: LLMs能够将指令与数据分离吗？我们具体指的是什么？

    Can LLMs Separate Instructions From Data? And What Do We Even Mean By That?

    [https://arxiv.org/abs/2403.06833](https://arxiv.org/abs/2403.06833)

    本研究提出了一种形式化的度量来量化指令与数据分离现象，以及一种可以从模型黑盒输出计算的经验变量，并引入了新数据集SEP，用于评估

    

    arXiv:2403.06833v1 公告类型: 跨 针对大型语言模型（LLMs）进行调节指令的技术取得了突破性的成果，为许多实际应用打开了无数新可能。然而，LLMs缺乏其他计算机科学领域已建立为规范的基本安全特性，比如指令与数据之间的分离，导致它们发生故障或易受第三方操控和干扰（例如通过间接提示/命令注入）。更糟糕的是，迄今为止，甚至没有确切定义这种分离究竟意味着什么以及如何测试其违反情况。本研究旨在填补这一空白。我们引入了一个正式的指标来量化指令与数据分离现象，以及一个可以从模型的黑盒输出计算的经验变量。我们还介绍了一个新的数据集SEP（应该执行还是处理？），该数据集允许评估

    arXiv:2403.06833v1 Announce Type: cross  Abstract: Instruction-tuned Large Language Models (LLMs) have achieved breakthrough results, opening countless new possibilities for many practical applications. However, LLMs lack elementary safety features that are established norms in other areas of computer science, such as the separation between instructions and data, causing them to malfunction or rendering them vulnerable to manipulation and interference by third parties e.g., via indirect prompt/command injection. Even worse, so far, there is not even an established definition of what precisely such a separation would mean and how its violation could be tested. In this work, we aim to close this gap. We introduce a formal measure to quantify the phenomenon of instruction-data separation as well as an empirical variant of the measure that can be computed from a model`s black-box outputs. We also introduce a new dataset, SEP (Should it be Executed or Processed?), which allows estimating th
    
[^2]: SecGPT：一种面向基于LLM系统的执行隔离架构

    SecGPT: An Execution Isolation Architecture for LLM-Based Systems

    [https://arxiv.org/abs/2403.04960](https://arxiv.org/abs/2403.04960)

    提出了一种面向LLM系统的执行隔离架构SecGPT，旨在解决第三方应用程序执行所引发的安全和隐私问题

    

    大型语言模型（LLMs）被扩展为系统，如ChatGPT，已经开始支持第三方应用程序。这些LLM应用程序利用LLMs的事实上基于自然语言的自动执行范式：即，应用程序及其交互是用自然语言定义的，提供对用户数据的访问，并被允许自由地相互交互以及与系统互动。这些LLM应用程序生态系统类似于早期计算平台的设置，在那里应用程序和系统之间缺乏足够的隔离。由于第三方应用程序可能不可信，并且受自然语言界面的不精确性加剧，当前的设计会为用户带来安全和隐私风险。在本文中，我们提出了SecGPT，一种面向LLM系统的架构，旨在缓解由第三方应用程序执行引起的安全性和隐私问题。SecGPT的关键思想是隔离应用程序的执行和更多的预

    arXiv:2403.04960v1 Announce Type: cross  Abstract: Large language models (LLMs) extended as systems, such as ChatGPT, have begun supporting third-party applications. These LLM apps leverage the de facto natural language-based automated execution paradigm of LLMs: that is, apps and their interactions are defined in natural language, provided access to user data, and allowed to freely interact with each other and the system. These LLM app ecosystems resemble the settings of earlier computing platforms, where there was insufficient isolation between apps and the system. Because third-party apps may not be trustworthy, and exacerbated by the imprecision of the natural language interfaces, the current designs pose security and privacy risks for users. In this paper, we propose SecGPT, an architecture for LLM-based systems that aims to mitigate the security and privacy issues that arise with the execution of third-party apps. SecGPT's key idea is to isolate the execution of apps and more pre
    
[^3]: 私有零阶优化的大型语言模型的私有微调

    Private Fine-tuning of Large Language Models with Zeroth-order Optimization. (arXiv:2401.04343v1 [cs.LG])

    [http://arxiv.org/abs/2401.04343](http://arxiv.org/abs/2401.04343)

    引入了DP-ZO，一种通过私有化零阶优化来保护大型语言模型训练数据隐私的方法。

    

    在私有数据集上对大型预训练模型进行微调可能会存在违反隐私的风险。差分隐私是一种通过强制算法稳定性来减轻隐私风险的框架。DP-SGD可以以保护隐私的方式训练具有私有数据的模型，但会带来性能损失和重大工程挑战。我们引入了DP-ZO，一种通过私有化零阶优化来保护训练数据隐私的大型语言模型微调方法。我们的方法设计的一个关键见解是，我们使用的零阶算法SPSA中的梯度方向始终是随机的，而仅依赖于私有数据的信息是步长，即一个标量。因此，我们只需要对标量步长进行隐私处理，这是存储效率高的方法。DP-ZO可以使用拉普拉斯噪声或高斯噪声来实现，在不同任务之间提供了隐私和效用之间的强大权衡。

    Fine-tuning large pretrained models on private datasets may run the risk of violating privacy. Differential privacy is a framework for mitigating privacy risks by enforcing algorithmic stability. DP-SGD enables training models with private data in a privacy-preserving manner, but raises new obstacles in the form of performance loss and significant engineering challenges. We introduce DP-ZO, a new method for fine-tuning large language models that preserves the privacy of training data by privatizing zeroth-order optimization. A key insight into the design of our method is that the direction of the gradient in SPSA, the zeroth-order algorithm we use, is always random and the only information that depends on private data is the step size, i.e., a scalar. Therefore, we only need to privatize the scalar step size, which is memory-efficient. DP-ZO, which can be instantiated with either Laplace or Gaussian noise, provides a strong privacy-utility trade-off across different tasks, and model si
    
[^4]: 关于最佳n对齐策略的理论保证

    Theoretical guarantees on the best-of-n alignment policy. (arXiv:2401.01879v1 [cs.LG])

    [http://arxiv.org/abs/2401.01879](http://arxiv.org/abs/2401.01879)

    该论文研究了对齐生成模型的最佳n对齐策略，并证明了之前文献中的某个分析表达式是错误的。研究者们提出了一个新的KL散度估计方法，并通过实验证明其有效性。

    

    一个简单有效的生成模型对齐方法是最佳n对齐策略，该策略从一个基本策略中抽取n个样本，并根据奖励函数对它们进行排序，选择排名最高的样本。文献中常用的分析表达式声称最佳n对齐策略与基本策略之间的KL散度等于$\log (n) (n-1)/n$。我们证明了该论断的不正确性，并展示了它只是实际KL散度的一个上界。我们还研究了在不同情况下该上界的紧致性。最后，我们提出了一种新的KL散度估计方法，并通过几个例子的实验证明它能提供一个紧致的近似。

    A simple and effective method for the alignment of generative models is the best-of-$n$ policy, where $n$ samples are drawn from a base policy, and ranked based on a reward function, and the highest ranking one is selected. A commonly used analytical expression in the literature claims that the KL divergence between the best-of-$n$ policy and the base policy is equal to $\log (n) (n-1)/n.$ We disprove the validity of this claim, and show that it is an upper bound on the actual KL divergence. We also explore the tightness of this upper bound in different regimes. Finally, we propose a new estimator for the KL divergence and empirically show that it provides a tight approximation through a few examples.
    

