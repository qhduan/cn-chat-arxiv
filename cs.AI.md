# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Foundation Reinforcement Learning: towards Embodied Generalist Agents with Foundation Prior Assistance.](http://arxiv.org/abs/2310.02635) | 本研究提出了一种基于具身基础先验的基础强化学习框架，通过加速训练过程来提高样本效率。 |
| [^2] | [Basic syntax from speech: Spontaneous concatenation in unsupervised deep neural networks.](http://arxiv.org/abs/2305.01626) | 该论文提出了一种基于语音的完全无监督的方法，可以直接从原始语音中建立基础语法模型。作者发现，在基于声音的单词记录上训练的卷积神经网络可以自发连接两个或三个单词，并且可以学会将单词嵌入到新的未见过的单词组合中，这是之前未报道的属性，这一发现对我们理解神经网络的学习方式和建立从原始声学输入中的语法及其演化的模型都有重要的意义。 |

# 详细

[^1]: 强化学习基础：朝向具有基础先验辅助的具身通用智能体

    Foundation Reinforcement Learning: towards Embodied Generalist Agents with Foundation Prior Assistance. (arXiv:2310.02635v1 [cs.RO])

    [http://arxiv.org/abs/2310.02635](http://arxiv.org/abs/2310.02635)

    本研究提出了一种基于具身基础先验的基础强化学习框架，通过加速训练过程来提高样本效率。

    

    最近人们已经表明，从互联网规模的数据中进行大规模预训练是构建通用模型的关键，正如在NLP中所见。为了构建具身通用智能体，我们和许多其他研究者假设这种基础先验也是不可或缺的组成部分。然而，目前尚不清楚如何以适当的具体形式表示这些具身基础先验，以及它们应该如何在下游任务中使用。在本文中，我们提出了一组直观有效的具身先验，包括基础策略、价值和成功奖励。所提出的先验是基于目标条件的MDP。为了验证其有效性，我们实例化了一个由这些先验辅助的演员-评论家方法，称之为基础演员-评论家（FAC）。我们将我们的框架命名为基础强化学习（FRL），因为它完全依赖于具身基础先验来进行探索、学习和强化。FRL的好处有三个。(1)样本效率高。通过基础先验加速训练过程，减少样本使用量。

    Recently, people have shown that large-scale pre-training from internet-scale data is the key to building generalist models, as witnessed in NLP. To build embodied generalist agents, we and many other researchers hypothesize that such foundation prior is also an indispensable component. However, it is unclear what is the proper concrete form to represent those embodied foundation priors and how they should be used in the downstream task. In this paper, we propose an intuitive and effective set of embodied priors that consist of foundation policy, value, and success reward. The proposed priors are based on the goal-conditioned MDP. To verify their effectiveness, we instantiate an actor-critic method assisted by the priors, called Foundation Actor-Critic (FAC). We name our framework as Foundation Reinforcement Learning (FRL), since it completely relies on embodied foundation priors to explore, learn and reinforce. The benefits of FRL are threefold. (1) Sample efficient. With foundation p
    
[^2]: 基于语音的基础语法：自发联接的自监督深度神经网络

    Basic syntax from speech: Spontaneous concatenation in unsupervised deep neural networks. (arXiv:2305.01626v1 [cs.CL])

    [http://arxiv.org/abs/2305.01626](http://arxiv.org/abs/2305.01626)

    该论文提出了一种基于语音的完全无监督的方法，可以直接从原始语音中建立基础语法模型。作者发现，在基于声音的单词记录上训练的卷积神经网络可以自发连接两个或三个单词，并且可以学会将单词嵌入到新的未见过的单词组合中，这是之前未报道的属性，这一发现对我们理解神经网络的学习方式和建立从原始声学输入中的语法及其演化的模型都有重要的意义。

    

    语法的计算模型主要基于文本。本文提出了一种完全无监督的方法，可以直接从原始语音中建立基础语法模型。我们重点研究了最普遍和基本的语法特性之一——联接。我们介绍了自发联接现象：卷积神经网络(CNN)在个别单词的声学记录上训练时，开始产生输出，这些输出将两个甚至三个单词连接在一起，而不会接触到具有多个单词的输入数据。此外，训练两个单词的网络可以学习将单词嵌入到新的未见过的单词组合中。据我们所知，这是在生成对抗网络环境下训练的原始语音CNN以前未报道的属性，它不仅对我们理解这些体系结构的学习方式有影响，还对建立从原始声学输入中的语法及其演化的模型有影响。

    Computational models of syntax are predominantly text-based. Here we propose that basic syntax can be modeled directly from raw speech in a fully unsupervised way. We focus on one of the most ubiquitous and basic properties of syntax -- concatenation. We introduce spontaneous concatenation: a phenomenon where convolutional neural networks (CNNs) trained on acoustic recordings of individual words start generating outputs with two or even three words concatenated without ever accessing data with multiple words in the input. Additionally, networks trained on two words learn to embed words into novel unobserved word combinations. To our knowledge, this is a previously unreported property of CNNs trained on raw speech in the Generative Adversarial Network setting and has implications both for our understanding of how these architectures learn as well as for modeling syntax and its evolution from raw acoustic inputs.
    

