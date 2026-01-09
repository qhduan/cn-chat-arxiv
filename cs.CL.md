# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Understanding In-Context Learning with a Pelican Soup Framework](https://arxiv.org/abs/2402.10424) | 提出了一个鹈鹕汤框架，包括常识知识库、自然语言分类任务的形式化以及意义关联的概念，并建立了一个$O(1/T)$的上下文学习损失界限，能够解释对未见任务的泛化。 |

# 详细

[^1]: 使用鹈鹕汤框架理解上下文学习

    Understanding In-Context Learning with a Pelican Soup Framework

    [https://arxiv.org/abs/2402.10424](https://arxiv.org/abs/2402.10424)

    提出了一个鹈鹕汤框架，包括常识知识库、自然语言分类任务的形式化以及意义关联的概念，并建立了一个$O(1/T)$的上下文学习损失界限，能够解释对未见任务的泛化。

    

    许多现有关于自然语言处理中的上下文学习的理论分析是基于潜变量模型的，它们存在理论与实践之间的差距。我们旨在通过提出一个理论框架，即鹈鹕汤框架，来弥合这些差距。在这个框架中，我们引入了（1）常识知识库的概念，（2）自然语言分类任务的一般形式化，以及（3）意义关联的概念。在这个框架下，我们可以建立一个$\mathcal{O}(1/T)$的上下文学习损失界限，这里$T$是演示中示例-标签对的数量。与先前的作品相比，我们的界限反映了动词选择和指令调整的影响。一个额外的"原子概念"概念使我们的框架能够解释对语言模型训练数据中未见任务的泛化。最后，我们提出了一个玩具设置，Calcutec，

    arXiv:2402.10424v1 Announce Type: cross  Abstract: Many existing theoretical analyses of in-context learning for natural language processing are based on latent variable models that leaves gaps between theory and practice. We aim to close these gaps by proposing a theoretical framework, the Pelican Soup Framework. In this framework, we introduce (1) the notion of a common sense knowledge base, (2) a general formalism for natural language classification tasks, and the notion of (3) meaning association. Under this framework, we can establish a $\mathcal{O}(1/T)$ loss bound for in-context learning, where $T$ is the number of example-label pairs in the demonstration. Compared with previous works, our bound reflects the effect of the choice of verbalizers and the effect of instruction tuning. An additional notion of \textit{atom concepts} makes our framework possible to explain the generalization to tasks unseen in the language model training data. Finally, we propose a toy setup, Calcutec,
    

