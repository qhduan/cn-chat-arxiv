# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Partially Recentralization Softmax Loss for Vision-Language Models Robustness](https://arxiv.org/abs/2402.03627) | 本文研究了通过修改预训练多模态模型的损失函数来提高对抗鲁棒性，通过限制前K个softmax输出。实验结果表明，经过微调后，模型的对抗鲁棒性显著提高，能够有效抵御常见的攻击。 |
| [^2] | [Superiority of Softmax: Unveiling the Performance Edge Over Linear Attention.](http://arxiv.org/abs/2310.11685) | 通过对softmax和线性注意力机制进行全面比较分析，本论文揭示了softmax注意力在大多数情况下优于线性注意力的潜在原因。 |

# 详细

[^1]: 近似的中心化softmax损失用于视觉-语言模型的鲁棒性

    Partially Recentralization Softmax Loss for Vision-Language Models Robustness

    [https://arxiv.org/abs/2402.03627](https://arxiv.org/abs/2402.03627)

    本文研究了通过修改预训练多模态模型的损失函数来提高对抗鲁棒性，通过限制前K个softmax输出。实验结果表明，经过微调后，模型的对抗鲁棒性显著提高，能够有效抵御常见的攻击。

    

    随着大型语言模型在自然语言处理任务中的突破，多模态技术变得非常流行。然而，已经证明多模态自然语言处理模型容易受到对抗攻击，即模型的输出可以通过对输入进行微小扰动而发生巨大变化。虽然计算机视觉和自然语言处理模型中已经提出了几种防御技术，但对多模态模型的鲁棒性还没有进行充分探索。在本文中，我们研究了通过修改预训练多模态模型的损失函数，通过限制前K个softmax输出来提供的对抗鲁棒性。基于评估和评分，我们的实验结果显示，在经过微调后，预训练模型的对抗鲁棒性可以显着提高，对抗常见的攻击有效。进一步的研究应该探索这类损失函数的输出多样性、泛化能力以及鲁棒性和性能之间的平衡。我们的代码将在之后提供。

    As Large Language Models make a breakthrough in natural language processing tasks (NLP), multimodal technique becomes extremely popular. However, it has been shown that multimodal NLP are vulnerable to adversarial attacks, where the outputs of a model can be dramatically changed by a perturbation to the input. While several defense techniques have been proposed both in computer vision and NLP models, the multimodal robustness of models have not been fully explored. In this paper, we study the adversarial robustness provided by modifying loss function of pre-trained multimodal models, by restricting top K softmax outputs. Based on the evaluation and scoring, our experiments show that after a fine-tuning, adversarial robustness of pre-trained models can be significantly improved, against popular attacks. Further research should be studying, such as output diversity, generalization and the robustness-performance trade-off of this kind of loss functions. Our code will be available after th
    
[^2]: Softmax的优越性：揭示其相对于线性注意力的性能优势

    Superiority of Softmax: Unveiling the Performance Edge Over Linear Attention. (arXiv:2310.11685v1 [cs.CL])

    [http://arxiv.org/abs/2310.11685](http://arxiv.org/abs/2310.11685)

    通过对softmax和线性注意力机制进行全面比较分析，本论文揭示了softmax注意力在大多数情况下优于线性注意力的潜在原因。

    

    大型Transformer模型在许多自然语言处理任务中取得了最先进的结果。在Transformer架构的重要组成部分中，注意力机制通过利用softmax函数捕捉序列中的标记交互起着关键作用。相反，线性注意力通过线性复杂度近似softmax操作，提供了一种计算效率更高的替代方法。然而，与传统的softmax注意力机制相比，它在性能上表现出明显的降级。在本文中，我们对这两种注意力机制进行了全面的比较分析，揭示了softmax注意力在大多数情况下优于线性注意力的潜在原因。

    Large transformer models have achieved state-of-the-art results in numerous natural language processing tasks. Among the pivotal components of the transformer architecture, the attention mechanism plays a crucial role in capturing token interactions within sequences through the utilization of softmax function.  Conversely, linear attention presents a more computationally efficient alternative by approximating the softmax operation with linear complexity. However, it exhibits substantial performance degradation when compared to the traditional softmax attention mechanism.  In this paper, we bridge the gap in our theoretical understanding of the reasons behind the practical performance gap between softmax and linear attention. By conducting a comprehensive comparative analysis of these two attention mechanisms, we shed light on the underlying reasons for why softmax attention outperforms linear attention in most scenarios.
    

