# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Volatility Surfaces using Generative Adversarial Networks.](http://arxiv.org/abs/2304.13128) | 本文提出了一种使用GAN高效计算波动率曲面的方法。所提出的GAN模型允许使用浅层网络，从而大大降低了计算成本。实验结果表明，在计算波动率曲面方面具有优势。 |

# 详细

[^1]: 使用生成对抗网络学习波动率曲面

    Learning Volatility Surfaces using Generative Adversarial Networks. (arXiv:2304.13128v1 [q-fin.CP])

    [http://arxiv.org/abs/2304.13128](http://arxiv.org/abs/2304.13128)

    本文提出了一种使用GAN高效计算波动率曲面的方法。所提出的GAN模型允许使用浅层网络，从而大大降低了计算成本。实验结果表明，在计算波动率曲面方面具有优势。

    

    本文提出了一种使用生成对抗网络（GAN）高效计算波动率曲面的方法。这种方法利用了GAN神经网络的特殊结构，一方面可以从训练数据中学习波动率曲面，另一方面可以执行无套利条件。特别地，生成器网络由鉴别器辅助训练，鉴别器评估生成的波动率是否与目标分布相匹配。同时，我们的框架通过引入惩罚项作为正则化项，训练GAN网络以满足无套利约束。所提出的GAN模型允许使用浅层网络，从而大大降低了计算成本。在实验中，我们通过与计算隐含和本地波动率曲面的最先进方法进行对比，展示了所提出的方法的性能。我们的实验结果表明，相对于人工神经网络（ANN）方法，我们的GAN模型在精度和实际应用中都具有优势。

    In this paper, we propose a generative adversarial network (GAN) approach for efficiently computing volatility surfaces. The idea is to make use of the special GAN neural architecture so that on one hand, we can learn volatility surfaces from training data and on the other hand, enforce no-arbitrage conditions. In particular, the generator network is assisted in training by a discriminator that evaluates whether the generated volatility matches the target distribution. Meanwhile, our framework trains the GAN network to satisfy the no-arbitrage constraints by introducing penalties as regularization terms. The proposed GAN model allows the use of shallow networks which results in much less computational costs. In our experiments, we demonstrate the performance of the proposed method by comparing with the state-of-the-art methods for computing implied and local volatility surfaces. We show that our GAN model can outperform artificial neural network (ANN) approaches in terms of accuracy an
    

