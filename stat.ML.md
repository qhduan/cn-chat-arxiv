# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Where is the Truth? The Risk of Getting Confounded in a Continual World](https://arxiv.org/abs/2402.06434) | 这篇论文研究了在一个连续学习环境中遭遇混淆的问题，通过实验证明了传统的连续学习方法无法忽略混淆，需要更强大的方法来处理这个问题。 |
| [^2] | [Analyzing Sharpness-aware Minimization under Overparameterization](https://arxiv.org/abs/2311.17539) | 本文分析了在过参数化条件下的锐度感知最小化方法。通过实证和理论结果，发现过参数化对锐度感知最小化具有重要影响，并且在过参数化增加的情况下，锐度感知最小化仍然受益。 |

# 详细

[^1]: 真相在哪里？在连续的世界中遭遇混淆的风险

    Where is the Truth? The Risk of Getting Confounded in a Continual World

    [https://arxiv.org/abs/2402.06434](https://arxiv.org/abs/2402.06434)

    这篇论文研究了在一个连续学习环境中遭遇混淆的问题，通过实验证明了传统的连续学习方法无法忽略混淆，需要更强大的方法来处理这个问题。

    

    如果一个数据集通过一个虚假相关性来解决，而这种相关性无法泛化到新数据，该数据集就是混淆的。我们将展示，在一个连续学习的环境中，混淆因素可能随着任务的变化而变化，导致的挑战远远超过通常考虑的遗忘问题。具体来说，我们从数学上推导了这种混淆因素对一组混淆任务的有效联合解空间的影响。有趣的是，我们的理论预测，在许多这样的连续数据集中，当任务进行联合训练时，虚假相关性很容易被忽略，但是在顺序考虑任务时，避免混淆要困难得多。我们构建了这样一个数据集，并通过实验证明标准的连续学习方法无法忽略混淆，而同时对所有任务进行联合训练则是成功的。我们的连续混淆数据集ConCon基于CLEVR图像，证明了需要更强大的连续学习方法来处理混淆问题。

    A dataset is confounded if it is most easily solved via a spurious correlation which fails to generalize to new data. We will show that, in a continual learning setting where confounders may vary in time across tasks, the resulting challenge far exceeds the standard forgetting problem normally considered. In particular, we derive mathematically the effect of such confounders on the space of valid joint solutions to sets of confounded tasks. Interestingly, our theory predicts that for many such continual datasets, spurious correlations are easily ignored when the tasks are trained on jointly, but it is far harder to avoid confounding when they are considered sequentially. We construct such a dataset and demonstrate empirically that standard continual learning methods fail to ignore confounders, while training jointly on all tasks is successful. Our continually confounded dataset, ConCon, is based on CLEVR images and demonstrates the need for continual learning methods with more robust b
    
[^2]: 在过参数化下分析锐度感知最小化

    Analyzing Sharpness-aware Minimization under Overparameterization

    [https://arxiv.org/abs/2311.17539](https://arxiv.org/abs/2311.17539)

    本文分析了在过参数化条件下的锐度感知最小化方法。通过实证和理论结果，发现过参数化对锐度感知最小化具有重要影响，并且在过参数化增加的情况下，锐度感知最小化仍然受益。

    

    在训练过参数化的神经网络时，尽管训练损失相同，但可以得到具有不同泛化能力的极小值。有证据表明，极小值的锐度与其泛化误差之间存在相关性，因此已经做出了更多努力开发一种优化方法，以显式地找到扁平极小值作为更具有泛化能力的解。然而，至今为止，关于过参数化对锐度感知最小化（SAM）策略的影响的研究还不多。在这项工作中，我们分析了在不同程度的过参数化下的SAM，并提出了实证和理论结果，表明过参数化对SAM具有重要影响。具体而言，我们进行了广泛的数值实验，涵盖了各个领域，并表明存在一种一致的趋势，即SAM在过参数化增加的情况下仍然受益。我们还发现了一些令人信服的案例，说明了过参数化的影响。

    Training an overparameterized neural network can yield minimizers of different generalization capabilities despite the same level of training loss. With evidence that suggests a correlation between sharpness of minima and their generalization errors, increasing efforts have been made to develop an optimization method to explicitly find flat minima as more generalizable solutions. However, this sharpness-aware minimization (SAM) strategy has not been studied much yet as to whether and how it is affected by overparameterization.   In this work, we analyze SAM under overparameterization of varying degrees and present both empirical and theoretical results that indicate a critical influence of overparameterization on SAM. Specifically, we conduct extensive numerical experiments across various domains, and show that there exists a consistent trend that SAM continues to benefit from increasing overparameterization. We also discover compelling cases where the effect of overparameterization is
    

