# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Growing Self-Reliance of Chinese Innovation](https://arxiv.org/abs/2606.26470) | 通过分析中国发明专利与全球科学文献的关联，发现中国创新对美国科学的依赖大幅下降，本土科学贡献已超越美国，表明限制政策已不符合美国实际战略地位。 |
| [^2] | [Large (and Deep) Factor Models](https://arxiv.org/abs/2402.06635) | 本文通过证明一个足够宽而任意深的神经网络训练出来的投资组合优化模型与大型因子模型等效，打开了深度学习在此领域中的黑盒子，并提供了一种封闭形式的推导方法。研究实证了不同架构选择对模型性能的影响，并证明了随着深度增加，模型在足够多数据下的表现逐渐提升，直至达到饱和。 |

# 详细

[^1]: 中国创新日益增强的自主性

    The Growing Self-Reliance of Chinese Innovation

    [https://arxiv.org/abs/2606.26470](https://arxiv.org/abs/2606.26470)

    通过分析中国发明专利与全球科学文献的关联，发现中国创新对美国科学的依赖大幅下降，本土科学贡献已超越美国，表明限制政策已不符合美国实际战略地位。

    

    arXiv:2606.26470v1 公告类型：新 摘要：美国政策日益试图通过限制中国获取美国科学成果来减缓其技术崛起，其假设前提是中国创新依赖美国科学。通过将中国全部发明专利与全球科学文献进行关联分析，我们发现这种依赖近年来正在下降：中国专利背后所依赖的中国本土科学成果占比从2000年的1%上升至2025年的26%，并于2021年超过美国所占份额。随着中国对美国科学成果依赖程度的减弱，限制获取的政策已与美国实际战略地位不相匹配。

    arXiv:2606.26470v1 Announce Type: new  Abstract: U.S. policy increasingly seeks to slow China's technological rise by restricting its access to American science, on the assumption that Chinese innovation depends on U.S. science. Linking the full corpus of Chinese invention patents to the global scientific literature, we show that this dependence has fallen in recent years: the share of the China-produced science behind Chinese patents rose from 1% in 2000 to 26% in 2025, overtaking the U.S. share in 2021. As China's reliance on U.S.-produced science fades, policies restricting access fall out of alignment with the U.S.' actual strategic position.
    
[^2]: 大型（和深度）因子模型

    Large (and Deep) Factor Models

    [https://arxiv.org/abs/2402.06635](https://arxiv.org/abs/2402.06635)

    本文通过证明一个足够宽而任意深的神经网络训练出来的投资组合优化模型与大型因子模型等效，打开了深度学习在此领域中的黑盒子，并提供了一种封闭形式的推导方法。研究实证了不同架构选择对模型性能的影响，并证明了随着深度增加，模型在足够多数据下的表现逐渐提升，直至达到饱和。

    

    我们打开了深度学习在投资组合优化中的黑盒子，并证明了一个足够宽而任意深的神经网络(DNN)被训练用来最大化随机贴现因子(SDF)的夏普比率等效于一个大型因子模型(LFM)：一个使用许多非线性特征的线性因子定价模型。这些特征的性质取决于DNN的体系结构，在一种明确可追踪的方式下。这使得首次可以推导出封闭形式的端到端训练的基于DNN的SDF。我们通过实证评估了LFMs，并展示了各种架构选择如何影响SDF的性能。我们证明了深度复杂性的优点：随着足够多的数据，DNN-SDF的外样总体表现会随着神经网络的深度而增加，当隐藏层达到约100层时达到饱和。

    We open up the black box behind Deep Learning for portfolio optimization and prove that a sufficiently wide and arbitrarily deep neural network (DNN) trained to maximize the Sharpe ratio of the Stochastic Discount Factor (SDF) is equivalent to a large factor model (LFM): A linear factor pricing model that uses many non-linear characteristics. The nature of these characteristics depends on the architecture of the DNN in an explicit, tractable fashion. This makes it possible to derive end-to-end trained DNN-based SDFs in closed form for the first time. We evaluate LFMs empirically and show how various architectural choices impact SDF performance. We document the virtue of depth complexity: With enough data, the out-of-sample performance of DNN-SDF is increasing in the NN depth, saturating at huge depths of around 100 hidden layers.
    

