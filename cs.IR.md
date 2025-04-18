# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Uncertainty Calibration for Counterfactual Propensity Estimation in Recommendation.](http://arxiv.org/abs/2303.12973) | 本文提出了多种不确定性校准技术，以改进推荐系统中倾向性估计的效果。经过实验验证，校准后的IPS估计器在Coat和yahoo数据集上表现更好。 |

# 详细

[^1]: 推荐系统中反事实倾向估计的不确定性校准

    Uncertainty Calibration for Counterfactual Propensity Estimation in Recommendation. (arXiv:2303.12973v1 [cs.AI])

    [http://arxiv.org/abs/2303.12973](http://arxiv.org/abs/2303.12973)

    本文提出了多种不确定性校准技术，以改进推荐系统中倾向性估计的效果。经过实验验证，校准后的IPS估计器在Coat和yahoo数据集上表现更好。

    

    在推荐系统中，由于选择偏差，许多评分信息都丢失了，这被称为非随机缺失。反事实逆倾向评分（IPS）被用于衡量每个观察到的评分的填充错误。虽然在多种情况下有效，但我们认为IPS估计的性能受到倾向性估计不确定性的限制。本文提出了多种代表性的不确定性校准技术，以改进推荐系统中倾向性估计的不确定性校准。通过对偏误和推广界限的理论分析表明，经过校准的IPS估计器优于未校准的IPS估计器。 Coat和yahoo数据集上的实验结果表明，不确定性校准得到改进，从而使推荐结果更好。

    In recommendation systems, a large portion of the ratings are missing due to the selection biases, which is known as Missing Not At Random. The counterfactual inverse propensity scoring (IPS) was used to weight the imputation error of every observed rating. Although effective in multiple scenarios, we argue that the performance of IPS estimation is limited due to the uncertainty miscalibration of propensity estimation. In this paper, we propose the uncertainty calibration for the propensity estimation in recommendation systems with multiple representative uncertainty calibration techniques. Theoretical analysis on the bias and generalization bound shows the superiority of the calibrated IPS estimator over the uncalibrated one. Experimental results on the coat and yahoo datasets shows that the uncertainty calibration is improved and hence brings the better recommendation results.
    

