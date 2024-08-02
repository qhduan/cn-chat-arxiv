# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Measuring the Dunkelflaute: How (not) to analyze variable renewable energy shortage](https://arxiv.org/abs/2402.06758) | 该论文探讨了如何定义、量化和比较可变可再生能源短缺事件，讨论了不同方法的优缺点，并提出了进一步的研究方法。 |
| [^2] | [Towards Enhanced Local Explainability of Random Forests: a Proximity-Based Approach.](http://arxiv.org/abs/2310.12428) | 这项研究提出了一种利用随机森林模型的特征空间中的邻近性来解释模型预测的方法，为模型预测提供了局部的解释性，与现有方法相辅相成。通过实验证明了这种方法在债券定价模型中的有效性。 |
| [^3] | [Multinomial Backtesting of Distortion Risk Measures.](http://arxiv.org/abs/2201.06319) | 这项研究提出了一种适用于一般失真风险测度的多项式反向测试方法，通过分层和随机化风险水平，扩展了反向测试模型的适用范围。 |

# 详细

[^1]: 测量Dunkelflaute：如何（不）分析可变可再生能源短缺

    Measuring the Dunkelflaute: How (not) to analyze variable renewable energy shortage

    [https://arxiv.org/abs/2402.06758](https://arxiv.org/abs/2402.06758)

    该论文探讨了如何定义、量化和比较可变可再生能源短缺事件，讨论了不同方法的优缺点，并提出了进一步的研究方法。

    

    随着可变可再生能源在全球能源系统中的重要性日益增加，人们对于了解可变可再生能源短缺（“Dunkelflauten”）时期的兴趣也越来越大。在不同的可再生能源发电技术和地点之间定义、量化和比较这种短缺事件，是一个非常复杂的挑战。不同文献中存在着各种方法，如水文学、风能和太阳能分析，或能源系统建模。先前分析的研究对象范围从特定位置的单一技术到多个地区的多样化技术组合，要么关注可变可再生能源的供应，要么关注其与电力需求的不匹配。我们提供了一种量化可变可再生能源短缺的方法概述。我们解释并批判性地讨论了不同方法在定义和识别短缺事件方面的优点和挑战，并提出了进一步的方法。

    As variable renewable energy sources increasingly gain importance in global energy systems, there is a growing interest in understanding periods of variable renewable energy shortage (``Dunkelflauten''). Defining, quantifying, and comparing such shortage events across different renewable generation technologies and locations presents a surprisingly intricate challenge. Various approaches exist in different bodies of literature, such as hydrology, wind and solar energy analysis, or energy system modeling. The subject of interest in previous analyses ranges from single technologies in specific locations to diverse technology portfolios across multiple regions, focusing either on supply from variable renewables or its mismatch with electricity demand. We provide an overview of methods for quantifying variable renewable energy shortage. We explain and critically discuss the merits and challenges of different approaches for defining and identifying shortage events and propose further method
    
[^2]: 实现随机森林的局部可解释性增强：基于邻近性的方法

    Towards Enhanced Local Explainability of Random Forests: a Proximity-Based Approach. (arXiv:2310.12428v1 [stat.ML])

    [http://arxiv.org/abs/2310.12428](http://arxiv.org/abs/2310.12428)

    这项研究提出了一种利用随机森林模型的特征空间中的邻近性来解释模型预测的方法，为模型预测提供了局部的解释性，与现有方法相辅相成。通过实验证明了这种方法在债券定价模型中的有效性。

    

    我们提出一种新的方法来解释随机森林（RF）模型的样本外性能，利用了任何RF都可以被表述为自适应加权K最近邻（KNN）模型的事实。具体而言，我们利用RF在特征空间中学到的点之间的邻近性，将随机森林的预测重写为训练数据点目标标签的加权平均值。这种线性性质有助于在训练集观测中为任何模型预测生成属性，从而为RF预测提供了局部的解释性，补充了SHAP等已有方法，这些方法则为特征空间维度上的模型预测生成属性。我们在训练于美国公司债券交易数据的债券定价模型中演示了这种方法，并将其与各种现有的模型解释方法进行了比较。

    We initiate a novel approach to explain the out of sample performance of random forest (RF) models by exploiting the fact that any RF can be formulated as an adaptive weighted K nearest-neighbors model. Specifically, we use the proximity between points in the feature space learned by the RF to re-write random forest predictions exactly as a weighted average of the target labels of training data points. This linearity facilitates a local notion of explainability of RF predictions that generates attributions for any model prediction across observations in the training set, and thereby complements established methods like SHAP, which instead generates attributions for a model prediction across dimensions of the feature space. We demonstrate this approach in the context of a bond pricing model trained on US corporate bond trades, and compare our approach to various existing approaches to model explainability.
    
[^3]: 多项式失真风险测度的反向测试方法

    Multinomial Backtesting of Distortion Risk Measures. (arXiv:2201.06319v2 [q-fin.RM] UPDATED)

    [http://arxiv.org/abs/2201.06319](http://arxiv.org/abs/2201.06319)

    这项研究提出了一种适用于一般失真风险测度的多项式反向测试方法，通过分层和随机化风险水平，扩展了反向测试模型的适用范围。

    

    我们通过提出一种适用于一般失真风险测度的多项式反向测试方法，扩展了反向测试模型适用的风险测度范围。该方法依赖于风险水平的分层和随机化。我们通过数值案例研究展示了我们方法的表现。

    We extend the scope of risk measures for which backtesting models are available by proposing a multinomial backtesting method for general distortion risk measures. The method relies on a stratification and randomization of risk levels. We illustrate the performance of our methods in numerical case studies.
    

