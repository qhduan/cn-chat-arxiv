# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Moonwalk: Inverse-Forward Differentiation](https://arxiv.org/abs/2402.14212) | Moonwalk引入了一种基于向量-逆-Jacobian乘积的新技术，加速前向梯度计算，显著减少内存占用，并在保持真实梯度准确性的同时，将计算时间降低了几个数量级。 |
| [^2] | [Gradient-flow adaptive importance sampling for Bayesian leave one out cross-validation for sigmoidal classification models](https://arxiv.org/abs/2402.08151) | 本研究引入了渐变流自适应重要性抽样的方法，用于稳定贝叶斯分类模型的留一交叉验证预测的蒙特卡罗近似，以评估模型的普适性。 |
| [^3] | [A Comprehensive Survey on Enterprise Financial Risk Analysis from Big Data Perspective.](http://arxiv.org/abs/2211.14997) | 本文从大数据角度综述了企业财务风险分析的研究现状，回顾了250多篇代表性文章。 |

# 详细

[^1]: Moonwalk：逆向-前向微分

    Moonwalk: Inverse-Forward Differentiation

    [https://arxiv.org/abs/2402.14212](https://arxiv.org/abs/2402.14212)

    Moonwalk引入了一种基于向量-逆-Jacobian乘积的新技术，加速前向梯度计算，显著减少内存占用，并在保持真实梯度准确性的同时，将计算时间降低了几个数量级。

    

    反向传播虽然在梯度计算方面有效，但在解决内存消耗和扩展性方面表现不佳。这项工作探索了前向梯度计算作为可逆网络中的一种替代方法，展示了它在减少内存占用的潜力，并不带来重大缺点。我们引入了一种基于向量-逆-Jacobian乘积的新技术，加速了前向梯度的计算，同时保留了减少内存和保持真实梯度准确性的优势。我们的方法Moonwalk在网络深度方面具有线性时间复杂度，与朴素前向的二次时间复杂度相比，在没有分配更多内存的情况下，从实证的角度减少了几个数量级的计算时间。我们进一步通过将Moonwalk与反向模式微分相结合来加速，以实现与反向传播相当的时间复杂度，同时保持更小的内存使用量。

    arXiv:2402.14212v1 Announce Type: cross  Abstract: Backpropagation, while effective for gradient computation, falls short in addressing memory consumption, limiting scalability. This work explores forward-mode gradient computation as an alternative in invertible networks, showing its potential to reduce the memory footprint without substantial drawbacks. We introduce a novel technique based on a vector-inverse-Jacobian product that accelerates the computation of forward gradients while retaining the advantages of memory reduction and preserving the fidelity of true gradients. Our method, Moonwalk, has a time complexity linear in the depth of the network, unlike the quadratic time complexity of na\"ive forward, and empirically reduces computation time by several orders of magnitude without allocating more memory. We further accelerate Moonwalk by combining it with reverse-mode differentiation to achieve time complexity comparable with backpropagation while maintaining a much smaller mem
    
[^2]: 渐变流自适应重要性抽样用于sigmoid分类模型的贝叶斯留一交叉验证

    Gradient-flow adaptive importance sampling for Bayesian leave one out cross-validation for sigmoidal classification models

    [https://arxiv.org/abs/2402.08151](https://arxiv.org/abs/2402.08151)

    本研究引入了渐变流自适应重要性抽样的方法，用于稳定贝叶斯分类模型的留一交叉验证预测的蒙特卡罗近似，以评估模型的普适性。

    

    我们引入了一组梯度流引导的自适应重要性抽样（IS）变换，用于稳定贝叶斯分类模型的点级留一交叉验证（LOO）预测的蒙特卡罗近似。可以利用这种方法来评估模型的普适性，例如计算与AIC类似的LOO或计算LOO ROC / PRC曲线以及派生的度量指标，如AUROC和AUPRC。通过变分法和梯度流，我们推导出两个简单的非线性单步变换，利用梯度信息将模型的预训练完整数据后验靠近目标LOO后验预测分布。这样，变换稳定了重要性权重。因为变换涉及到似然函数的梯度，所以结果的蒙特卡罗积分依赖于模型Hessian的Jacobian行列式。我们推导出了这些Jacobian行列式的闭合精确公式。

    We introduce a set of gradient-flow-guided adaptive importance sampling (IS) transformations to stabilize Monte-Carlo approximations of point-wise leave one out cross-validated (LOO) predictions for Bayesian classification models. One can leverage this methodology for assessing model generalizability by for instance computing a LOO analogue to the AIC or computing LOO ROC/PRC curves and derived metrics like the AUROC and AUPRC. By the calculus of variations and gradient flow, we derive two simple nonlinear single-step transformations that utilize gradient information to shift a model's pre-trained full-data posterior closer to the target LOO posterior predictive distributions. In doing so, the transformations stabilize importance weights. Because the transformations involve the gradient of the likelihood function, the resulting Monte Carlo integral depends on Jacobian determinants with respect to the model Hessian. We derive closed-form exact formulae for these Jacobian determinants in
    
[^3]: 从大数据角度看企业财务风险分析的综述研究

    A Comprehensive Survey on Enterprise Financial Risk Analysis from Big Data Perspective. (arXiv:2211.14997v3 [q-fin.RM] UPDATED)

    [http://arxiv.org/abs/2211.14997](http://arxiv.org/abs/2211.14997)

    本文从大数据角度综述了企业财务风险分析的研究现状，回顾了250多篇代表性文章。

    

    企业财务风险分析旨在预测企业未来的财务风险。由于其广泛而重要的应用，企业财务风险分析一直是金融和管理领域的核心研究主题。基于先进的计算机科学和人工智能技术，企业风险分析研究正在经历快速发展并取得重要进展。因此，全面评估相关研究既有必要性又具挑战性。虽然已经存在一些有价值和令人印象深刻的关于企业风险分析的综述，但这些综述单独介绍了方法，缺乏企业财务风险分析的最新进展。相反，本文尝试从大数据的角度提供企业风险分析方法的系统文献综述，回顾了超过250篇代表性文章。

    Enterprise financial risk analysis aims at predicting the future financial risk of enterprises. Due to its wide and significant application, enterprise financial risk analysis has always been the core research topic in the fields of Finance and Management. Based on advanced computer science and artificial intelligence technologies, enterprise risk analysis research is experiencing rapid developments and making significant progress. Therefore, it is both necessary and challenging to comprehensively review the relevant studies. Although there are already some valuable and impressive surveys on enterprise risk analysis from the perspective of Finance and Management, these surveys introduce approaches in a relatively isolated way and lack recent advances in enterprise financial risk analysis. In contrast, this paper attempts to provide a systematic literature survey of enterprise risk analysis approaches from Big Data perspective, which reviews more than 250 representative articles in the 
    

