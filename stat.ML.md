# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Cram Method for Efficient Simultaneous Learning and Evaluation](https://arxiv.org/abs/2403.07031) | Cram方法是一种同时学习和评估的高效方法，利用整个样本进行训练和测试，比传统的样本分割策略更高效。 |

# 详细

[^1]: 用于高效同时学习和评估的Cram方法

    The Cram Method for Efficient Simultaneous Learning and Evaluation

    [https://arxiv.org/abs/2403.07031](https://arxiv.org/abs/2403.07031)

    Cram方法是一种同时学习和评估的高效方法，利用整个样本进行训练和测试，比传统的样本分割策略更高效。

    

    我们介绍了“Cram”方法，这是一种通用且高效的方法，使用通用的机器学习（ML）算法进行同时学习和评估。在批处理数据的单次传递中，该方法反复训练ML算法并测试其经验性能。由于它同时利用了整个样本进行学习和评估，所以Cram方法比样本分割要高效得多。Cram方法还自然地适用于在线学习算法，使其实施具有计算效率。为了展示Cram方法的强大之处，我们考虑了标准策略学习设置，其中将Cram应用于相同数据以开发个性化治疗规则（ITR）并估计如果学习的ITR被部署将会产生的平均结果。我们展示了在最小一组假设下，由此产生的Cram评估估计器是一致且渐近的。

    arXiv:2403.07031v1 Announce Type: new  Abstract: We introduce the "cram" method, a general and efficient approach to simultaneous learning and evaluation using a generic machine learning (ML) algorithm. In a single pass of batched data, the proposed method repeatedly trains an ML algorithm and tests its empirical performance. Because it utilizes the entire sample for both learning and evaluation, cramming is significantly more data-efficient than sample-splitting. The cram method also naturally accommodates online learning algorithms, making its implementation computationally efficient. To demonstrate the power of the cram method, we consider the standard policy learning setting where cramming is applied to the same data to both develop an individualized treatment rule (ITR) and estimate the average outcome that would result if the learned ITR were to be deployed. We show that under a minimal set of assumptions, the resulting crammed evaluation estimator is consistent and asymptoticall
    

