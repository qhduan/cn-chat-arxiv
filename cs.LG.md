# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ForestColl: Efficient Collective Communications on Heterogeneous Network Fabrics](https://arxiv.org/abs/2402.06787) | ForestColl是一种针对任意网络拓扑生成高效调度的工具，通过构建广播/聚合生成跨越树的通信调度，实现了理论上的最小网络拥塞，并在实验中表现出高于供应商自带通信库的性能。 |
| [^2] | [Improving the forecast accuracy of wind power by leveraging multiple hierarchical structure](https://arxiv.org/abs/2308.03472) | 通过整合风电场中风力发电机的横截面和时间层次结构，构建跨时层次结构，从而提高风电场的预测准确性。 |

# 详细

[^1]: ForestColl: 异构网络结构上高效的集合通信

    ForestColl: Efficient Collective Communications on Heterogeneous Network Fabrics

    [https://arxiv.org/abs/2402.06787](https://arxiv.org/abs/2402.06787)

    ForestColl是一种针对任意网络拓扑生成高效调度的工具，通过构建广播/聚合生成跨越树的通信调度，实现了理论上的最小网络拥塞，并在实验中表现出高于供应商自带通信库的性能。

    

    随着现代深度神经网络模型越来越大，加速器之间的集合通信（如allreduce等）成为一个重要的性能瓶颈。在当今高度多样化和异构的网络结构下设计高效的通信调度是一项具有挑战性的任务。本文提出了一种名为ForestColl的工具，它能够为任意网络拓扑生成高效的调度。ForestColl使用广播/聚合生成跨越树作为通信调度，实现了理论上的最小网络拥塞。其调度生成运行在强多项式时间内，且具有高扩展性。ForestColl支持包括交换网络和直接连接在内的任何网络结构，以及任何网络图结构。我们在多集群的AMD MI250和NVIDIA A100平台上评估了ForestColl。与供应商自己优化的通信库RCCL和NCCL相比，ForestColl的调度性能提高了高达52％。ForestColl还优于其他...

    As modern DNN models grow ever larger, collective communications between the accelerators (allreduce, etc.) emerge as a significant performance bottleneck. Designing efficient communication schedules is challenging given today's highly diverse and heterogeneous network fabrics. In this paper, we present ForestColl, a tool that generates efficient schedules for any network topology. ForestColl constructs broadcast/aggregation spanning trees as the communication schedule, achieving theoretically minimum network congestion. Its schedule generation runs in strongly polynomial time and is highly scalable. ForestColl supports any network fabrics, including both switching fabrics and direct connections, as well as any network graph structure. We evaluated ForestColl on multi-cluster AMD MI250 and NVIDIA A100 platforms. ForestColl's schedules achieved up to 52\% higher performance compared to the vendors' own optimized communication libraries, RCCL and NCCL. ForestColl also outperforms other s
    
[^2]: 通过利用多层次结构提高风力发电的预测准确性

    Improving the forecast accuracy of wind power by leveraging multiple hierarchical structure

    [https://arxiv.org/abs/2308.03472](https://arxiv.org/abs/2308.03472)

    通过整合风电场中风力发电机的横截面和时间层次结构，构建跨时层次结构，从而提高风电场的预测准确性。

    

    可再生能源发电对全球减碳至关重要。预测可再生能源，特别是风能，具有挑战性，因为风能发电受气候条件的不确定性影响。最近通过协调实现的层次预测在短期内显著提高了风能预测的质量。我们利用风电场中风力发电机的横截面和时间层次结构，构建横时层次结构，进一步研究跨横截面和时间维度的整合如何增加风电场的预测准确性。我们发现，跨时间协调在多个时间汇总中优于单独跨横截面协调。此外，基于机器学习的跨时协调预测表现出对较粗时间聚合的高准确性。

    arXiv:2308.03472v2 Announce Type: replace  Abstract: Renewable energy generation is of utmost importance for global decarbonization. Forecasting renewable energies, particularly wind energy, is challenging due to the inherent uncertainty in wind energy generation, which depends on weather conditions. Recent advances in hierarchical forecasting through reconciliation have demonstrated a significant increase in the quality of wind energy forecasts for short-term periods. We leverage the cross-sectional and temporal hierarchical structure of turbines in wind farms and build cross-temporal hierarchies to further investigate how integrated cross-sectional and temporal dimensions can add value to forecast accuracy in wind farms. We found that cross-temporal reconciliation was superior to individual cross-sectional reconciliation at multiple temporal aggregations. Additionally, machine learning based forecasts that were cross-temporally reconciled demonstrated high accuracy at coarser tempora
    

