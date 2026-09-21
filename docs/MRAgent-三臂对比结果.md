# MRAgent 跑通流程（conv-26）

答题模型：`deepseek-flash`（thinking 关闭）；
Embedding：本地 `bge-base-en-v1.5`；
Judge：`deepseek-v4-pro`；
样本：conv-26（选它是因为它覆盖了LoCoMo中的5种题型），共199 题（其中对抗题 47 道不计入 judge，计分 152 道）；

## 一、结果

| 实验臂 | 多跳 (32) | 时间 (37) | 开放域 (13) | 单跳 (70) | **总体 (152)** |
| --- | --- | --- | --- | --- | --- |
| **MRAgent**（主动重构 + 工具循环） | 78.1% | **100.0%** | 69.2% | 95.7% | **90.79%** |
| Full-context（全文塞入 prompt） | 71.9% | 43.2% | **76.9%** | 95.7% | 76.32% |
| Passive（一次 top-k 直接答题） | 71.9% | 35.1% | 61.5% | 85.7% | 68.42% |

```
MRAgent      138/152 = 90.79%   ████████████████████████████████████▏
Full-context 116/152 = 76.32%   ██████████████████████████████▌
Passive      104/152 = 68.42%   ███████████████████████████▍
```

## 二、三个关键结论

**1. 多轮工具循环带来了 +22.4 个点的提升**（90.79 vs 68.42）。
两条臂的原料、embedding、初始检索**完全相同**——唯一差别是 Passive 拿到初始上下文后
一次性作答，MRAgent 则可以最多 8 轮地继续调用工具。所以这 22.4 个点可以归因于
「主动重构」本身，这是最干净的一次消融。

**2. 时间类题的差距是压倒性的：100% vs 35.1% / 43.2%。**
这是全表最有说服力的一行。原因很具体：`query_conversation_time` 工具能返回
「事件所在对话的发生时间」，模型据此把相对表述（"the sunday before 25 May"）锚定成绝对日期。
两条对照臂都没有这个能力。看实例：

```
Q: When did Caroline give a speech at a school?
   gold:          The week before 9 June 2023
   MRAgent:       the week before 9 June 2023   ← judge 判对，字面也完全正确
   Full-context:  Not mentioned                 ← 信息就在它读到的上下文里，却放弃了
   Passive:       the week before 7 May 2023    ← 锚定到了错误的对话

Q: When did Melanie go to the pottery workshop?
   gold:          The Friday before 15 July 2023
   MRAgent:       14 July 2023                  ← 正确
   Full-context:  Not mentioned
   Passive:       6 May 2023
```

统计上，时间类题里「MRAgent 判对、且另外两条臂都判错」的共有 **14 道**。

> 注：之前这里举的 "charity race" 例子不成立——那一题 MRAgent 和 Full-context 都答
> "20 May 2023"、judge 都判对，只有 Passive 答错，说明不了"Full-context 不行"。
> 上面换成的是逐题核对过的例子。

**3. 一个反直觉的结果：全文塞进 prompt 反而全面优于一次性检索。**
Full-context（76.32%）显著高于 Passive（68.42%），在单跳题上甚至和 MRAgent 打平（都是 95.7%），
在开放域题上还**超过了 MRAgent**（76.9% vs 69.2%）。

说明**在这个 embedding 模型下，"退化成一次 top-k"比"根本不检索"损失更大。**
这一定程度上可以反过来说明 MRAgent 的收益主要不是来自"更好的检索"，而是来自"能反复检索"(反复地工具调用)。

## 三、成本

| 项目 | API 调用 | 成本（低峰） |
| --- | --- | --- |
| MRAgent 臂 | 1,844 | $1.065 |
| Passive + Full-context | 1,021 | $0.581 |
| Judge × 3 次评测 | ≈ 460 | ≈ $0.30 |
| **合计** | ≈ 3,300 | **≈ $1.95** |

MRAgent 的成本是另外两条臂**合计的 1.8 倍**（多出来的全是工具循环里的调用）。
换算成性价比：多花 $0.9，换来 +14~22 个点，很划算。（但是这一点与论文中不矛盾，论文中的MRAgent最划算是相对于其它的记忆方法而言，但这里的成本并不是与其它记忆方法的对比）

PS:
**结论不能和论文直接比。** 
只跑了一段对话（论文是 10 段汇总 1986 题），judge 和 embedding 也都换了（原文的embedding模型是openai的text-embedding-3-large，然后这里用的是GitHub上下载的BAAI/bge-base-en-v1.5）。
而且用的模型也不同（原文是Gemini、Claude；这里用的是deepseek）
