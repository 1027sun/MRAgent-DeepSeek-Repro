# MRAgent 复现规划

> 仓库：<https://github.com/Ji-shuo/MRAgent>
> 论文：*Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents*（ICML 2026，arXiv:2606.06036）
> 本文档基于仓库源码 + 论文原文 + 在你本机上的实际试跑结果写成。

---

## 0. 结论速览

| 问题 | 结论 |
| --- | --- |
| **要不要租 GPU？** | **不用。** 全部推理走 OpenRouter API，`torch` 只用来做 CPU 上的 L2 归一化 |
| **要下载什么？** | ① 代码仓库（<1 MB）② Python 依赖（约 470 MB，其中 torch 占 230 MB）③ LoCoMo 数据集（**仓库自带，不用下**）④ 一个 OpenRouter API key |
| **要下载的东西都下好了** | 连 275 MB 的 LongMemEval（Git LFS 文件）也已经下载 + sha256 校验 + 装进仓库了。**数据层面不需要再下任何东西** |
| **最小可跑通** | 单条对话 `conv-30`（105 个问题），约 10–20 分钟、$1 以内 |
| **全量 LoCoMo** | 10 段对话 / 272 个 session / 1986 个问题 |
| **全量费用（估算）** | Gemini-2.5-Flash 约 **$15**；gpt-4o-mini 约 $7；gpt-4o 约 $110；Claude-Sonnet-4.5 约 $140 |
| **全量耗时（估算）** | 1.5–3 小时（单条对话内部 10 线程并发） |
| **最大的坑** | pip 下载（已解决，见 §6.1）；系统默认 Python 是 3.14 装不上这些锁死的版本，必须用 3.11 |

---

## 1. 这个项目在做什么

### 1.1 核心思想

传统 RAG 记忆是 **被动检索（passive retrieval）**：把历史切块、向量化、按相似度取 top-k 塞进 prompt。作者认为这不对——认知科学里记忆是**主动重构（active reconstruction）**：由线索触发，沿着中间表征传播，逐步重建出完整记忆。

于是提出 **Cue–Tag–Content（线索–标签–内容）图** + 让 LLM 自己参与记忆访问的两阶段系统：

```
线索 (cue: 实体/关键词)  --标签 (tag: 语义桥梁)-->  内容 (episode: 具体事件)
```

标签是「语义桥梁」，让检索方向由语义相关性决定，而不是沿固定结构边盲目扩展；同时在检索过程中 LLM 可以多轮调用工具、根据已获得的证据动态剪枝。

### 1.2 代码里的两个阶段

**阶段一：建图（每个对话样本只做一次，结果会缓存）**

| 步骤 | 函数 | 做什么 |
| --- | --- | --- |
| `rewrite` | `agent/agent.py: rewrite_sample` | 逐 session 调用 LLM：消解代词、把相对时间转成绝对日期、打 tag、抽 topic、抽人物级事实。输出经 JSON Schema 校验，失败会带错误信息重试最多 3 次 |
| `extract_keyword` | `agent/agent.py: extract_keyword` | 对每个改写后的句子抽 2–30 个关键词（键节点） |
| `embed` | `data/embed_rewrite.py: embed_sample` | 句子 / topic / 问题向量化，存成 `.pkl` |
| `store` | `agent/agent.py: store_event_new` + `memory/system.py` | 建内存图：关键节点 KeyNode、EpisodeEvent、Topic、Persona、Link |

**阶段二：答题（每个问题一次）**

`Agent.answer_question` 的流程：

1. `extract_question_keys` —— 从问题抽关键词（含同义/时态变体）+ 时间约束
2. `evaluate_relations_over_graph` —— 用关键词在图里找匹配事件（full / partial matches）
3. 粗检索：`K1=80` 个候选（向量相似度）
4. 细检索：`K2=20`，两路 LLM 重排（打分 `ANSWER_SORT_PROMPT` + 选择 `ANSWER_SORT_PROMPT2`）
5. `select_key_tag` —— 当某个 key 关联的 tag 超过 `TAG_MAX=15` 时再让 LLM 打分
6. **工具调用循环** —— 最多 `MAX_ROUNDS=8` 轮，7 个工具：
   `edges_by_tag` / `query_conversation_time` / `query_event_keywords` / `query_event_context` / `query_personal_information` / `query_personal_aspect` / `query_topic_events`

### 1.3 评测

- 指标：F1 + LLM-as-judge（`gpt-4o-mini`，temperature 0）+ evidence recall
- 对抗题（LoCoMo category 5）用字符串匹配判"Not mentioned"，**不计入 F1/judge**——这与论文"排除对抗题"的设置一致

### 1.4 论文用的配置（复现时对齐这些）

| 组件 | 论文 | 仓库里对应 |
| --- | --- | --- |
| 推理 backbone | Gemini-2.5-Flash / Claude-Sonnet-4.5 | `--model gemini` / `--model claude` |
| Judge | GPT-4o-mini, temp 0 | `eval/judge.py` 里写死 `openai/gpt-4o-mini` |
| Embedding | — | `openai/text-embedding-3-large`（3072 维），经 OpenRouter |
| 轮数上限 | 8 轮 / 每轮 ≤10 次工具调用 | `MAX_ROUNDS=8`、`MAX_TOOL_CALLS=50` |
| 独立重复 | 3 次取均值±标准差 | 仓库不自动重复，需自己换 `--file` 跑 3 遍 |

> **一个有用的发现**：`--re_model` 可以单独指定"检索/答题"用的模型。论文里的 `MRAgent*`（记忆用 Gemini 建、检索答题用 Claude）在仓库里就是
> `--model gemini --re_model claude`，不需要改代码。

---

## 2. 要不要租卡：不用

**明确结论：不需要 GPU。**

依据（都在代码里）：

1. 所有 LLM 调用走 `https://openrouter.ai/api/v1`（`common/config.py: OPENROUTER_URL`、`llm/controller.py`），是远程 API。
2. 所有 embedding 也走 OpenRouter（`llm/embeddings.py`：`text-embedding-3-large`）。
3. `torch` 只在 `llm/rag_utils.py` 出现一次，做的是 `torch.nn.functional.normalize(embeddings, dim=-1)`——**把向量 L2 归一化**。代码里甚至写了 `device = "cuda:0" if torch.cuda.is_available() else "cpu"`，但有 GPU 时也只是拿它做归一化，纯属浪费。
   - 仓库 README 也明说：*"a CPU build is sufficient"*。
4. 没有 transformers / sentence-transformers / 任何本地模型权重。

**你实际需要的是：** 网络能连上 OpenRouter + 一个能付费的 API key + 约 1.5 GB 磁盘（虚拟环境）。

> 唯一会让人想到 GPU 的场景是：以后想省 embedding 的钱，改成跑本地 `bge-m3` 之类。但按当前规模（全量 LoCoMo 的 embedding 成本约 $0.2）完全没必要。

---

## 3. 要下载 / 准备的东西

| # | 东西 | 大小 | 必要性 | 说明 |
| --- | --- | --- | --- | --- |
| 1 | 仓库代码 | <1 MB | 必须 | 已在 `work/MRAgent` |
| 2 | Python 3.11 虚拟环境 | ~1.5 GB | 必须 | torch 2.8.0 占 230 MB（下载）/ 1.4 GB（解压后） |
| 3 | **OpenRouter API key** | — | 必须 | 充值即可，chat / embedding / judge 共用一个 key |
| 4 | `data/dataset_locomo.json` | 2.7 MB | 必须 | **仓库自带**，已验证可直接加载 |
| 5 | `data/dataset_LM.json` | **275 MB**（262.3 MiB） | LongMemEval 才需要 | Git LFS 文件，网页下载/zip 只会拿到 134 字节的指针。**已下载并校验完毕** |
| 6 | 可选：NLTK data | ~10 MB | 评测才需要 | `eval/` 里用到（当前代码路径未强制需要） |

### 关于 LoCoMo 的一个说明

论文 D.1 写"LoCoMo 有 50 段对话"，但**仓库里是 10 段对话（1986 个问题）**，也就是官方 `locomo10.json` 那个发布版。复现时以仓库为准即可，不影响方法本身。

规模实测（我直接统计的）：10 段对话 · 272 个 session · 5882 轮对话 · 72.7 万字符 · 1986 个问题。

每题类型分布：single-hop 841、adversarial 446、temporal 321、multi-hop 282、open-domain 96。

---

## 4. 我已在你这台电脑上验证过的事

不是纸面推演，下面每一条都是实际跑出来的：

| 验证项 | 结果 |
| --- | --- |
| Python 3.11.9 + `torch 2.8.0+cpu` 安装 | 成功（CUDA 不可用，符合预期） |
| 仓库 18 个模块全部 import | **18/18 成功，无 Windows 兼容问题** |
| LoCoMo 数据经 `data/get_data.py` 加载 | 成功：10 samples / 272 sessions / 1986 questions |
| **离线端到端跑通**（用 stub 替换 LLM 和 embedding） | 成功：`conv-26` 199 题、`conv-30` 105 题全部产出结果 |
| 检索 + 双路重排分支 | 成功覆盖（conv-30 触发 95 次重排调用） |
| 评测脚本 `eval/evaluate_reasoning.py` | 成功（F1 分类统计 + judge 统计都跑出来了） |
| LongMemEval 数据集下载 + 校验 | 275,038,753 字节，**sha256 与 LFS oid 完全一致** |
| LongMemEval 走仓库自己的 `get_data("LM", ...)` 加载 | 成功：500 samples / 500 questions / 23,867 sessions，2.8 秒加载完 |

LongMemEval 实测结构（之前没法验证，现在确认与 `data/get_data.py` 的解析逻辑完全吻合）：

```
500 个样本，每个样本恰好 1 个问题，共 500 题
category 分布：multi-session 133、temporal-reasoning 133、knowledge-update 78、
              single-session-user 70、single-session-assistant 56、single-session-preference 30
每个样本约 53 个 session；全部样本合计 246,750 轮对话 / 2.45 亿字符（≈6100 万 token）
```

但 `data/get_data.py` 对 LM 做了过滤：**只保留 user 的发言**（`if dataset == "LM" and speaker != "user": continue`）。
实际喂给模型的量是 122,416 轮 / 3072 万字符 ≈ **770 万 token，平均每个样本约 1.5 万 token**，
其余约 87% 的 assistant 回复被丢弃。

离线跑通产生的真实产物（可作为"没配 key 也能先验证环境"的证据）：

```
data/locomo/rewrite_gemini/conv-30_rewrite.json        57 KB
data/locomo/keyword_gemini/conv-30_keyword.json        26 KB
data/locomo/embedding/gpt_gemini/conv-30_embedding.pkl 130 KB
result/locomo/conv-30_result_gemini_mock.jsonl         24 KB   ← 105 行
```

也就是说：**代码本身在 Windows + Python 3.11 上没有障碍，剩下的只是配 API key 花钱跑真数据。**

---

## 5. 目录与产物说明

```
run.py                      # 唯一入口
common/config.py            # 所有超参 + 路径模板（K1=80/K2=20/MAX_ROUNDS=8 ...）
agent/agent.py              # 两阶段主流程（43 KB，核心）
agent/tools.py              # 7 个工具的 schema + 分发
memory/system.py            # 图数据结构（KeyNode/EpisodeEvent/Topic/Persona/Link）
memory/controller.py        # 图查询实现
llm/controller.py           # OpenRouter 封装 + 工具调用循环
llm/embeddings.py           # embedding 客户端
prompts/prompts.py          # 全部提示词
prompts/schema.py           # 输出 JSON 校验
eval/evaluate_reasoning.py  # 评测入口

运行时产物：
data/<dataset>/rewrite_<model>/<sample>_rewrite.json      # 阶段一缓存
data/<dataset>/keyword_<model>/<sample>_keyword.json      # 阶段一缓存
data/<dataset>/embedding/gpt_<model>/<sample>_embedding.pkl
result/<dataset>/<sample>_result_<model>_<file>.jsonl     # ★ 唯一预测输出
log/<dataset>/                                            # 每题推理轨迹
```

**缓存机制很重要**：某一步只要输出文件已存在就跳过。想重跑某一步，就删掉对应文件。

---

## 6. 复现步骤（可直接复制）

### 6.0 前置：确认 Python 版本

你机器上默认的 `python` 是 **3.14**（`E:\python`），而 `requirements.txt` 锁的 `numpy==1.26.4` / `torch==2.8.0` 没有 3.14 的 wheel，**必须用 3.11**：

```
C:\Users\wangchang\AppData\Local\Programs\Python\Python311\python.exe
```

### 6.1 依赖安装（这里的坑我已经踩过并解决了）

```powershell
cd C:\Users\wangchang\Documents\Codex\2026-09-18\https-github-com-ji-shuo-mragent\work\MRAgent

# 用 3.11 建虚拟环境
C:\Users\wangchang\AppData\Local\Programs\Python\Python311\python.exe -m venv ..\venv311

# 装依赖：两个关键点
#   --no-cache-dir : 你本机 pip 缓存已有 6.7 GB，pip 会在里面卡死（实测卡 10 分钟没有任何进展）
#   -i 清华源      : pypi.org 直连只有 ~27 KB/s，清华源实测 12.5 MB/s
..\venv311\Scripts\python.exe -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

如果 `regex==2022.4.24` 报编译错误（它在 PyPI 上只有源码包，需要 MSVC 编译器），把它换成有 wheel 的新版本即可——它在整个项目里只在 `eval/evaluation.py` 用了一句 `regex.sub(r'\b(a|an|the|and)\b', ' ', text)`，版本完全不敏感：

```powershell
..\venv311\Scripts\python.exe -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple "regex>=2023.0"
```

### 6.2 配置 API key

```powershell
Copy-Item .env.example .env
# 编辑 .env，把 OPENROUTER_API_KEY 换成你自己的
```

在 <https://openrouter.ai> 注册、充值、生成 key。**第一步先验证 key 能用**（OpenRouter 的 `/embeddings` 端点是较新的能力，值得先确认）：

```powershell
..\venv311\Scripts\python.exe -c "from llm.embeddings import get_openai_embedding as g; v=g(['hello'], model='openai/text-embedding-3-large'); print(len(v), len(v[0]))"
# 期望输出：1 3072
```

### 6.3 先跑一条对话（最小验证，约 10–20 分钟）

```powershell
..\venv311\Scripts\python.exe run.py --data locomo --model gemini --file smoke --sample 30
```

产物：`result/locomo/conv-30_result_gemini_smoke.jsonl`（105 行）。

> 提示：代码里最小的运行粒度就是"一条完整对话"（105–199 题），没有"只跑 N 题"的参数。想更省，可以临时把 `data/dataset_locomo.json` 里某段的 `qa` 列表截短。

### 6.4 全量 LoCoMo

```powershell
..\venv311\Scripts\python.exe run.py --data locomo --model gemini --file full
```

中断了直接重跑，已完成的题会跳过（按结果文件行数续跑）。

### 6.5 评测

```powershell
..\venv311\Scripts\python.exe eval\evaluate_reasoning.py --data locomo --model gemini --file full --allfile
```

输出每题类型的 F1 和 LLM-judge 准确率，并写 `result_judge_locomo_gemini_full.jsonl`。

### 6.6 对齐论文里的 `MRAgent*`（可选）

```powershell
..\venv311\Scripts\python.exe run.py --data locomo --model gemini --re_model claude --file star
```

### 6.7 LongMemEval（可选，重得多）

数据集**已经就位**（275 MB 已下载、sha256 校验通过并放进 `data/dataset_LM.json`），可以直接跳到下面的命令。

如果需要重新下载，**不要用网页下载或 zip 解压**（LFS 指针只有 134 字节）。两种方式：

```powershell
# 方式 A：git lfs（需先装 git-lfs）
git lfs install
git clone https://github.com/Ji-shuo/MRAgent.git

# 方式 B：直接拿 LFS 对象（我已写好脚本，校验 sha256）
node <本目录>\scripts\download_lm_lfs.mjs ..\MRAgent\data\dataset_LM.json
```

然后按类别分别跑（这是三个独立实验）：

```powershell
..\venv311\Scripts\python.exe run.py --data LM --model gemini --file lm --ca 0 --lm_batch 1   # multi-session
..\venv311\Scripts\python.exe run.py --data LM --model gemini --file lm --ca 1 --lm_batch 1   # single-session-user
..\venv311\Scripts\python.exe run.py --data LM --model gemini --file lm --ca 2 --lm_batch 1   # temporal-reasoning
```

---

## 7. 费用与时间估算

### 7.1 LoCoMo（1986 题）

估算依据：每题约 4–6 次 LLM 调用（抽题关键词 1 次 + 两路重排 2 次 + 答题工具循环 2–3 轮），
单题约 **1.8 万 input token / 1 千 output token**（两路重排每次要读最多 80 个候选句子，是主要开销）。

| 模型 | 输入 $/M | 输出 $/M | 全量估算 |
| --- | --- | --- | --- |
| **google/gemini-2.5-flash**（论文主配置） | 0.30 | 2.50 | **≈ $16** |
| openai/gpt-4o-mini | 0.15 | 0.60 | ≈ $7 |
| qwen/qwen3-max | 0.78 | 3.90 | ≈ $36 |
| openai/gpt-4o | 2.50 | 10.0 | ≈ $110 |
| anthropic/claude-sonnet-4.5（论文另一配置） | 3.00 | 15.0 | ≈ $140 |

另外：embedding 全场约 $0.2，judge（gpt-4o-mini）约 $0.3，**可以忽略**。
论文是 3 次独立重复取均值，如果全都要复现就把上面数字 ×3。

### 7.2 时间

| 阶段 | 估算 |
| --- | --- |
| 单条 LoCoMo 对话（105–199 题） | 10–20 分钟 |
| LoCoMo 全量 | 1.5–3 小时（样本内 10 线程并发，样本间串行） |
| LongMemEval | **大幅增加，而且没法靠并发救**（见下） |

### 7.3 LongMemEval 的两个额外陷阱

**陷阱一：并发在这里失效。** `run.py` 的并发是"同一个样本内的多个问题并行 10 线程"，
但 LongMemEval 每个样本**只有 1 个问题**，所以 LM 实际是**一个样本接一个样本串行跑**。
按论文报告的单样本 586 秒算：

| 范围 | 样本数 | 串行耗时 |
| --- | --- | --- |
| 只跑 README 推荐的三类（multi-session / single-session-user / temporal-reasoning） | 336 | **≈ 55 小时** |
| 全部 6 类 | 500 | ≈ 81 小时 |

缓解办法只有一个：**按 category 拆成多个进程并行**（`--ca 0/1/2` 各起一个），能压到约 1/3。

**陷阱二：输出 token 比输入贵。** 建图阶段的 `extract_keyword` 对**每个句子**输出 2–30 个关键词，
输出量非常大，而输出单价通常是输入的 5–8 倍。

| 模型 | 全量 LM（500 题）粗略估算 |
| --- | --- |
| openai/gpt-4o-mini | ≈ $20 |
| google/gemini-2.5-flash | ≈ $60–90 |
| anthropic/claude-sonnet-4.5 | ≈ $400–500 |

再加上 embedding 约 $1。**结论：LM 用 Gemini 跑一轮的钱（$60–90）和 LoCoMo（$16）不是一个量级，
时间（几十小时）更不是一个量级。建议先把 LoCoMo 做扎实。**

### 7.4 论文给的成本对比（可以直接引用）

MRAgent 在 LongMemEval 上每样本 118k token / 586 秒，而 LangMem 是 3,268k token / 1,210 秒，A-Mem 是 632k / 1,122 秒——**省 token 是这篇论文的卖点之一**，复现时这个数字比绝对准确率更容易对齐。

---

## 8. 已知的坑与风险

| 风险 | 严重度 | 说明 / 对策 |
| --- | --- | --- |
| OpenRouter 的 `/embeddings` 端点 | 中 | 端点存在且 `text-embedding-3-large` 在列（我已确认，$0.13/M token），但它比 chat 端点新。**第一步就先单独验证**（§6.2） |
| 默认 Python 是 3.14 | 高 | numpy 1.26.4 / torch 2.8.0 无 3.14 wheel，必须显式用 3.11 |
| pip 缓存卡死 | 高 | 已解决：`--no-cache-dir` |
| `regex==2022.4.24` 需编译 | 中 | 换成 `regex>=2023`，无兼容风险 |
| LongMemEval 的 LFS 文件 | 中 | 网页下载得到的是 134 字节指针。本机 GitHub 直连约 20–27 KB/s，275 MB 要 3 小时以上；建议走代理或换网络 |
| 结果可复现性 | 中 | 代码用了 10 线程并发 + `seed=66`；论文又是 3 次重复取均值。想对齐数字要按论文跑 3 遍 |
| 对抗题口径 | 低 | 仓库会跑 category 5，但评测里单独用字符串匹配、不进 F1/judge，与论文一致；如果你只比 F1 需注意分母 |
| 缓存导致"改了代码没生效" | 低 | 阶段一产物存在就跳过，改 prompt 后记得删 `data/<dataset>/{rewrite,keyword,embedding}` 对应文件 |

---

## 9. 建议的执行顺序

```
第 1 步  环境 + 两套数据集已全部就绪（我已完成）———— 直接用 work\venv311 + work\MRAgent
第 2 步  申请 OpenRouter key，充 $10–20
第 3 步  单独验证 embedding 端点（§6.2 那行命令，30 秒）
第 4 步  跑 conv-30 单条对话（10–20 分钟，<$1）—— 先确认全链路真的通
第 5 步  看 log/locomo/ 里的推理轨迹，确认工具调用是合理的，而不是报错兜底
第 6 步  跑 eval 看单条对话的 F1/judge 是否在论文量级
第 7 步  全量 LoCoMo（$16 左右，2 小时）
第 8 步  再考虑要不要做 LongMemEval（$60–90、几十小时，见 §7.3）
```

**建议就在第 4 步之后先停一下**：单条对话跑通 + 结果是合理的，再花全量的钱，比一次性全量跑完发现口径不对要划算得多。

---

## 附：本目录提供的脚本

| 脚本 | 用途 |
| --- | --- |
| `scripts/mock_e2e.py` | **不花钱的端到端自检**：用 stub 替换 LLM/embedding，跑完整流水线。放在仓库根目录执行 `python mock_e2e.py 30` |
| `scripts/mock_eval.py` | 用 stub judge 跑评测脚本，验证 F1/统计代码 |
| `scripts/smoke_imports.py` | 检查仓库所有模块能否 import |
| `scripts/smoke_data.py` | 加载 LoCoMo 并打印数据规模统计 |
| `scripts/download_lm_lfs.mjs` | 下载 LongMemEval 的 LFS 对象并校验 sha256（绕开 git-lfs） |
| `scripts/lm_stats.py` | 统计 LongMemEval 规模（样本数 / 类别 / token 量） |
| `scripts/lm_load.py` | 走仓库自己的 `get_data("LM", ...)` 加载并抽查一条样本 |
