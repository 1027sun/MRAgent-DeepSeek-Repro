# MRAgent × DeepSeek 改造说明

改造日期：2026-09-21　目标：用 DeepSeek 跑 LoCoMo，并与"不用 MRAgent"的方案对比。

## 一、改了什么（7 个文件，原文件都在 `work\MRAgent\_backup_original\`）

| 文件 | 改动 | 为什么 |
| --- | --- | --- |
| `common/config.py` | 端点/密钥/模型改为环境变量驱动；新增 `LLM_DROP_PARAMS`、`LLM_EXTRA_BODY`；新增 `--maxq`；新增 `USAGE_LOG` | 换服务商不用改代码；DeepSeek 的参数怪癖需要可配置 |
| `llm/controller.py` | 客户端改用 `config.LLM_BASE_URL/LLM_API_KEY`；请求前剔除不支持的参数、合并 `extra_body`；每次调用记录 token 用量 | DeepSeek 不支持 `seed` / `parallel_tool_calls`，且默认开 thinking 模式 |
| `llm/rag_utils.py` | 新增本地 embedding 后端（`EMBED_BACKEND=local`），含 **L2 归一化** 和 BGE 查询前缀 | DeepSeek 没有 embeddings API，需要第二来源 |
| `llm/embeddings.py` | base_url / key / 模型名改为环境变量 | 便于把 embedding 指向别家 |
| `eval/judge.py` | judge 端点与模型可独立配置（`JUDGE_*`） | 论文用 gpt-4o-mini，这里默认用 DeepSeek 强模型 |
| `run.py` | 支持 `--maxq N` 限制每题样本的题目数 | 让 pilot 只花几分钱 |
| `.env` / `.env.example` | 写入完整配置 | 所有开关集中在一处 |

## 二、三个关键细节（都已在代码里落实）

**1. 关闭 thinking 模式。** DeepSeek 的 thinking 默认开启且 effort=high，思维链按**输出 token** 计费，会让成本翻几倍；而且 thinking 模式下 `temperature` 被忽略（代码依赖 `temperature=0` 的确定性）。现在通过
`LLM_EXTRA_BODY='{"thinking":{"type":"disabled"}}'` 显式关闭。

**2. 剔除 DeepSeek 不支持的参数。** 代码原本每次都传 `seed=66` 和 `parallel_tool_calls=True`，这两个在 DeepSeek 文档里都不存在。危险之处在于：一旦返回 400，答题阶段会被静默当成"no information available"，所有题都答"无信息"却看不出报错。现在通过 `LLM_DROP_PARAMS=seed,parallel_tool_calls` 移除。

**3. Embedding 必须归一化。** `common/utils.py` 的 `topk_answers_by_similarity()` 默认用**点积**，全项目没有任何一处传 `cosine`——它等价于余弦只是因为向量被 L2 归一化过。本地路径用 `normalize_embeddings=True`，实测行范数精确为 1.0。

## 三、已验证（不需要花 API 钱）

| 检查项 | 结果 |
| --- | --- |
| 全部 `.py` 语法 | 通过（0 失败） |
| 17 个模块 import | 通过（0 失败） |
| 配置解析（`--model deepseek`） | `MODEL=RE_MODEL=deepseek-flash`；端点 `https://api.deepseek.com` |
| 本地 embedding | **离线**从缓存加载 bge-base，输出 (N, 768) float32，行范数 = 1.0 |
| embedding 语义正确性 | 查询"Caroline 参加互助小组"与对应句相似度 0.6709，为三者最高 |
| 端到端整合（真实 embedding + 假 LLM） | 跑通 conv-30 的 8 题：rewrite → 真实向量 → 关键词 → 建图 → 检索 → 答题 → 写结果 |

## 四、怎么跑

### 第一步：填入 DeepSeek key

编辑 `work\MRAgent\.env`，把这一行填上（**不要把 key 发在聊天里**）：

```ini
LLM_API_KEY=sk-你的DeepSeekKey
```

### 第二步：先跑 10 题验证兼容性（<$0.05）

```powershell
cd E:\虚拟C盘\MRAgent_reproduction\work\MRAgent
..\venv311\Scripts\python.exe run.py --data locomo --model deepseek --file ds26 --sample 26 --maxq 10
```

重点看：`log\locomo\` 里的日志有没有报错、"no information available" 是不是异常多（多为 400 静默失败的信号）。

### 第三步：跑完整 conv-26（199 题，约 $0.5–1）

```powershell
..\venv311\Scripts\python.exe run.py --data locomo --model deepseek --file ds26 --sample 26
```

可中断续跑：已答完的题按结果文件行数跳过。想强制重建，删 `data\locomo\` 下对应缓存文件。

### 第四步：评测

```powershell
..\venv311\Scripts\python.exe eval\evaluate_reasoning.py --data locomo --model deepseek --file ds26 --allfile
```

### 第五步：查真实成本

```powershell
Get-Content ..\..\work\MRAgent\log\locomo\token_usage.jsonl |
  ConvertFrom-Json |
  Measure-Object -Property prompt_tokens,completion_tokens -Sum
```

（或直接看 DeepSeek 平台的用量面板。注意低峰时段是半价：北京时间工作日 09:00–12:00、14:00–18:00 之外都是低峰。）

## 五、对比实验（"用 / 不用 MRAgent"）

`conv-26` 跑通后，加两条对照臂：

1. **Passive**：复用同一套记忆缓存，去掉工具调用循环，一次 top-k 直接答题——这是最干净的消融
2. **Full-context**：不检索，整段对话塞进 prompt（DeepSeek 上下文 1M，conv-26 才约 1.5 万 token）

三条臂共用同一套 embedding 和同一个 judge，只有检索策略不同。

## 六、注意事项

- **缓存是按模型名分目录的**：`--model deepseek` 用 `data\locomo\rewrite_deepseek\`。换模型名会重新建图。
- **别把 mock 产物混进来**：假 LLM 跑出来的数据如果留在 `data\locomo\` 里，真跑时会被当缓存复用。跑真实验前请确认该目录为空（当前已清空）。
- `--model deepseek3.5` 之类不存在的名字会**原样透传**给 API 并报错；`config.py` 里那些 OpenRouter 模型名（如 `claude3.5`）在 OpenRouter 上个别已下架。
