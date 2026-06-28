# 📋 政策通 & 职业顾问 AI Agent

自动搜索最新政策、解读官方文件、提供职业建议的 AI 智能助手。

## 🎯 核心功能

| 功能 | 描述 |
|------|------|
| 🔍 **政策查询解读** | 自动搜索国家/地方最新政策（就业、人才、补贴、社保），用人话解读 |
| 💼 **行业趋势分析** | 查询薪资水平、岗位需求、技能要求，对比不同城市 |
| 📝 **简历优化** | 分析简历内容，给出具体修改建议 |
| 🎯 **面试辅导** | 针对岗位提供面试问题、回答思路和技巧 |

## 🏗️ 技术架构

```
用户提问 → FastAPI → Claude API（Agent 循环）
                         ↓
                    需要搜索？→ Tavily Search API
                         ↓
                    SSE 流式返回前端
```

- **后端**: Python FastAPI + Anthropic Claude API
- **前端**: 原生 HTML/CSS/JS（SPA 聊天界面）
- **搜索**: Tavily Search API
- **Agent 模式**: ReAct（推理 + 行动循环）

## 🚀 快速开始

### 1. 安装依赖

```bash
cd "policy-career-advisor"
pip install -r requirements.txt
```

### 2. 配置 API Keys

```bash
# 复制配置模板
copy .env.example .env

# 编辑 .env 文件，填入你的 API Keys
# ANTHROPIC_API_KEY=sk-ant-xxxxxxxx  （从 https://console.anthropic.com/ 获取）
# TAVILY_API_KEY=tvly-xxxxxxxx       （从 https://tavily.com/ 获取）
```

### 3. 启动服务

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

### 4. 打开浏览器

访问 **http://localhost:8000**

## 📁 项目结构

```
policy-career-advisor/
├── backend/
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 配置管理
│   ├── agent/
│   │   ├── core.py          # Agent 核心循环（ReAct）
│   │   ├── tools.py         # 工具定义（4个搜索工具）
│   │   └── prompts.py       # 系统提示词
│   ├── routers/
│   │   └── chat.py          # 聊天 API（SSE 流式）
│   └── models/
│       └── schemas.py       # Pydantic 模型
├── frontend/
│   ├── index.html           # 聊天界面
│   ├── style.css            # 样式（深色模式）
│   └── app.js               # SSE 流式接收 + 交互
├── requirements.txt
├── .env.example
└── README.md
```

## 🛠️ Agent 工具

| 工具名 | 功能 | 特点 |
|--------|------|------|
| `search_web` | 通用联网搜索 | 适合大多数信息查询 |
| `search_policy` | 政策专项搜索 | 优先搜索 gov.cn 政府网站 |
| `search_job_market` | 岗位市场搜索 | 多维度：薪资 + 需求 + 趋势 |
| `get_current_time` | 获取当前时间 | 判断政策时效性 |

## ⚙️ 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `ANTHROPIC_API_KEY` | ✅ | Claude API 密钥 |
| `TAVILY_API_KEY` | ✅ | Tavily 搜索 API 密钥 |
| `CLAUDE_MODEL` | ❌ | 模型选择，默认 `claude-sonnet-4-6` |
| `PORT` | ❌ | 服务端口，默认 `8000` |

## ⚠️ 免责声明

AI 生成内容仅供参考，重要决策请核实官方信息。
