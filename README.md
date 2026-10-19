# NexusChat

> **An agentic, human-in-the-loop LLM chatbot** built with LangGraph, Groq, Tavily (via MCP) and durable Postgres checkpointing.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agents-1C3C3C?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-qwen3.8--27b-F55036?style=flat-square)
![MCP](https://img.shields.io/badge/MCP-Tavily_Search-2E7D6E?style=flat-square)
![Postgres](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📖 Overview

NexusChat is **not** a naive "call the LLM and return the text" chatbot. It is a **graph-based agent** where every user message flows through a pipeline of specialized nodes that guard, validate, rewrite, generate, verify and optionally search the live web before answering.

Key ideas:

- **Stateful graph** — the whole conversation logic is a `StateGraph` compiled with a shared `AgentState` typed dictionary.
- **Safety first** — a guardrail node blocks harmful/jailbreak/empty queries before anything else runs.
- **Self-correcting queries** — a query checker detects vague or history-dependent phrasing and a rewriter rewrites it using chat history.
- **Freshness-aware answers** — a groundedness grader detects time-sensitive questions and triggers a live web search.
- **Human-in-the-loop** — web search never runs silently; the graph `interrupt()`s and waits for the user's **Y/N** approval.
- **Durable memory** — all conversation state is checkpointed to **PostgreSQL** via `PostgresSaver`, so threads survive restarts.

---

## ✨ Features

| Feature | Description |
| --- | --- |
| 🛡️ **Guardrails** | Structured LLM classifier blocks harmful, illegal, jailbreak, PII and gibberish queries. |
| 🔍 **Query validation & rewriting** | Vague/ambiguous queries are detected and rewritten into standalone queries using chat history. |
| ✍️ **Context-aware generation** | Answers are grounded in web context when available, otherwise in the model's knowledge + history. |
| 🌐 **Live web search** | Tavily search via the **Model Context Protocol (MCP)** remote server. |
| ⏸️ **Human-in-the-loop approval** | LangGraph `interrupt()` pauses the graph until the user allows or cancels web search. |
| ♻️ **Groundedness re-check** | After searching, the answer is regenerated with fresh context; search can only run once per turn. |
| 💾 **Postgres checkpointing** | `PostgresSaver` persists every thread's state — resume conversations across process restarts. |
| 🧵 **Threaded conversations** | Each conversation gets a UUID `thread_id` used as the checkpointer key. |
| 🐳 **Dockerized** | One-command stack: Postgres 16 + the app, with healthchecks and dependency ordering. |
| 📊 **LangSmith ready** | Tracing env vars are wired in `.env` for full graph execution observability. |

---

## 🧱 Tech Stack

### Core

| Layer | Technology | Why |
| --- | --- | --- |
| Language | **Python ≥ 3.13** (pinned `3.13`) | Modern typing, `asyncio` improvements |
| Agent orchestration | **LangGraph** `>=1.2.15` | Explicit state machine for multi-step agent logic, interrupts, conditional edges |
| LLM framework | **LangChain** `>=1.4.6` / `langchain-core` `>=1.6.10` | Prompts, chains, output parsers, message types |
| LLM provider | **Groq** via `langchain-groq` — model `qwen/qwen3.8-27b` | Ultra-fast inference |
| Structured output | **Pydantic** + `with_structured_output()` | Typed guardrail / checker / grader decisions |

### Tools & Memory

| Layer | Technology | Why |
| --- | --- | --- |
| Web search | **Tavily** remote MCP server | Real-time search results |
| MCP adapter | **`langchain-mcp-adapters`** `>=0.3.2` + **`mcp`** `>=1.30.0` | Calls MCP tools like LangChain tools (`streamable_http` transport) |
| Checkpointing | **`langgraph-checkpoint-postgres`** `>=3.2.0` | Durable thread state in Postgres |
| DB driver | **psycopg 3** (`psycopg[binary,pool]`) `>=3.3.6` | Async-friendly Postgres driver used by the saver |

### Infrastructure

| Layer | Technology |
| --- | --- |
| Config | `python-dotenv` (`.env` loading) |
| Package mgmt | **uv** (`pyproject.toml` + `uv.lock`) *or* pip (`requirements.txt`) |
| Containers | **Docker** (`python:3.13-slim`) + **Docker Compose** (`postgres:16-alpine`) |
| Observability | **LangSmith** (optional tracing) |
| License | MIT |

---

## 🏗️ Architecture

### High-level components

```mermaid
flowchart LR
    subgraph Client["CLI Client"]
        U[👤 User]
    end

    subgraph App["NexusChat App (backend.py)"]
        G["🤖 LangGraph StateGraph<br/>'Nexus' compiled"]
        CP[("🛢️ PostgresSaver<br/>checkpointer")]
    end

    subgraph LLMs["Groq"]
        QW["qwen/qwen3.8-27b<br/>(guardrails, rewrite, generate, grade)"]
    end

    subgraph External["External Services"]
        TV[("🔍 Tavily Remote MCP<br/>mcp.tavily.com")]
        PG[("🐘 PostgreSQL 16<br/>checkpoints / threads")]
        LS[("📊 LangSmith<br/>tracing (optional)")]
    end

    U -->|"Ask Anything"| G
    G -->|structured calls| QW
    G -->|"interrupt() → Y/N"| U
    G --- CP
    CP <--> PG
    G -->|"streamable_http"| TV
    G -.-> LS
```

### Project structure

```
NexusChat/
├── backend.py               # 🧠 The entire agent: state, nodes, graph, CLI loop
├── mcp_client.py            # 🔌 Tavily MCP client (async web-search tool)
├── main.py                  # 🚪 Minimal uv/entry-point stub
├── pyproject.toml           # 📦 uv project metadata + dependencies
├── uv.lock                  # 🔒 Locked dependency tree (uv)
├── requirements.txt         # 📦 pip-style dependency list
├── Dockerfile               # 🐳 App image (python:3.13-slim)
├── docker-compose.yml       # 🐳 App + Postgres 16 stack
├── .env                     # 🔑 Secrets (NOT committed — create your own)
├── .python-version          # 3.13
├── .dockerignore
├── .gitignore
├── LICENSE                  # MIT © 2026 Yogeshwaran D
└── README.md
```

---

## 🔁 Agent Workflow

### 1. Graph (state machine) — the brain of NexusChat

```mermaid
flowchart TD
    START([🟢 START]) --> GUARD

    GUARD{"🛡️ guardrail<br/>safe query?"}
    GUARD -->|"❌ blocked"| END1([🔴 END])
    GUARD -->|"✅ allowed"| QC

    QC{"🔍 query_check<br/>clear & self-contained?"}
    QC -->|"needs rewrite"| RW
    QC -->|"clear"| GEN

    RW{"✍️ rewriter<br/>rewrite w/ chat history"}
    RW --> GEN

    GEN{"✍️ generate<br/>answer from context/history"}
    GEN --> GRD

    GRD{"🎯 grounded<br/>answer up-to-date?"}
    GRD -->|"yes — timeless"| END2([🟢 END ✅ answer])
    GRD -->|"no — time-sensitive"| WS

    WS{"⏸️ interrupt()<br/>allow web search? (Y/N)"}
    WS -->|"N — cancelled"| END3([🔴 END ⛔ cancelled])
    WS -->|"Y"| SEARCH

    SEARCH[("🌐 tavily_search<br/>via MCP")] --> GEN2

    GEN2["✍️ generate #2<br/>answer w/ fresh context"]
    GEN2 --> GRD2{"🎯 grounded<br/>searched already?"}
    GRD2 -->|"yes"| END4([🟢 END ✅ answer])

    style GUARD fill:#f9d0d0,stroke:#c0392b
    style QC fill:#fdebd0,stroke:#e67e22
    style RW fill:#fdebd0,stroke:#e67e22
    style GRD fill:#d6eaf8,stroke:#2980b9
    style WS fill:#e8daef,stroke:#8e44ad
    style SEARCH fill:#d1f2eb,stroke:#1abc9c
    style GEN fill:#d5f5e3,stroke:#27ae60
    style GEN2 fill:#d5f5e3,stroke:#27ae60
```

### 2. Sequence diagram — a full turn, end to end

```mermaid
sequenceDiagram
    actor User
    participant CLI as backend.py (CLI loop)
    participant G as LangGraph "Nexus"
    participant LLM as Groq qwen3.8-27b
    participant PG as PostgreSQL
    participant TV as Tavily MCP

    User->>CLI: "who won the match today?"
    CLI->>G: invoke(inputs, config{thread_id})
    G->>PG: load checkpoint(thread_id)

    rect rgb(249, 208, 208)
        Note over G,LLM: 🛡️ Stage 1 — Guardrail
        G->>LLM: guardrail_prompt | structured(llm_guardrails)
        LLM-->>G: guard = "yes"
    end

    rect rgb(253, 235, 208)
        Note over G,LLM: 🔍 Stage 2 — Query check / rewrite
        G->>LLM: query_checker_llm (structured)
        LLM-->>G: needs_rewrite_query = "yes"
        G->>LLM: rewriter_prompt + history
        LLM-->>G: "who won the football match on 18 Oct 2026?"
    end

    rect rgb(213, 245, 227)
        Note over G,LLM: ✍️ Stage 3 — Generation #1
        G->>LLM: generation_prompt (empty context)
        LLM-->>G: draft answer
        G->>PG: save checkpoint
    end

    rect rgb(214, 234, 248)
        Note over G,LLM: 🎯 Stage 4 — Groundedness grading
        G->>LLM: grounded_llm (structured) + today's date
        LLM-->>G: grounded = "no" (time-sensitive)
    end

    rect rgb(232, 218, 239)
        Note over G,User: ⏸️ Stage 5 — Human-in-the-loop interrupt
        G-->>CLI: __interrupt__ = {action: Web Search, approval: ...}
        CLI->>User: "Allow for {...}? Y/N"
        User->>CLI: Y
        CLI->>G: Command(resume="Y")
    end

    rect rgb(209, 242, 235)
        Note over G,TV: 🌐 Stage 6 — Web search via MCP
        G->>TV: MultiServerMCPClient → tavily_search(query)
        TV-->>G: results → concatenated content
        G->>PG: save checkpoint
    end

    rect rgb(213, 245, 227)
        Note over G,LLM: ✍️ Stage 7 — Regeneration + final grade
        G->>LLM: generation_prompt (with web context)
        LLM-->>G: final grounded answer
        G->>LLM: grounded_llm → web_search_done = True
        LLM-->>G: pass (no second search)
        G->>PG: save final checkpoint
    end

    G-->>CLI: result{generation}
    CLI->>User: "Nexus : <answer>"
```

### 3. Interrupt / resume loop (human-in-the-loop)

```mermaid
flowchart LR
    A["Nexus.invoke(inputs)"] --> B{"__interrupt__ present?"}
    B -->|"No"| C["Print result['generation']<br/>✅ done"]
    B -->|"Yes"| D["Print permission prompt"]
    D --> E["User enters Y/N"]
    E --> F["inputs = Command(resume=choice)"]
    F --> A
```

---

## 🧬 Shared State (`AgentState`)

Every node reads and writes this `TypedDict`; `messages` is reducer-merged via `add_messages`.

| Field | Type | Meaning |
| --- | --- | --- |
| `messages` | `Annotated[list[BaseMessage], add_messages]` | Full conversation history (human + AI turns) |
| `original_query` | `str` | Raw user input (used by guardrail & grader) |
| `query` | `str` | Possibly rewritten, standalone query |
| `generation` | `str` | The assistant's answer |
| `websearch` | `bool` | Grader decision: is a web search needed? |
| `websearch_context` | `str` | Concatenated Tavily result content |
| `validate_query` | `bool` | Checker decision: rewrite needed? |
| `grounded_answer` | `bool` | Answer grounding flag |
| `guardrail` | `bool` | `True` = safe to proceed, `False` = blocked |
| `web_search_done` | `bool` | Prevents searching more than once per turn |
| `web_search_cancelled` | `bool` | User declined the search permission |

---

## 🧩 Node Reference

| Node | Function | LLM usage | Writes |
| --- | --- | --- | --- |
| `guardrail` | `guardrail_node` | `llm.with_structured_output(llm_guardrails)` | `guardrail` |
| `query_check` | `ground_query` | `llm.with_structured_output(llm_validate_query)` | `validate_query`, `query` |
| `rewriter` | `rewriter_query` | `prompt \| llm \| StrOutputParser` (with history) | `query` |
| `generate` | `generation` | `prompt \| llm \| StrOutputParser` (context + history) | `generation`, `messages` |
| `grounded` | `grounded_answer` | `llm.with_structured_output(llm_ground_answer)` + today's date | `websearch` |
| `web_search` | `websearch` | `interrupt()` → `asyncio.run(tavily_search)` | `websearch_context`, `web_search_done`, `web_search_cancelled` |

**Conditional routers:**

| Router | Logic |
| --- | --- |
| `guardrail_should_continue` | safe → `query_check`, blocked → `END` |
| `query_checker_should_continue` | needs rewrite → `rewriter`, else → `generate` |
| `grounded_should_continue` | search needed & not done → `web_search`, else → `END` |
| `after_websearch` | cancelled → `END`, else → `generate` (regenerate) |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) *or* pip
- **Docker + Docker Compose** (optional, for the containerized stack)
- API keys: **[Groq](https://console.groq.com/keys)** and **[Tavily](https://app.tavily.com/)**

### 1. Clone

```bash
git clone https://github.com/Yogeshwaran197/NexusChat.git
cd NexusChat
```

### 2. Configure environment

Create a `.env` file in the project root:

```dotenv
# --- LLM ---
GROQ_API_KEY=gsk_your_groq_key_here

# --- Web search (MCP) ---
TAVILY_API_KEY=tvly_your_tavily_key_here

# --- Durable memory ---
DATABASE_URL=postgresql://user:password@localhost:5432/nexuschat

# --- Optional: LangSmith tracing ---
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=lsv2_your_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=nexuschat

# --- Docker Compose Postgres bootstrap ---
POSTGRES_USER=nexus
POSTGRES_PASSWORD=change_me
POSTGRES_DB=nexuschat
```

> ⚠️ **Never commit `.env`.** It is already listed in `.gitignore` and `.dockerignore`. If you ever pushed real credentials (e.g. a hosted DB URL inside `docker-compose.yml`), **rotate those secrets immediately** — treat them as compromised.

### 3. Install dependencies

**Option A — uv (recommended):**

```bash
uv sync
```

**Option B — pip:**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Database

The checkpointer needs a reachable PostgreSQL and will auto-run `checkpointer.setup()` (creates the checkpoint tables) on first boot.

- **Local via Docker:**

  ```bash
  docker compose up -d db
  ```

  Then point `DATABASE_URL` at `postgresql://nexus:change_me@localhost:5432/nexuschat`.

- **Managed Postgres:** any Postgres 14+ (e.g. Render, Supabase, Neon). `sslmode=require` is appended automatically if missing.

### 5. Run

```bash
python backend.py
```

Example session:

```text
Ask Anything : hello
Nexus : Hi there! How can I help you today?

Ask Anything : who won the cricket world cup?
Allow for {'action': 'Web Search', 'query': '...', 'approval': 'allow web search?'}?
Y/N Allow WebSearch?Y
Nexus : Australia won the 2023 ICC Cricket World Cup, defeating India by 6 wickets...
```

Type `exit` or `end` to quit.

---

## 🐳 Docker Deployment

The Compose file runs **two services**: `db` (Postgres 16-alpine with a named volume `pgdata` and a `pg_isready` healthcheck) and `app` (built from `Dockerfile`, gated on `service_healthy`).

```mermaid
flowchart LR
    subgraph Compose["docker compose up"]
        APP["app container<br/>python:3.13-slim<br/>CMD: python backend.py"]
        DB["db container<br/>postgres:16-alpine<br/>volume: pgdata"]
    end
    ENV[".env"] --> APP
    APP -->|"depends_on: healthy"| DB
    APP -->|"DATABASE_URL (override)"| DB
    TV[("Tavily MCP")] --- APP
    GQ[("Groq API")] --- APP
    APP -. TTY/STDIN .-> U[👤 interactive CLI]
```

```bash
# start everything (db healthy → app starts)
docker compose up --build

# stop
docker compose down

# stop and wipe chat history
docker compose down -v
```

Notes:

- `stdin_open: true` + `tty: true` keep the interactive input loop usable inside the container.
- `DATABASE_URL` in the `environment:` block **overrides** any value from `.env` — replace the placeholder with your own Postgres URL (preferably referenced as an env var rather than hardcoded).

---

## ⚙️ Configuration Cheat Sheet

| What | Where | Default / Notes |
| --- | --- | --- |
| LLM model | `backend.py` → `ChatGroq(model=...)` | `qwen/qwen3.8-27b` |
| Guardrail rules | `guardrail_node` system prompt | Blocks harmful / jailbreak / PII / gibberish |
| Rewrite criteria | `ground_query` system prompt | Vague, pronoun-dependent, typo-heavy → rewrite |
| Generation rules | `generation` system prompt | Prefer context; cite sources; graceful fallback |
| Freshness policy | `grounded_answer` prompt | News/prices/scores/roles → search; greetings/opinions → skip |
| MCP transport | `mcp_client.py` | `streamable_http` → `https://mcp.tavily.com/mcp/?tavilyApiKey=...` |
| Thread identity | CLI loop | `thread_id = str(uuid.uuid4())` per session |
| SSL | `get_database_url()` | Appends `sslmode=require` if absent |
| Checkpoint setup | import-time | `checkpointer.setup()` creates tables automatically |

---

## 🧠 How Memory Works

```mermaid
flowchart LR
    subgraph Turn["Per user turn"]
        M["messages reducer<br/>(add_messages)"]
    end
    N1[Node] --> M
    M --> N2[Node]
    M --> CP["PostgresSaver<br/>(thread_id)"]
    CP --> T[("checkpoints table<br/>in PostgreSQL")]
    T -. resume .-> N1
```

- The `add_messages` reducer appends, rather than replaces, messages.
- `get_history(state)` returns everything **before the latest `HumanMessage`**, which is injected into the rewriter/generator prompts as `MessagesPlaceholder("history")`.
- Because the checkpointer keys on `thread_id`, you can change the CLI to reuse a fixed ID and conversations will **resume across restarts**.

---

## 🛣️ Roadmap

- [ ] Web UI (FastAPI / Streamlit) replacing the CLI loop
- [ ] Configurable model selection per environment variable
- [ ] Tool registry beyond Tavily (RAG, SQL, code interpreter)
- [ ] Unit tests for each graph node + graph-level integration tests
- [ ] Streaming token output in the CLI
- [ ] Rate limiting & retry/backoff for Groq and Tavily
- [ ] Docker secrets / `.env` injection without hardcoded URLs

---

## 🤝 Contributing

Contribations welcome — open an issue or PR.

1. Fork & clone
2. Create a feature branch: `git checkout -b feature/amazing`
3. Commit: `git commit -m "Add amazing feature"`
4. Push & open a PR

---

## 📄 License

Released under the **MIT License** — see [`LICENSE`](LICENSE).

```
Copyright (c) 2026 Yogeshwaran D
```

---

## 👤 Author

**Yogeshwaran D** — [github.com/Yogeshwaran197](https://github.com/Yogeshwaran197)

Project repo: [https://github.com/Yogeshwaran197/NexusChat](https://github.com/Yogeshwaran197/NexusChat)

---

<div align="center">

**NexusChat** · *Guardrails → Validate → Rewrite → Generate → Ground → Human-approved Search → Answer*

Made with 🐍 Python, 🕸️ LangGraph, ⚡ Groq, 🌐 Tavily MCP & 🐘 PostgreSQL

</div>
