# Cognitive Agent System - Session 6

A modular AI agent built on a **four-role cognitive architecture** that orchestrates Memory, Perception, Decision, and Action to solve complex multi-step tasks.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      AGENT.PY                               │
│                  (Main Orchestrator)                        │
│  • Manages iteration loop (max 20 iterations)               │
│  • Coordinates all four cognitive roles                     │
│  • Handles MCP tool session                                 │
└─────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   MEMORY     │  │  ARTIFACTS   │  │ LLM GATEWAY  │
│  (Singleton) │  │  (Singleton) │  │   (HTTP)     │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Four Cognitive Roles

1. **MEMORY** (`memory.py`) - Persistent Storage
   - Keyword-based retrieval (no LLM cost)
   - LLM classification for ambiguous content
   - Stores: facts, preferences, tool outcomes, scratchpad notes
   - Storage: `state/memory.json`

2. **PERCEPTION** (`perception.py`) - Goal Orchestrator
   - Decomposes queries into bounded goals
   - Maintains goal state with "sticky-done" flags
   - Attaches artifacts to goals when needed
   - Uses: Gemini LLM with structured output

3. **DECISION** (`decision.py`) - Action Picker
   - Chooses next action for one goal
   - Returns: answer OR tool call (never both)
   - Receives artifact content for analysis
   - Uses: Gemini LLM with tool-calling

4. **ACTION** (`action.py`) - Tool Executor
   - Dispatches MCP tools (no LLM)
   - Guards against artifact handles in arguments
   - Creates artifacts for large results (>4KB)
   - Returns: result text + optional artifact ID

5. **ARTIFACTS** (`artifacts.py`) - Binary Store
   - Content-addressable storage (SHA-256)
   - Stores large tool results separately
   - Format: `art:<sha256-prefix>`
   - Storage: `state/artifacts/`

## 🔄 Execution Flow

```
USER QUERY
    ↓
┌─────────────────────────────────────────────────────────┐
│ INITIALIZATION                                          │
│ • memory.remember(query) - Classify and store query     │
│ • Start MCP session                                     │
│ • Load available tools                                  │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ ITERATION LOOP (1-20)                                   │
│                                                         │
│ STEP 1: MEMORY READ                                    │
│ • memory.read(query, history)                          │
│ • Returns: list[MemoryItem] (keyword search)           │
│                                                         │
│ STEP 2: PERCEPTION                                     │
│ • perception.observe(query, hits, history, goals)      │
│ • Decomposes query into goals (first call)             │
│ • Marks goals done based on history                    │
│ • Attaches artifact to next unfinished goal            │
│ • Returns: Observation (updated goals)                 │
│                                                         │
│ CHECK: All goals done? → YES: EXIT LOOP                │
│                        → NO: CONTINUE                   │
│                                                         │
│ AGENT: Load artifact bytes if attached                 │
│ • artifacts.get_bytes(artifact_id)                     │
│                                                         │
│ STEP 3: DECISION                                       │
│ • decision.next_step(goal, hits, attached, tools)      │
│ • Receives artifact content in prompt                  │
│ • Returns: answer OR tool_call                         │
│                                                         │
│ IF ANSWER:                                             │
│ • Record to history                                    │
│ • Continue loop                                        │
│                                                         │
│ IF TOOL CALL:                                          │
│ STEP 4: ACTION                                         │
│ • action.execute(session, tool_call)                   │
│ • Guards against "art:" in arguments                   │
│ • Dispatches tool via MCP                              │
│ • Creates artifact if result >4KB                      │
│ • Returns: (result_text, artifact_id)                  │
│                                                         │
│ • memory.record_outcome() - Store result               │
│ • Append to history                                    │
│ • Continue loop                                        │
└─────────────────────────────────────────────────────────┘
    ↓
FINAL ANSWER (extracted from history)
```

## 📊 Data Flow: Artifacts

**Key Insight**: Artifacts flow from Perception → Agent → Decision (NOT to Action)

```
ITERATION 1: Fetch Wikipedia page
├─ Decision: Calls fetch_url("https://en.wikipedia.org/wiki/Claude_Shannon")
└─ Action: Fetches 85KB HTML → Creates art:abc123

ITERATION 2: Extract information
├─ Memory: Returns MemoryItem with artifact_id="art:abc123"
├─ Perception: Sets goal.attach_artifact_id = "art:abc123"
│              (Decides: "next goal needs this content")
├─ AGENT: Loads bytes = artifacts.get_bytes("art:abc123")
│         Creates attached = [("art:abc123", bytes)]
├─ Decision: Receives 85KB Wikipedia HTML in prompt
│            LLM reads content, extracts birth date
│            Returns answer: "Claude Shannon was born April 30, 1916"
└─ Action: NOT CALLED (Decision returned answer)
```

**Why?** Decision needs to READ artifact content to answer questions. Action only CREATES new artifacts from tool results.

## 🔑 Key Invariants

1. **Memory First**: `memory.remember(query)` runs BEFORE loop
2. **Memory Every Iteration**: `memory.read()` at TOP of each iteration
3. **Perception Always Runs**: Maintains goal state every iteration
4. **One Goal at a Time**: Decision works on single unfinished goal
5. **Sticky-Done**: Goals never flip from done→undone
6. **No Artifact Handles in Tools**: Action guards against "art:" in arguments
7. **Threshold-Based Artifacts**: Results >4KB automatically stored

## 🛠️ Setup & Usage

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Start LLM Gateway
cd llm_gatewayV3
./run.sh
# Gateway runs at http://localhost:8101
```

### Running the Agent
```python
import asyncio
from agent import run

# Single query
result = asyncio.run(run("Fetch Wikipedia page on Claude Shannon and tell me his birth date"))
print(result)

# Multiple queries (from agent.py)
queries = [
    "Find 3 family-friendly things to do in Tokyo this weekend.",
    "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date.",
    "My mom's birthday is 15 May 2026. Remember that.",
    "Search for 'Python asyncio best practices', read top 3 results."
]

for query in queries:
    result = asyncio.run(run(query))
    print(f"\nQUERY: {query}")
    print(f"ANSWER: {result}\n")
```

### Command Line
```bash
python agent.py "Your query here"
```

## 📁 Project Structure

```
.
├── agent.py           # Main orchestrator
├── memory.py          # Memory service (keyword search + LLM classification)
├── perception.py      # Goal orchestrator (LLM-based)
├── decision.py        # Action picker (LLM-based)
├── action.py          # Tool executor (no LLM)
├── artifacts.py       # Binary artifact store
├── schemas.py         # Pydantic data models
├── mcp_server.py      # MCP tool server
├── state/
│   ├── memory.json    # Persistent memory
│   └── artifacts/     # Binary artifacts (.bin + .json)
└── llm_gatewayV3/     # LLM Gateway service
```

## 🧠 LLM Usage

| Role | Provider | Temperature | Purpose |
|------|----------|-------------|---------|
| Perception | Gemini | 1.0 | Structured goal decomposition |
| Decision | Gemini | 0.7 | Tool-calling and reasoning |
| Memory | Gemini | 0.3 | Classification and relevance |

**Retry Logic**: All LLM calls retry 4 times with exponential backoff (5s, 10s, 20s) for 502/503/429 errors.

## 🎯 Example: Multi-Step Query

**Query**: "Fetch Wikipedia page on Claude Shannon and tell me his birth date"

```
ITERATION 1:
├─ Memory: (empty)
├─ Perception: Goal 1: "Fetch Wikipedia page on Claude Shannon"
├─ Decision: Calls fetch_url("https://en.wikipedia.org/wiki/Claude_Shannon")
└─ Action: Fetches 85KB HTML → Creates art:abc123

ITERATION 2:
├─ Memory: Returns tool_outcome with art:abc123
├─ Perception: Attaches art:abc123 to Goal 1
├─ Agent: Loads 85KB HTML bytes
├─ Decision: Reads HTML, extracts "April 30, 1916"
│            Returns answer
└─ Goal 1 marked DONE

FINAL ANSWER: "Claude Shannon was born on April 30, 1916"
```

## 🔒 Safety Features

1. **Artifact Handle Guard**: Action refuses to dispatch tools with "art:" in arguments
2. **Sticky-Done**: Goals never regress from done to undone
3. **Positional Identity**: Goals identified by position, not IDs (anti-hallucination)
4. **Force-Attach**: Synthesis goals automatically get artifacts attached
5. **Content Validation**: Decision must fetch full content, not use search snippets

## 📝 Memory Types

- **fact**: Durable observed truth (dates, locations, relationships)
- **preference**: User preferences or habits
- **tool_outcome**: Results from MCP tool calls
- **scratchpad**: Temporary working notes

## 🔧 Configuration

```python
# agent.py
MAX_ITERATIONS = 20
GATEWAY_URL = "http://localhost:8101"

# action.py
ARTIFACT_THRESHOLD_BYTES = 4096  # 4KB threshold
```

## 🚀 Advanced Features

- **Content-Addressable Storage**: Artifacts deduplicated by SHA-256
- **Keyword Search**: Fast retrieval without LLM cost
- **Structured Output**: Gemini JSON schema for reliable parsing
- **MCP Integration**: Extensible tool system via Model Context Protocol
- **Persistent Memory**: Survives across runs

## 📊 Performance

- **Memory Read**: O(n) keyword search, no LLM cost
- **Perception**: 1 LLM call per iteration
- **Decision**: 1 LLM call per iteration
- **Action**: No LLM cost, pure execution
- **Total**: ~2 LLM calls per iteration

## 🐛 Debugging

Enable verbose logging:
```python
# Each module prints with [module_name] prefix
# Example output:
[agent] === ITERATION 1 ===
[agent] Memory hits: 3
[agent] Perception Goals:
  - [OPEN] Fetch Wikipedia page
[agent] Decision: Call Tool 'fetch_url'
[action] Created Artifact: art:abc123
```

## 📚 References

- **MCP Protocol**: Model Context Protocol for tool integration
- **LLM Gateway V3**: Unified interface for multiple LLM providers
- **Cognitive Architecture**: Four-role separation of concerns

---

**Made with Bob** 🤖