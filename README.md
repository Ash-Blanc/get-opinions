# OPINIONS

Get perspectives on any idea. Just input your thought.

An agent-powered system that **dynamically discovers relevant voices on the web**, retrieves real opinions, and synthesizes them into a clear verdict with pros, cons, and a bottom line.

## Installation

```bash
git clone https://github.com/Ash-Blanc/get-opinions.git
cd get-opinions
uv tool install -e .
```

## Setup

```bash
cp .env.example .env
```

Add your API keys to `.env`:

| Variable | Required | Purpose |
|---|---|---|
| `EXA_API_KEY` | Yes | Semantic web search |
| `PARALLEL_API_KEY` | Yes | Fast search + content extraction |
| `MISTRAL_API_KEY` | Recommended | Embeddings (primary) |
| `OPENROUTER_API_KEY` | Yes | LLM synthesis + agent models |
| `CEREBRAS_API_KEY` | Optional | High-performance LLM fallback |

## Quick Start

```bash
# Just ask — personas are discovered automatically
opinions ask "how to win competitive hackathons in the AI era"

# Ask about startup decisions
opinions ask "should I bootstrap or raise VC funding?"

# Ask about technical topics
opinions ask "is Rust worth learning in 2025?"
```

## How It Works

```
Your Question
    → Parallel Search discovers who has direct experience
    → Indices built from their real discussions
    → RAG retrieves most relevant opinions
    → Synthesis gives you TL;DR, pros/cons, bottom line
```

1. **Dynamic Persona Discovery** — Searches the web to find people who've actually done the thing (not a hardcoded list of influencers)
2. **Opinion Indexing** — Scrapes real discussions, cleans content, generates embeddings, stores locally
3. **Vector Search** — Retrieves the most semantically relevant opinions via cosine similarity
4. **Synthesis** — Structured verdict: TL;DR → What People Like → Concerns → Key Tensions → Bottom Line

## Output Format

```markdown
# OPINIONS: how to win competitive hackathons in the AI era

---

## TL;DR
Winning requires solving a real problem fast, not just adding AI. ...

## What People Like
- Build a working demo first — judges care about function over form
- "AI fluency" across the stack matters more than any single model

## Concerns Raised
- "Just add AI" is not a strategy — most teams fail on problem clarity
- Human verification is non-negotiable; AI-generated code has silent bugs

## Key Tensions
Technical depth vs. pitch quality — some winners had rough demos but nailed the story.

## Bottom Line
Pick a relatable problem, build the simplest thing that works, budget time for the pitch.
```

## CLI Reference

```bash
# Ask for opinions (auto-discovers personas)
opinions ask "your question here"

# Build a persona index manually
opinions build "Pieter Levels" --queries "pieter levels bootstrap,levels.io advice"

# List all built indices
opinions list

# Clear all indices (forces fresh builds next run)
opinions clear
```

## Architecture

```
opinions/
├── agents.py     # Agentica agents (synthesis, extraction)
├── tools.py      # Parallel/Exa search, embeddings, content cleaning
├── personas.py   # Persona index management
├── pipeline.py   # RAG retrieval pipeline
└── main.py       # CLI (cli2)
```

Data stored locally:
- `opinions/` — Persona indices with embeddings
- `reports/` — Generated markdown reports

## To-Do / Ideas (Contributing)

We welcome contributions! Here are some ideas for future features and use cases:
- **Full-Stack Next.js App**: A web interface for the `get-opinions` engine for VCs/investors due diligence.
- **Web Sentiment Engine**: A broader tool to track and analyze the general sentiment of the web regarding a specific topic, product, or trend over time.
- **Investor Due Diligence Workflows**: Specialized reports and workflows to evaluate founders, startups, and market sentiment.

## License

MIT
