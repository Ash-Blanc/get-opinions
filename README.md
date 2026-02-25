# OPINIONS

Get perspectives on any idea. Just input your thought.

An agent-powered system that retrieves real opinions from influential individuals and communities, then synthesizes them into pros, cons, and constructive feedback.

## Installation

```bash
# Clone and install
git clone https://github.com/Ash-Blanc/get-opinions.git
cd get-opinions
uv sync

# Or with pip
pip install get-opinions
```

## Setup

1. Copy the example env file:
```bash
cp .env.example .env
```

2. Add your API keys to `.env`:
```bash
# Required
EXA_API_KEY=your-exa-api-key        # Get at https://exa.ai
MISTRAL_API_KEY=your-mistral-key    # Get at https://mistral.ai
OPENROUTER_API_KEY=your-key         # Get at https://openrouter.ai
```

## Quick Start

```bash
# Just input your idea - personas are auto-selected
opinions "Should I bootstrap my startup or raise VC funding?"

# That's it! The agent will:
# 1. Select relevant personas (e.g., Naval, Indie Hackers, VC Twitter)
# 2. Build indices if needed
# 3. Retrieve relevant opinions
# 4. Synthesize into pros, cons, feedback
```

## Usage Examples

### Basic Usage (Auto-Select Personas)

```bash
# Ask about a startup decision
opinions "Should I build in public or stealth mode?"

# Ask about AI/ML topics
opinions "Will AI agents replace software engineers?"

# Ask about career decisions
opinions "Is it worth doing a PhD in ML right now?"

# Ask about product ideas
opinions "Building a competitor to Notion - good idea?"
```

### Specify Personas Manually

```bash
# Get technical perspectives
opinions "Rust vs Go for backend services" --personas "karpathy,hn"

# Get investor + builder views
opinions "My SaaS idea" --personas "naval,garry,vc-twitter,indie-hackers"

# Mix individuals and groups
opinions "AI regulation" --personas "sam,musk,hn,tech-twitter"
```

### Build Persona Indices

```bash
# Build individual persona index
opinions build "Andrej Karpathy"

# Build group/community index
opinions build "Hacker News" --type group

# Custom search queries
opinions build "Elon Musk" --queries "elon musk twitter,elon interview,elon thoughts"

# Faster build without agent extraction
opinions build "Naval" --no-agent
```

### Manage Indices

```bash
# List all built indices
opinions list

# Get stats for specific index
opinions list --stats karpathy

# Clear all indices
opinions clear --confirm
```

## Available Personas

### Individuals
| ID | Name | Expertise |
|---|---|---|
| `karpathy` | Andrej Karpathy | AI/ML, Tesla, practical ML |
| `musk` | Elon Musk | Tech vision, AI, EVs |
| `naval` | Naval Ravikant | Wealth, startups, philosophy |
| `garry` | Garry Tan | YC, startups, building |
| `sam` | Sam Altman | OpenAI, AGI, startups |
| `paul` | Paul Graham | Essays, startups, YC |
| `balaji` | Balaji Srinivasan | Crypto, network states |
| `patel` | Patrick Collison | Stripe, progress, infrastructure |

### Groups/Communities
| ID | Name | Perspective |
|---|---|---|
| `tech-twitter` | Tech Twitter | Rapid takes, hype-aware |
| `hn` | Hacker News | Technical, skeptical |
| `reddit-tech` | Reddit Tech | Worker-perspective, depth |
| `kaggle` | Kaggle Forums | Competitive ML, pragmatic |
| `indie-hackers` | Indie Hackers | Bootstrappers, revenue-focused |
| `vc-twitter` | VC Twitter | Investor mindset, markets |

## How It Works

```
Your Idea → Agent Selects Personas → RAG Search → Synthesized Output
                ↓                         ↓
         "karpathy, hn"           Vector similarity over
         "naval, tech-twitter"    real opinion embeddings
```

1. **Persona Selection Agent** - Analyzes your topic and picks 2-4 relevant personas
2. **Opinion Indexing** - Scrapes real discussions, generates embeddings, stores locally
3. **Vector Search** - Retrieves top-k most relevant opinions using cosine similarity
4. **Synthesis Agent** - Structures opinions into pros, cons, constructive feedback

## Architecture

```
opinions/
├── agents.py      # Agentica agents (selection, synthesis, extraction)
├── tools.py       # Embeddings, search, scraping utilities  
├── personas.py    # Persona index management
├── pipeline.py    # RAG retrieval pipeline
└── main.py        # CLI with cli2
```

Data stored locally in:
- `opinions/` - Persona indices with embeddings
- `reports/` - Generated reports

## Output Example

```markdown
# OPINIONS: how to win any big competitive hackathon in ai age
_2026-02-25 20:03 · 2 indices · 15 opinions_

---

## TL;DR
The consensus is that winning AI hackathons requires a strategic approach that goes beyond just adding AI. Key elements include solving a problem everyone understands, embracing outside perspectives for feedback, and being "AI-fluent" across the entire development stack.

## What People Like
- **Solving Real Pain First:** The most important factor is choosing a problem that resonates universally and addresses a genuine need.
- **AI Fluency:** Winning teams understand how to collaborate effectively with AI, focusing on mastering techniques that work across various AI systems.

## Concerns Raised
- **Over-reliance on AI Output:** A critical truth is that AI can generate syntactically perfect code that contains logical errors. Human verification is "non-negotiable".
- **"Just Add AI" is Not a Strategy:** Teams that focus on technology alone, without a strong underlying strategy for problem-solving, are less likely to win.

## Key Tensions
Opinions diverge on the balance between technical implementation and the clarity of the idea and its presentation. Some emphasize functional demos, while others stress a compelling pitch.

## Bottom Line
To win an AI hackathon, focus on solving a relatable problem with a clear strategy, leverage AI tools for speed, but always budget time for human verification and compelling presentation.

---

## Key Opinions Retrieved

### karpathy
> Technology is just the enabler — strategy is everything else.
> — *twitter, relevance: 0.82*

### hn
> The freedom of not having a board to answer to is underrated...
> — *hn, relevance: 0.79*

---

## Sources
- [karpathy](https://twitter.com/karpathy/status/123456789)
- [hn](https://news.ycombinator.com/item?id=123456)
```

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `EXA_API_KEY` | Yes | Web search for opinions |
| `MISTRAL_API_KEY` | Yes | Embeddings (primary) |
| `OPENROUTER_API_KEY` | Yes | Embeddings fallback + LLM synthesis |
| `FIRECRAWL_API_KEY` | No | Better scraping quality |

## License

MIT
