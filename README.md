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
# OPINIONS: Should I bootstrap or raise VC?
_2025-02-25 14:30 · 3 perspectives · 12 opinions retrieved_

## Retrieved Opinions

### Naval Ravikant (individual)
> "Bootstrapping forces you to focus on revenue from day one..."
> _— twitter, relevance: 0.87_

### Indie Hackers (group)
> "The freedom of not having a board to answer to is underrated..."
> _— reddit, relevance: 0.82_

---

## Synthesis

### Pros of Bootstrapping
- Full ownership and control
- Revenue focus from day one
- Freedom from board pressure

### Pros of VC
- Faster growth potential
- Access to network/resources
- Can tackle larger markets

### Constructive Feedback
Consider your market size and personal runway. If you're in a winner-take-all 
market, VC might be necessary. If you can reach profitability quickly with 
a niche product, bootstrap.
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
