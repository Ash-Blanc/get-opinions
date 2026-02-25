# OPINIONS

Get perspectives on any idea. Just input your thought.

An agent-powered system that retrieves real opinions from influential individuals and communities, then synthesizes them into pros, cons, and constructive feedback.

## Quick Start

```bash
# Install
pip install get-opinions

# Just input your idea - personas are auto-selected
opinions "Should I bootstrap or raise VC funding?"

# Or specify personas manually
opinions "AI agents will replace developers" --personas "karpathy,hn,naval"
```

## How It Works

1. **Input your idea** - Any thought, concept, or question
2. **Auto-select personas** - Agent picks 2-4 relevant perspectives from:
   - **Individuals**: Andrej Karpathy, Elon Musk, Naval Ravikant, Garry Tan, Sam Altman, Paul Graham, Balaji Srinivasan, Patrick Collison
   - **Groups**: Tech Twitter, Hacker News, Reddit Tech, Kaggle, Indie Hackers, VC Twitter
3. **Retrieve opinions** - RAG search across indexed real opinions with embeddings
4. **Synthesize** - Get structured pros, cons, and constructive feedback

## Features

- **RAG-based retrieval** - Vector similarity search over real opinions
- **Agentica-powered agents** - Smart persona selection and opinion synthesis
- **Auto-indexing** - Builds persona indices on first use
- **Mistral embeddings** - With OpenRouter fallback

## Commands

```bash
opinions "your idea here"           # Auto-select personas
opinions ask "your idea"            # Explicit ask command
opinions run "idea" "karpathy,hn"   # Specify personas
opinions build "Elon Musk"          # Build a persona index
opinions build "HN" --type group    # Build group index
opinions list                       # List built indices
opinions clear --confirm            # Clear all indices
```

## Environment Variables

```bash
EXA_API_KEY=your-exa-key           # Required: Web search
MISTRAL_API_KEY=your-mistral-key   # Embeddings (primary)
OPENROUTER_API_KEY=your-key        # Embeddings fallback + LLM
```

## Architecture

```
opinions/
├── agents.py      # Agentica agents (selection, synthesis, extraction)
├── tools.py       # Embeddings, search, scraping utilities
├── personas.py    # Persona index management
├── pipeline.py    # RAG retrieval pipeline
└── main.py        # CLI interface
```

Each persona has an indexed collection of opinions with embeddings stored in `opinions/{persona_id}.json`.

## License

MIT
