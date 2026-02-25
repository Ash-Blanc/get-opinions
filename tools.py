import os
import json
import re
import time
import hashlib
import urllib.request
import urllib.error
from dotenv import load_dotenv
from exa_py import Exa

load_dotenv()

# Initialize services
exa = Exa(api_key=os.environ.get("EXA_API_KEY", ""))
PARALLEL_KEY = os.environ.get("PARALLEL_API_KEY", "")
MISTRAL_KEY = os.environ.get("MISTRAL_API_KEY", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
CEREBRAS_KEY = os.environ.get("CEREBRAS_API_KEY", "")
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "mistral-embed")

# Embedding cache
EMBEDDING_CACHE: dict[str, list[float]] = {}
EMBEDDING_CACHE_HITS = 0
EMBEDDING_CACHE_MISSES = 0

# High-performance models
HIGH_PERFORMANCE_MODELS = {
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "models": {
            "text-embedding-3-small": "zai-glm-4.7",  # Fast embeddings
            "text-embedding-3-large": "zai-glm-4.7",
            "mistral-embed": "zai-glm-4.7",
            "openai/text-embedding-3-small": "zai-glm-4.7",
        },
    }
}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def generate_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


# ── Content cleaning ──────────────────────────────────────────

_JUNK_PATTERNS = re.compile(
    r"|".join([
        r"Section Title:.*",           # Parallel extract navigation headers
        r"Content:\s*$",               # Empty content markers
        r"\[Open in app\].*",          # App store links
        r"Skip to main content.*",     # Skip-nav
        r"Open menu.*",               # Menu triggers
        r"Expand user menu.*",
        r"Go to Reddit Home.*",
        r"Iniciar sesión.*",           # Non-English UI
        r"Regístrate.*",
        r"Registrarse con.*",
        r"Descargar la app.*",
        r"No te pierdas lo que.*",
        r"Ver posts nuevos.*",
        r"Ir al contenido principal.*",
        r"Abrir menú.*",
        r"Abrir navegación.*",
        r"Las personas en X son.*",
        r"\| --- \|.*",               # Table separators
        r"\|\|.*\|.*prev.*next.*",     # HN thread navigation
        r"reply \|",                  # HN reply markers
        r"Califique las Notas.*",
        r"Traducir post",
        r"Write a response",
        r"What are your thoughts\?",
        r"Cancel\s*Respond",
        r"See all responses",
        r"Follow Us On Social Media",
        r"Share\s*$",
        r"Sitemap\s*$",
    ]),
    re.IGNORECASE | re.MULTILINE,
)

_LINK_RE = re.compile(r"\[([^\]]*)\]\(http[^)]+\)")
_URL_RE = re.compile(r"https?://\S+")


def clean_text(text: str) -> str:
    """Strip scraping artifacts, nav HTML, and boilerplate from raw content."""
    # Remove junk lines
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if _JUNK_PATTERNS.search(line):
            continue
        # Skip lines that are mostly markdown links or bare URLs
        stripped = _LINK_RE.sub("", line)
        stripped = _URL_RE.sub("", stripped)
        if len(stripped.strip()) < 15 and len(line) > 30:
            continue
        cleaned.append(line)

    result = "\n".join(cleaned)
    # Collapse multiple newlines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _get_embedding_cache_key(text: str) -> str:
    """Generate a cache key for embedding based on text content."""
    return generate_id(text)


async def get_embedding(text: str) -> list[float]:
    """Get embedding using the most efficient available method with caching."""
    global EMBEDDING_CACHE_HITS, EMBEDDING_CACHE_MISSES

    # Check cache first
    cache_key = _get_embedding_cache_key(text)
    if cache_key in EMBEDDING_CACHE:
        EMBEDDING_CACHE_HITS += 1
        return EMBEDDING_CACHE[cache_key]

    EMBEDDING_CACHE_MISSES += 1

    # Try Cerebras first (fastest)
    if CEREBRAS_KEY:
        try:
            embedding = await _get_embedding_cerebras(text)
            EMBEDDING_CACHE[cache_key] = embedding
            return embedding
        except Exception:
            pass  # Fall back to other methods

    # Try Mistral (good balance of speed/quality)
    if MISTRAL_KEY:
        try:
            embedding = await _get_embedding_mistral(text)
            EMBEDDING_CACHE[cache_key] = embedding
            return embedding
        except Exception:
            pass  # Fall back to other methods

    # Try OpenRouter (reliable fallback)
    if OPENROUTER_KEY:
        try:
            embedding = await _get_embedding_openrouter(text)
            EMBEDDING_CACHE[cache_key] = embedding
            return embedding
        except Exception:
            pass  # Last resort

    raise ValueError("No embedding provider available or all failed")


async def _get_embedding_cerebras(text: str) -> list[float]:
    """Get embedding via Cerebras API (fastest option)."""
    if not CEREBRAS_KEY:
        raise ValueError("CEREBRAS_API_KEY not set")

    # Use the most efficient model for embeddings
    model = HIGH_PERFORMANCE_MODELS["cerebras"]["models"].get(
        EMBEDDINGS_MODEL, "zai-glm-4.7"
    )

    req = urllib.request.Request(
        f"{HIGH_PERFORMANCE_MODELS['cerebras']['base_url']}/embeddings",
        headers={
            "Authorization": f"Bearer {CEREBRAS_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/get-opinions",
            "X-Title": "get-opinions",
        },
        data=json.dumps(
            {
                "model": model,
                "input": text[:8000],
                "stream": False,
            }
        ).encode("utf-8"),
    )

    with urllib.request.urlopen(req, timeout=20) as response:
        res = json.loads(response.read().decode("utf-8"))
        return res["data"][0]["embedding"]


async def _get_embedding_mistral(text: str) -> list[float]:
    """Get embedding via Mistral API."""
    if not MISTRAL_KEY:
        raise ValueError("MISTRAL_API_KEY not set")

    req = urllib.request.Request(
        "https://api.mistral.ai/v1/embeddings",
        headers={
            "Authorization": f"Bearer {MISTRAL_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(
            {
                "model": "mistral-embed",
                "input": text[:8000],
            }
        ).encode("utf-8"),
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        res = json.loads(response.read().decode("utf-8"))
        return res["data"][0]["embedding"]


async def _get_embedding_openrouter(text: str) -> list[float]:
    """Get embedding via OpenRouter (uses OpenAI models)."""
    if not OPENROUTER_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/get-opinions",
            "X-Title": "get-opinions",
        },
        data=json.dumps(
            {
                "model": "openai/text-embedding-3-small",
                "input": text[:8000],
            }
        ).encode("utf-8"),
    )
    with urllib.request.urlopen(req, timeout=25) as response:
        res = json.loads(response.read().decode("utf-8"))
        return res["data"][0]["embedding"]


async def get_high_performance_llm(
    prompt: str,
    model: str = "zai-glm-4.7",
    max_tokens: int = 4000,
    reasoning_effort: str = "medium",
) -> dict:
    """Get LLM response using the most efficient available method.

    Optimized for speed with fallback options.

    Args:
        prompt: The user prompt
        model: Preferred model (may be substituted based on provider)
        max_tokens: Maximum tokens to generate
        reasoning_effort: Reasoning effort level (low/medium/high)
    """
    errors = []

    # Try Cerebras first (fastest)
    if CEREBRAS_KEY:
        try:
            return await _call_cerebras_llm(prompt, model, max_tokens, reasoning_effort)
        except Exception as e:
            errors.append(f"Cerebras: {e}")

    # Try Parallel (high-accuracy, optimized for AI agents)
    if PARALLEL_KEY:
        parallel_models = {
            "zai-glm-4.7": "speed",  # Fast inference
            "llama3.1-8b": "speed",
            "gpt-oss-120b": "quality",  # Deep analysis
        }
        p_model = parallel_models.get(model, "speed")
        try:
            return await _call_parallel_llm(prompt, p_model, max_tokens)
        except Exception as e:
            errors.append(f"Parallel: {e}")

    # Try OpenRouter (reliable fallback)
    if OPENROUTER_KEY:
        openrouter_models = {
            "zai-glm-4.7": "anthropic/claude-3-haiku",
            "llama3.1-8b": "meta-llama/llama-3.1-8b-instruct",
            "gpt-oss-120b": "openai/gpt-3-turbo",
        }
        or_model = openrouter_models.get(model, "anthropic/claude-3-haiku")
        try:
            return await _call_openrouter_llm(prompt, or_model, max_tokens)
        except Exception as e:
            errors.append(f"OpenRouter: {e}")

    raise ValueError(f"All LLM providers failed: {'; '.join(errors)}")


async def _call_cerebras_llm(
    prompt: str,
    model: str = "zai-glm-4.7",
    max_tokens: int = 4000,
    reasoning_effort: str = "medium",
) -> dict:
    """Call Cerebras LLM API."""
    if not CEREBRAS_KEY:
        raise ValueError("CEREBRAS_API_KEY not set")

    req = urllib.request.Request(
        f"{HIGH_PERFORMANCE_MODELS['cerebras']['base_url']}/chat/completions",
        headers={
            "Authorization": f"Bearer {CEREBRAS_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/get-opinions",
            "X-Title": "get-opinions",
        },
        data=json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort,
                "stream": False,
            }
        ).encode("utf-8"),
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


async def _call_openrouter_llm(
    prompt: str, model: str = "zai-glm-4.7", max_tokens: int = 4000
) -> dict:
    """Call OpenRouter LLM API."""
    if not OPENROUTER_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/get-opinions",
            "X-Title": "get-opinions",
        },
        data=json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": False,
            }
        ).encode("utf-8"),
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


async def _call_parallel_llm(
    prompt: str, model: str = "speed", max_tokens: int = 4000
) -> dict:
    """Call Parallel LLM API."""
    if not PARALLEL_KEY:
        raise ValueError("PARALLEL_API_KEY not set")

    req = urllib.request.Request(
        "https://api.parallel.ai/chat/completions",
        headers={
            "Authorization": f"Bearer {PARALLEL_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/get-opinions",
            "X-Title": "get-opinions",
        },
        data=json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": False,
            }
        ).encode("utf-8"),
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


async def parallel_search(
    queries: list[str],
    objective: str = "",
    max_results: int = 10,
    max_chars_per_result: int = 5000,
) -> list[dict]:
    """Search the web using Parallel.ai Search API.

    Returns LLM-optimized excerpts ranked by reasoning utility.
    Much higher quality than traditional search for opinion discovery.

    Args:
        queries: Search queries to execute
        objective: Natural language context for what we're looking for
        max_results: Maximum number of results to return
        max_chars_per_result: Maximum chars per excerpt

    Returns:
        List of {url, title, publish_date, excerpts[]}
    """
    if not PARALLEL_KEY:
        return []

    body = {
        "search_queries": queries[:5],  # API limit
        "max_results": max_results,
        "excerpts": {"max_chars_per_result": max_chars_per_result},
    }
    if objective:
        body["objective"] = objective

    req = urllib.request.Request(
        "https://api.parallel.ai/v1beta/search",
        headers={
            "x-api-key": PARALLEL_KEY,
            "Content-Type": "application/json",
            "parallel-beta": "search-extract-2025-10-10",
        },
        data=json.dumps(body).encode("utf-8"),
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res.get("results", [])
    except Exception as e:
        print(f"  ↳ Parallel search failed: {e}")
        return []


async def parallel_extract(
    urls: list[str],
    objective: str = "",
    max_chars_per_result: int = 5000,
) -> list[dict]:
    """Extract content from URLs using Parallel.ai Extract API.

    Pulls focused, objective-aligned excerpts from specific pages.

    Args:
        urls: URLs to extract content from
        objective: What to focus on when extracting
        max_chars_per_result: Max chars per extracted result

    Returns:
        List of {url, title, excerpts[]}
    """
    if not PARALLEL_KEY:
        return []

    body = {
        "urls": urls[:10],  # Reasonable batch limit
        "excerpts": True,
    }
    if objective:
        body["objective"] = objective

    req = urllib.request.Request(
        "https://api.parallel.ai/v1beta/extract",
        headers={
            "x-api-key": PARALLEL_KEY,
            "Content-Type": "application/json",
            "parallel-beta": "search-extract-2025-10-10",
        },
        data=json.dumps(body).encode("utf-8"),
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res.get("results", [])
    except Exception as e:
        print(f"  ↳ Parallel extract failed: {e}")
        return []


async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings for multiple texts using the cached provider chain."""
    if not texts:
        return []
    return [await get_embedding(text) for text in texts]


async def discover_personas_for_topic(topic: str) -> list[dict]:
    """Use Parallel Search to discover real voices relevant to the topic.

    Returns list of {name, search_queries} for dynamically discovered personas.
    """
    if not PARALLEL_KEY:
        return []

    print(f"  🔍 Discovering relevant voices via Parallel Search...")

    # Focus on ADVICE and STRATEGY, not participants or news coverage
    discovery_results = await parallel_search(
        queries=[
            f"{topic} advice tips how to strategy",
            f"{topic} lessons learned what worked what didn't",
            f"{topic} reddit discussion personal experience",
        ],
        objective=f"Find blog posts, forum discussions, and personal accounts from "
                  f"people sharing advice, tips, and strategies about: {topic}. "
                  f"IGNORE: press releases, product pages, team rosters, university news, "
                  f"job titles, and anything that is NOT someone sharing their personal "
                  f"opinion or advice. Prioritize first-person 'how I did X' posts.",
        max_results=10,
        max_chars_per_result=3000,
    )

    if not discovery_results:
        return []

    # Use LLM to extract persona names and generate targeted queries
    excerpts_for_llm = []
    for r in discovery_results[:8]:
        title = r.get("title", "")
        url = r.get("url", "")
        excs = r.get("excerpts", [])
        preview = clean_text("\n".join(excs))[:400] if excs else ""
        excerpts_for_llm.append(f"Title: {title}\nURL: {url}\nExcerpt: {preview}")

    extract_prompt = f"""From these search results about "{topic}", identify 2-3 people or communities who SHARE ADVICE about this topic.

SEARCH RESULTS:
{chr(10).join(excerpts_for_llm)}

For each persona, return JSON:
[
  {{"name": "Person Name or Community", "search_queries": ["query1 about their advice", "query2 targeting their tips"]}}
]

RULES:
- ONLY pick people who share ADVICE, TIPS, or PERSONAL EXPERIENCE
- REJECT organizers, employees, product managers, press releases, team rosters
- REJECT anyone whose content is just news coverage or announcements
- If a Reddit/HN thread has rich discussion, include that subreddit as a persona
- Make search queries target their OPINIONS and ADVICE, not their bio or team page
- Return valid JSON array only, no explanation"""

    try:
        result = await get_high_performance_llm(extract_prompt, max_tokens=1000)
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Parse JSON from response
        # Handle markdown code blocks
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        personas = json.loads(content.strip())
        if isinstance(personas, list) and personas:
            personas = personas[:3]  # Cap at 3 to keep builds fast
            for p in personas:
                print(f"    Found: {p.get('name', '?')}")
            return personas
    except Exception as e:
        print(f"    ↳ Persona extraction failed: {e}")

    return []


async def select_personas_for_topic(topic: str) -> list[str]:
    """Discover relevant personas for a topic using web search.

    First tries dynamic discovery via Parallel Search to find actual
    relevant voices. Falls back to the hardcoded agent selector.
    """
    # Try dynamic discovery first
    discovered = await discover_personas_for_topic(topic)
    if discovered:
        # Store the discovery results for _build_default_index to use
        select_personas_for_topic._last_discovery = discovered
        return [p["name"] for p in discovered]

    # Fallback to agent-based selection from hardcoded list
    from agents import create_persona_selector_agent
    agent = create_persona_selector_agent()
    try:
        result = await agent.call(
            list[str],
            f"Select the most relevant personas for this topic:\n\n{topic}",
        )
        return result
    except Exception as e:
        print(f"  ↳ Persona selection failed: {e}, using defaults")
        return ["hn", "reddit_tech"]

# Store class-level attribute for passing discovery data
select_personas_for_topic._last_discovery = None


async def synthesize_with_agent(topic: str, opinions: list[dict]) -> str:
    """Use agentica agent to synthesize retrieved opinions."""
    from agents import create_synthesis_agent

    # Clean opinion text before sending to agent
    opinions_text = "\n\n".join(
        [
            f"[Source: {o['persona_name']}, {o['source_platform']}]\n{clean_text(o['opinion'])[:500]}"
            for o in opinions[:12]
        ]
    )

    agent = create_synthesis_agent()
    result = await agent.call(
        str,
        f"TOPIC: {topic}\n\nRETRIEVED OPINIONS:\n{opinions_text}",
    )
    return result


async def extract_opinions_with_agent(
    text: str, url: str, platform: str
) -> list["Opinion"]:
    """Use agentica agent to extract opinions from scraped content."""
    from agents import create_research_agent
    from personas import Opinion

    agent = create_research_agent()

    schema = {
        "type": "object",
        "properties": {
            "opinions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "quote": {"type": "string"},
                        "context": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["quote"],
                },
            }
        },
        "required": ["opinions"],
    }

    try:
        result = await agent.call(
            dict,
            f"Extract opinions from this content.\n\nURL: {url}\nPlatform: {platform}\n\nContent:\n{text[:5000]}",
        )

        opinions = []
        for item in result.get("opinions", []):
            quote = item.get("quote", "")
            if len(quote) < 50 or len(quote) > 2000:
                continue

            opinion = Opinion(
                id=generate_id(quote),
                text=quote,
                source_url=url,
                source_platform=platform,
                author="",
                context=item.get("context", ""),
                topic_tags=item.get("tags", []),
                embedding=[],
                created_at="",
            )
            opinions.append(opinion)

        return opinions
    except Exception as e:
        print(f"    ↳ Agent extraction failed: {e}, using fallback")
        return extract_opinions(text, url, platform)


async def build_persona_index(
    persona_id: str,
    persona_name: str,
    persona_type: str,
    search_queries: list[str],
    max_opinions: int = 200,
    use_embeddings: bool = True,
    use_agent: bool = True,
) -> "PersonaIndex":
    """
    Build an opinion index for a persona by scraping real discussions.
    Uses Parallel.ai Search (primary) and Exa (fallback) for discovery.
    Agentica agents can be used for higher quality opinion extraction.
    """
    from personas import PersonaIndex, Opinion, save_persona_index

    print(f"\n{'═' * 50}")
    print(f"  Building index: {persona_name}")
    print(f"  Type: {persona_type}")
    print(f"  Max opinions: {max_opinions}")
    print(f"  Use agent: {use_agent}")
    print(f"{'═' * 50}\n")

    index = PersonaIndex(
        id=persona_id,
        name=persona_name,
        type=persona_type,
        description=f"Opinion index for {persona_name}",
        search_queries=search_queries,
        opinions=[],
    )

    all_opinions = []

    # === PRIMARY: Parallel.ai Search (LLM-optimized, single-hop) ===
    if PARALLEL_KEY:
        objective = f"Find real opinions, discussions, and perspectives about {persona_name}. Focus on forums, social media posts, blog comments, and community discussions where people express genuine views."
        print(f"  ⚡ Using Parallel.ai Search (primary)")
        parallel_results = await parallel_search(
            queries=search_queries,
            objective=objective,
            max_results=15,
            max_chars_per_result=5000,
        )

        if parallel_results:
            print(f"    Found {len(parallel_results)} results via Parallel")
            for r in parallel_results:
                url = r.get("url", "")
                excerpts = r.get("excerpts", [])
                text = "\n\n".join(excerpts) if excerpts else ""
                if len(text) < 100:
                    continue

                platform = "web"
                if "twitter.com" in url or "x.com" in url:
                    platform = "twitter"
                elif "reddit.com" in url:
                    platform = "reddit"
                elif "news.ycombinator.com" in url:
                    platform = "hn"

                if use_agent:
                    chunks = await extract_opinions_with_agent(text, url, platform)
                else:
                    chunks = extract_opinions(
                        text,
                        url,
                        platform,
                        persona_name if persona_type == "individual" else "",
                    )
                all_opinions.extend(chunks)
        else:
            print(f"    ↳ Parallel search returned no results, falling back to Exa")

    # === FALLBACK: Exa search (if Parallel unavailable or returned nothing) ===
    if not all_opinions:
        print(f"  🔍 Using Exa Search {'(fallback)' if PARALLEL_KEY else '(primary)'}")
        for i, query in enumerate(search_queries, 1):
            print(f"  [{i}/{len(search_queries)}] Searching: {query}")

            try:
                results = exa.search_and_contents(
                    query=query,
                    num_results=15,
                    type="neural",
                    text={"max_characters": 3000},
                )

                for r in results.results:
                    text = r.text or ""
                    if len(text) < 100:
                        continue

                    platform = "web"
                    if "twitter.com" in r.url or "x.com" in r.url:
                        platform = "twitter"
                    elif "reddit.com" in r.url:
                        platform = "reddit"
                    elif "news.ycombinator.com" in r.url:
                        platform = "hn"

                    if use_agent:
                        chunks = await extract_opinions_with_agent(text, r.url, platform)
                    else:
                        chunks = extract_opinions(
                            text,
                            r.url,
                            platform,
                            persona_name if persona_type == "individual" else "",
                        )
                    all_opinions.extend(chunks)

            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue

            time.sleep(0.5)

    print(f"\n  Found {len(all_opinions)} raw opinions")

    if len(all_opinions) > max_opinions:
        all_opinions = all_opinions[:max_opinions]

    if use_embeddings and (MISTRAL_KEY or OPENROUTER_KEY):
        print("  Generating embeddings...")
        batch_size = 10
        for i in range(0, len(all_opinions), batch_size):
            batch = all_opinions[i : i + batch_size]
            texts = [o.text for o in batch]
            embeddings = await get_embeddings_batch(texts)
            for j, emb in enumerate(embeddings):
                if i + j < len(all_opinions):
                    all_opinions[i + j].embedding = emb
            print(
                f"    Embedded {min(i + batch_size, len(all_opinions))}/{len(all_opinions)}"
            )

    index.opinions = all_opinions
    save_persona_index(index)

    print(f"\n  ✓ Index saved: {len(all_opinions)} opinions")
    return index


def extract_opinions(
    text: str, url: str, platform: str, author_hint: str = ""
) -> list["Opinion"]:
    """Extract individual opinions from scraped text with aggressive junk filtering."""
    from personas import Opinion

    text = clean_text(text)
    opinions = []
    paragraphs = text.split("\n\n")

    SKIP_PHRASES = [
        "subscribe", "newsletter", "advertisement", "sponsored",
        "sign in", "sign up", "log in", "create account",
        "cookie", "privacy policy", "terms of service",
        "open in app", "download the app", "member-only",
        "read more", "view specs", "promoted",
        "press enter or click", "latest mobiles",
    ]

    for para in paragraphs:
        para = para.strip()

        # Must have enough real words (not just links/formatting)
        words = re.findall(r"[a-zA-Z]{3,}", para)
        if len(words) < 10 or len(para) > 2000:
            continue

        lower = para.lower()

        # Skip boilerplate
        if any(skip in lower for skip in SKIP_PHRASES):
            continue

        # Skip if mostly link/URL content
        url_chars = sum(len(m) for m in _URL_RE.findall(para))
        if url_chars > len(para) * 0.3:
            continue

        # Skip Twitter UI noise
        if platform == "twitter":
            if para.startswith("@") or para.startswith("Replying to"):
                continue

        opinion = Opinion(
            id=generate_id(para),
            text=para,
            source_url=url,
            source_platform=platform,
            author=author_hint or "unknown",
            context="",
            topic_tags=[],
            embedding=[],
            created_at="",
        )
        opinions.append(opinion)

    return opinions


async def search_opinions(
    query: str,
    persona_ids: list[str],
    top_k: int = 10,
) -> list[dict]:
    """
    Search for relevant opinions across persona indices using vector similarity.
    Returns ranked list of opinions with similarity scores.
    """
    from personas import load_persona_index

    query_embedding = await get_embedding(query)

    results = []

    for pid in persona_ids:
        index = load_persona_index(pid)
        if not index:
            continue

        for opinion in index.opinions:
            if opinion.embedding and query_embedding:
                score = cosine_similarity(query_embedding, opinion.embedding)
            else:
                score = 0.0

            if score > 0.3:
                results.append(
                    {
                        "persona_id": pid,
                        "persona_name": index.name,
                        "persona_type": index.type,
                        "opinion": opinion.text,
                        "source_url": opinion.source_url,
                        "source_platform": opinion.source_platform,
                        "similarity": round(score, 3),
                    }
                )

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


async def generate_response_from_opinions(
    topic: str,
    opinions: list[dict],
    use_agent: bool = True,
) -> str:
    """Generate a clear, structured verdict grounded in retrieved opinions."""

    if use_agent:
        try:
            return await synthesize_with_agent(topic, opinions)
        except Exception as e:
            print(f"  ↳ Agent synthesis failed: {e}, using fallback")

    # Clean opinions before synthesis
    cleaned_opinions = []
    for o in opinions[:12]:
        text = clean_text(o["opinion"])[:500]
        if len(text) > 50:
            cleaned_opinions.append(
                f"[{o['persona_name']}, {o['source_platform']}]: {text}"
            )

    if not cleaned_opinions:
        return "No meaningful opinions could be extracted for this topic."

    opinions_text = "\n\n".join(cleaned_opinions)

    prompt = f"""Analyze these real opinions and give a clear, structured verdict.

TOPIC: {topic}

OPINIONS FROM THE WEB:
{opinions_text}

Write your analysis in this EXACT format:

## TL;DR
[2-3 sentence verdict. What's the consensus? Is the idea good, bad, or mixed? Be direct.]

## What People Like
- [Specific positive point, with a short quote if available]
- [Another pro grounded in the opinions]

## Concerns Raised
- [Specific criticism or worry, with a short quote if available]
- [Another con]

## Key Tensions
[1-2 sentences on where opinions diverge the most]

## Bottom Line
[1-2 sentence actionable takeaway. What should someone do with this information?]

RULES:
- Every point MUST be grounded in the provided opinions — do not invent
- SILENTLY IGNORE opinions that are NOT relevant to the topic (press releases, product pages, job titles, team rosters)
- NEVER fabricate a theme from unrelated content — if opinions don't address the topic, say so
- Be concise and direct, not academic
- Use short inline quotes ("like this") to show real voices
- If opinions don't cover a section, skip it
- Write for someone who wants a quick, useful answer"""

    try:
        result = await get_high_performance_llm(
            prompt, max_tokens=3000, reasoning_effort="high"
        )
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        # Structured fallback — don't dump raw text
        print(f"  ↳ LLM synthesis failed: {e}, generating minimal summary")
        summary_lines = [
            f"## Summary (auto-generated, LLM unavailable)\n",
            f"**Topic:** {topic}\n",
            f"**{len(cleaned_opinions)} opinions retrieved.** Key points:\n",
        ]
        for i, co in enumerate(cleaned_opinions[:5], 1):
            text = co.split("]: ", 1)[-1] if "]: " in co else co
            summary_lines.append(f"{i}. {text[:200]}{'...' if len(text) > 200 else ''}\n")
        summary_lines.append(f"\n*Full synthesis unavailable due to: {e}*")
        return "\n".join(summary_lines)


def save_to_file(filename: str, content: str) -> str:
    """Save content to reports/ directory. Returns file path."""
    os.makedirs("reports", exist_ok=True)
    path = f"reports/{filename}"
    with open(path, "w") as f:
        f.write(content)
    return path
