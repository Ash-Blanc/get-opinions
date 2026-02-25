import os
import json
import time
import hashlib
import urllib.request
import urllib.error
from typing import Optional
from dotenv import load_dotenv
from exa_py import Exa
from firecrawl import FirecrawlApp

load_dotenv()

exa = Exa(api_key=os.environ["EXA_API_KEY"])
firecrawl = FirecrawlApp(api_key=os.environ["FIRECRAWL_API_KEY"])
PARALLEL_KEY = os.environ.get("PARALLEL_API_KEY", "")
MISTRAL_KEY = os.environ.get("MISTRAL_API_KEY", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "mistral-embed")


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


async def get_embedding_mistral(text: str) -> list[float]:
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


async def get_embedding_openrouter(text: str) -> list[float]:
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
    with urllib.request.urlopen(req, timeout=30) as response:
        res = json.loads(response.read().decode("utf-8"))
        return res["data"][0]["embedding"]


async def get_embedding(text: str) -> list[float]:
    """Get embedding with Mistral primary, OpenRouter fallback."""
    try:
        return await get_embedding_mistral(text)
    except Exception as e:
        print(f"  ↳ Mistral embedding failed: {e}")
        try:
            print("  ↳ Falling back to OpenRouter...")
            return await get_embedding_openrouter(text)
        except Exception as e2:
            print(f"  ↳ OpenRouter fallback failed: {e2}")
            return []


async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings for multiple texts with fallback."""
    if not texts:
        return []

    embeddings = []
    for text in texts:
        emb = await get_embedding(text)
        embeddings.append(emb)
    return embeddings


async def call_llm(prompt: str, system: str = "") -> str:
    """Call LLM via OpenRouter for synthesis."""
    if not OPENROUTER_KEY and not PARALLEL_KEY:
        raise ValueError("No LLM API key configured")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if OPENROUTER_KEY:
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
                    "model": "anthropic/claude-3-haiku",
                    "messages": messages,
                    "stream": False,
                }
            ).encode("utf-8"),
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res["choices"][0]["message"]["content"]

    if PARALLEL_KEY:
        req = urllib.request.Request(
            "https://api.parallel.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PARALLEL_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps(
                {
                    "model": "base",
                    "messages": messages,
                    "stream": False,
                }
            ).encode("utf-8"),
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res["choices"][0]["message"]["content"]

    raise ValueError("No LLM API available")


async def select_personas_for_topic(topic: str) -> list[str]:
    """Use agent to select the most relevant personas for a topic."""
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
        return ["karpathy", "hn"]


async def synthesize_with_agent(topic: str, opinions: list[dict]) -> str:
    """Use agentica agent to synthesize retrieved opinions."""
    from agents import create_synthesis_agent

    opinions_text = "\n\n".join(
        [
            f"[{o['persona_name']}] (relevance: {o['similarity']})\n{o['opinion']}"
            for o in opinions[:15]
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
    Uses Exa to find relevant content and Firecrawl to extract opinions.
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
    """Extract individual opinions from scraped text."""
    from personas import Opinion

    opinions = []

    paragraphs = text.split("\n\n")

    for para in paragraphs:
        para = para.strip()
        if len(para) < 50 or len(para) > 2000:
            continue

        if platform == "twitter":
            if para.startswith("@") or para.startswith("Replying to"):
                continue

        if any(
            skip in para.lower()
            for skip in ["subscribe", "newsletter", "advertisement", "sponsored"]
        ):
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
    """Generate pros/cons/feedback response grounded in retrieved opinions."""

    if use_agent:
        try:
            return await synthesize_with_agent(topic, opinions)
        except Exception as e:
            print(f"  ↳ Agent synthesis failed: {e}, using fallback")

    opinions_text = "\n\n".join(
        [
            f"[{o['persona_name']}] (relevance: {o['similarity']})\n{o['opinion']}"
            for o in opinions[:15]
        ]
    )

    prompt = f"""Based on these real opinions found in discussions, provide a structured analysis.

TOPIC: {topic}

RETRIEVED OPINIONS:
{opinions_text}

Generate a response with these sections:
1. PROS - positive points people actually mentioned
2. CONS - concerns/criticisms people actually mentioned  
3. CONSTRUCTIVE FEEDBACK - actionable advice based on the discourse

Use ONLY points grounded in the retrieved opinions. Quote directly when impactful.
Format as markdown."""

    try:
        return await call_llm(
            prompt,
            system="You synthesize real opinions into structured analysis. Ground every point in the provided quotes.",
        )
    except Exception as e:
        return f"Error generating response: {e}\n\nRaw opinions:\n{opinions_text}"


def save_to_file(filename: str, content: str) -> str:
    """Save content to reports/ directory. Returns file path."""
    os.makedirs("reports", exist_ok=True)
    path = f"reports/{filename}"
    with open(path, "w") as f:
        f.write(content)
    return path
