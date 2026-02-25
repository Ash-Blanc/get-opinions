import json
from datetime import datetime
from typing import Literal
import asyncio

from personas import (
    PersonaIndex,
    load_persona_index,
    save_persona_index,
    list_persona_indices,
    normalize_id,
    OPINIONS_DIR,
)
from tools import (
    build_persona_index,
    search_opinions,
    generate_response_from_opinions,
    save_to_file,
    select_personas_for_topic,
)


class OpinionPipeline:
    """
    RAG-based opinion retrieval pipeline.

    Each persona has an indexed collection of real opinions with embeddings.
    For any topic, relevant opinions are retrieved and synthesized.
    """

    def __init__(self):
        pass

    def _format_report(self, topic: str, results: dict, opinions: list[dict]) -> str:
        """Format results into clean markdown report."""
        from tools import clean_text

        lines = [
            f"# OPINIONS: {topic}",
            f"_{datetime.now().strftime('%Y-%m-%d %H:%M')} · {results['personas_queried']} indices · {len(opinions)} opinions_",
            "",
        ]

        # Synthesis is the main event
        lines.append("---")
        lines.append("")
        synthesis = results.get("synthesis", "Unable to generate synthesis.")
        lines.append(synthesis)
        lines.append("")

        # Compact opinion evidence section
        lines.append("---")
        lines.append("")
        lines.append("## Key Opinions Retrieved")
        lines.append("")

        sources = []
        current_persona = None
        for o in opinions:
            if o["persona_name"] != current_persona:
                current_persona = o["persona_name"]
                lines.append(f"### {current_persona}")
                lines.append("")

            # Clean and truncate the opinion text
            text = clean_text(o["opinion"])
            text = text.replace("\n", " ").strip()
            if len(text) > 250:
                text = text[:247] + "..."

            platform = o.get("source_platform", "web")
            sim = round(o.get("similarity", 0), 2)
            lines.append(f"> {text}")
            lines.append(f"> — *{platform}, relevance: {sim}*")
            lines.append("")

            # Collect sources for footer
            url = o.get("source_url", "")
            if url and url not in [s[1] for s in sources]:
                sources.append((current_persona, url))

        # Sources footer
        if sources:
            lines.append("---")
            lines.append("")
            lines.append("## Sources")
            lines.append("")
            for persona, url in sources[:15]:
                lines.append(f"- [{persona}]({url})")
            lines.append("")

        return "\n".join(lines)

    async def run(
        self,
        topic: str,
        persona_identifiers: list[str] = None,
        top_k: int = 15,
        build_missing: bool = True,
        use_agent: bool = True,
        auto_select: bool = True,
    ) -> str:
        """
        Search for opinions on a topic across persona indices.

        Args:
            topic: The idea/thought/concept to get opinions on
            persona_identifiers: List of persona names/IDs to search (auto-selected if None)
            top_k: Number of opinions to retrieve
            build_missing: Build index for personas that don't exist
            use_agent: Use agentica agent for synthesis
            auto_select: Auto-select relevant personas if not specified

        Returns:
            Formatted markdown report
        """
        print(f"\n{'═' * 60}")
        print(f"  OPINIONS — RAG Retrieval")
        print(f"  Topic: {topic[:50]}{'...' if len(topic) > 50 else ''}")
        print(f"{'═' * 60}\n")

        if persona_identifiers is None:
            if auto_select:
                print("⟐ Selecting relevant personas...")
                persona_identifiers = await select_personas_for_topic(topic)
                print(f"  Selected: {', '.join(persona_identifiers)}")
            else:
                return "No personas specified and auto_select disabled."

        persona_ids = []
        persona_indices = []

        for identifier in persona_identifiers:
            pid = normalize_id(identifier)
            index = load_persona_index(pid)

            if index:
                persona_ids.append(pid)
                persona_indices.append(index)
                print(
                    f"  ✓ Loaded index: {index.name} ({index.total_opinions} opinions)"
                )
            elif build_missing:
                print(f"  ⟐ Building index for: {identifier}")
                index = await self._build_default_index(identifier)
                if index:
                    persona_ids.append(pid)
                    persona_indices.append(index)
            else:
                print(f"  ✗ No index found: {identifier}")

        if not persona_ids:
            return "No persona indices available. Run `opinions build` first."

        print(f"\n⟐ Searching for relevant opinions...")
        opinions = await search_opinions(
            topic, persona_ids, top_k=top_k * len(persona_ids)
        )

        print(f"  Found {len(opinions)} relevant opinions")

        if not opinions:
            return "No relevant opinions found in indices."

        top_opinions = opinions[:top_k]

        print(
            f"\n⟐ Generating synthesis (agent={'enabled' if use_agent else 'fallback'})..."
        )
        synthesis = await generate_response_from_opinions(
            topic, top_opinions, use_agent=use_agent
        )

        results = {
            "topic": topic,
            "personas_queried": len(persona_ids),
            "synthesis": synthesis,
        }

        report = self._format_report(topic, results, top_opinions)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"opinions_{ts}.md"
        path = save_to_file(filename, report)

        print(f"\n{'═' * 60}")
        print(f"  ✓ Report saved → {path}")
        print(f"{'═' * 60}\n")

        return report

    async def _build_default_index(
        self, identifier: str, use_agent: bool = False
    ) -> PersonaIndex | None:
        """Build a default index for a persona using discovered or default queries."""
        pid = normalize_id(identifier)

        # Check if dynamic discovery provided targeted queries for this persona
        search_queries = None
        discovery = select_personas_for_topic._last_discovery
        if discovery:
            for p in discovery:
                if p.get("name", "").lower() == identifier.lower():
                    search_queries = p.get("search_queries", [])
                    break

        # Fall back to generic queries if no discovery data
        if not search_queries:
            search_queries = [
                f'"{identifier}" advice tips',
                f'"{identifier}" opinion perspective',
                f'"{identifier}" discussion insights',
                f'{identifier} thoughts experience',
            ]

        try:
            index = await build_persona_index(
                persona_id=pid,
                persona_name=identifier,
                persona_type="individual",
                search_queries=search_queries,
                max_opinions=50,
                use_embeddings=True,
                use_agent=use_agent,
            )
            return index
        except Exception as e:
            print(f"    ✗ Failed to build index: {e}")
            return None

    async def build_index(
        self,
        name: str,
        persona_type: Literal["individual", "group"],
        search_queries: list[str],
        max_opinions: int = 200,
        use_agent: bool = True,
    ) -> PersonaIndex:
        """Build a new persona index."""
        pid = normalize_id(name)

        return await build_persona_index(
            persona_id=pid,
            persona_name=name,
            persona_type=persona_type,
            search_queries=search_queries,
            max_opinions=max_opinions,
            use_embeddings=True,
            use_agent=use_agent,
        )


def list_indices() -> None:
    """List all persona indices."""
    indices = list_persona_indices()

    if not indices:
        print("\nNo persona indices found.")
        print(f"Index directory: {OPINIONS_DIR}/")
        print("\nBuild an index with: opinions build <name> <search_queries...>")
        return

    print(f"\nPERSONA INDICES ({OPINIONS_DIR}/):")
    print("-" * 60)

    for idx in sorted(indices, key=lambda x: x.total_opinions, reverse=True):
        print(f"\n  {idx.id}")
        print(f"    Name: {idx.name}")
        print(f"    Type: {idx.type}")
        print(f"    Opinions: {idx.total_opinions}")
        print(
            f"    Last indexed: {idx.last_indexed[:10] if idx.last_indexed else 'never'}"
        )


def get_index_stats(persona_id: str) -> dict:
    """Get statistics for a persona index."""
    index = load_persona_index(persona_id)
    if not index:
        return {"error": "Index not found"}

    platforms = {}
    for o in index.opinions:
        platforms[o.source_platform] = platforms.get(o.source_platform, 0) + 1

    return {
        "id": index.id,
        "name": index.name,
        "type": index.type,
        "total_opinions": index.total_opinions,
        "platforms": platforms,
        "last_indexed": index.last_indexed,
        "search_queries": index.search_queries,
    }
