import asyncio
import json
import os
import shutil
import cli2
from pipeline import OpinionPipeline, list_indices, get_index_stats
from personas import OPINIONS_DIR


class DefaultGroup(cli2.Group):
    """Group that uses 'ask' as default command when topic is provided directly."""

    def __call__(self, *argv):
        self.exit_code = 0
        if not argv:
            return self.help(error="Provide a topic to get opinions")

        if argv[0] in ("help", "build", "list", "clear", "run", "ask"):
            result = self[argv[0]](*argv[1:])
            self.exit_code = self[argv[0]].exit_code
        else:
            result = self["ask"](*argv)
            self.exit_code = self["ask"].exit_code

        return result


cli = DefaultGroup(
    doc="OPINIONS — Get perspectives on any idea. Just input your thought."
)


@cli.cmd(name="ask")
def ask_command(
    topic: str,
    personas: str = "",
    top_k: int = 15,
    no_build: bool = False,
    no_agent: bool = False,
):
    """
    Get opinions on your idea/thought/concept.

    Personas are auto-selected based on relevance to your topic.
    Use --personas to specify manually.

    :param topic: Your idea, thought, or concept to get opinions on
    :param personas: (optional) Comma-separated personas, e.g., 'karpathy,hn'
    :param top_k: Number of opinions to retrieve (default: 15)
    :param no_build: Don't build missing indices automatically
    :param no_agent: Disable agentica agents
    """
    persona_identifiers = None
    if personas:
        persona_identifiers = [p.strip() for p in personas.split(",") if p.strip()]

    pipeline = OpinionPipeline()
    report = asyncio.run(
        pipeline.run(
            topic=topic,
            persona_identifiers=persona_identifiers,
            top_k=top_k,
            build_missing=not no_build,
            use_agent=not no_agent,
            auto_select=True,
        )
    )
    cli2.print(report)


@cli.cmd
def run(
    topic: str,
    personas: str,
    top_k: int = 15,
    no_build: bool = False,
    no_agent: bool = False,
):
    """
    Run with explicit personas (advanced).

    :param topic: The idea/thought/concept to get opinions on
    :param personas: Comma-separated persona names
    :param top_k: Number of opinions to retrieve (default: 15)
    :param no_build: Don't build missing indices automatically
    :param no_agent: Disable agentica agents
    """
    persona_identifiers = [p.strip() for p in personas.split(",") if p.strip()]

    if not persona_identifiers:
        cli2.print("No personas specified.", color="red")
        return 1

    pipeline = OpinionPipeline()
    report = asyncio.run(
        pipeline.run(
            topic=topic,
            persona_identifiers=persona_identifiers,
            top_k=top_k,
            build_missing=not no_build,
            use_agent=not no_agent,
            auto_select=False,
        )
    )
    cli2.print(report)


@cli.cmd
def build(
    name: str,
    type: str = "individual",
    queries: str = "",
    max_opinions: int = 200,
    no_agent: bool = False,
):
    """
    Build a persona index by scraping real opinions.

    :param name: Persona name (e.g., 'Elon Musk', 'Tech Twitter')
    :param type: Persona type: 'individual' or 'group' (default: individual)
    :param queries: Comma-separated search queries (if omitted, uses defaults)
    :param max_opinions: Maximum opinions to index (default: 200)
    :param no_agent: Disable agentica agents for extraction
    """
    if type not in ("individual", "group"):
        cli2.print("type must be 'individual' or 'group'", color="red")
        return 1

    search_queries = []
    if queries:
        search_queries = [q.strip() for q in queries.split(",")]
    else:
        search_queries = [
            f'"{name}" opinion',
            f'"{name}" thoughts',
            f'"{name}" discussion',
            f'"{name}" said',
        ]

    pipeline = OpinionPipeline()
    asyncio.run(
        pipeline.build_index(
            name=name,
            persona_type=type,
            search_queries=search_queries,
            max_opinions=max_opinions,
            use_agent=not no_agent,
        )
    )


@cli.cmd(name="list")
def list_indices_cmd(stats: str = ""):
    """
    List all persona indices.

    :param stats: Show stats for a specific persona ID
    """
    if stats:
        result = get_index_stats(stats)
        if "error" in result:
            cli2.print(result["error"], color="red")
            return 1
        cli2.print(json.dumps(result, indent=2, default=str))
    else:
        list_indices()


@cli.cmd
def clear(confirm: bool = False):
    """
    Clear all persona indices.

    :param confirm: Confirm deletion
    """
    if not confirm:
        cli2.print("Use --confirm to delete all persona indices", color="yellow")
        return 1

    if os.path.exists(OPINIONS_DIR):
        shutil.rmtree(OPINIONS_DIR)
        cli2.print(f"Cleared all indices from {OPINIONS_DIR}/", color="green")
    else:
        cli2.print("No indices to clear", color="yellow")


main = cli.entry_point
