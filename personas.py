import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

OPINIONS_DIR = os.environ.get("OPINIONS_DIR", "opinions")
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "text-embedding-3-small")


@dataclass
class Opinion:
    id: str
    text: str
    source_url: str
    source_platform: str
    author: str
    context: str
    topic_tags: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "source_url": self.source_url,
            "source_platform": self.source_platform,
            "author": self.author,
            "context": self.context,
            "topic_tags": self.topic_tags,
            "embedding": self.embedding[:10] if self.embedding else [],
            "created_at": self.created_at,
        }


@dataclass
class PersonaIndex:
    id: str
    name: str
    type: str  # "individual" or "group"
    description: str
    search_queries: list[str]
    opinions: list[Opinion] = field(default_factory=list)
    last_indexed: str = ""
    total_opinions: int = 0

    def index_path(self) -> str:
        return os.path.join(OPINIONS_DIR, f"{self.id}.json")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "search_queries": self.search_queries,
            "last_indexed": self.last_indexed,
            "total_opinions": len(self.opinions),
            "opinions": [o.to_dict() for o in self.opinions[:5]],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PersonaIndex":
        opinions = [
            Opinion(**o) if isinstance(o, dict) else o for o in d.get("opinions", [])
        ]
        return cls(
            id=d["id"],
            name=d["name"],
            type=d["type"],
            description=d.get("description", ""),
            search_queries=d.get("search_queries", []),
            opinions=opinions,
            last_indexed=d.get("last_indexed", ""),
            total_opinions=len(opinions),
        )


def get_index_path(persona_id: str) -> str:
    os.makedirs(OPINIONS_DIR, exist_ok=True)
    return os.path.join(OPINIONS_DIR, f"{persona_id}.json")


def load_persona_index(persona_id: str) -> Optional[PersonaIndex]:
    path = get_index_path(persona_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        return PersonaIndex.from_dict(data)
    return None


def save_persona_index(index: PersonaIndex) -> None:
    index.last_indexed = datetime.now().isoformat()
    index.total_opinions = len(index.opinions)
    path = get_index_path(index.id)

    full_data = {
        "id": index.id,
        "name": index.name,
        "type": index.type,
        "description": index.description,
        "search_queries": index.search_queries,
        "last_indexed": index.last_indexed,
        "total_opinions": len(index.opinions),
        "opinions": [
            {
                "id": o.id,
                "text": o.text,
                "source_url": o.source_url,
                "source_platform": o.source_platform,
                "author": o.author,
                "context": o.context,
                "topic_tags": o.topic_tags,
                "embedding": o.embedding,
                "created_at": o.created_at,
            }
            for o in index.opinions
        ],
    }
    with open(path, "w") as f:
        json.dump(full_data, f, indent=2)


def list_persona_indices() -> list[PersonaIndex]:
    indices = []
    if os.path.exists(OPINIONS_DIR):
        for filename in os.listdir(OPINIONS_DIR):
            if filename.endswith(".json"):
                try:
                    idx = load_persona_index(filename[:-5])
                    if idx:
                        indices.append(idx)
                except:
                    pass
    return indices


def normalize_id(name: str) -> str:
    return name.lower().replace(" ", "-").replace("_", "-").replace("/", "-").strip()
