"""Retriever configurations: the Phase 3 retriever and switchable retrieval improvements.

A RetrieverConfig sets every retrieval knob. PHASE3_CONFIG ("phase3") is the Phase 3 retriever
exactly, with every improvement off:

1. bm25_stemming: Porter stems for corpus and query words in BM25.
2. query_expansion: FOMC synonym-map terms added to BM25 queries (BM25 only), at
   expansion_weight times a query term's weight.
3. date_prior (and year_prior): a score factor for chunks of a meeting in the month or on the
   day (a year, for year_prior) the question names.
4. recency_prior: when as_of is set and the question asks about the current state ("still",
   "currently", "has the Fed started"), newer chunks get 1 + recency_prior * 0.5 ** (age /
   recency_half_life_days), age being the days from release to as_of.
5. statement_boost: a score factor for statement chunks.
6. rrf_weights and rrf_k: how BM25 and dense rankings are fused.
7. reranker_model (and reranker_revision): another cross-encoder than settings.reranker_model.

The priors (3-5) multiply the final score of each method (the fused score for hybrid, the
cross-encoder's for hybrid_rerank, whose hybrid candidates are ranked with the priors too) and
only reorder chunks that already passed the as_of and doc_types filters.

Named presets: "phase3", defined here and never edited, and those in retriever_presets.json
("tuned", ...), so tuning a preset edits only that file.

The production retriever (POST /api/ask and /api/ask/stream, make ask, make smoke, and
get_retriever without a config) is production_config(settings): the preset settings.retriever_preset
names (RETRIEVER_PRESET). That setting is the one place the default is chosen: "tuned" since
Phase 7, when the TEST switch rule of docs/rag-tuning.md held; "phase3" stays selectable.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.config import Settings
from app.rag.bm25 import DEFAULT_EXPANSION_WEIGHT
from app.rag.filters import MetadataPrior

RRF_K = 60
MIN_CANDIDATES = 50
PHASE3_PRESET = "phase3"  # the Phase 3 retriever: the baseline every change is measured from
PRESETS_FILE = Path(__file__).with_name("retriever_presets.json")
_LABEL_FIELDS = frozenset({"name", "description"})  # everything else changes the rankings


class RetrieverConfig(BaseModel):
    """One retriever configuration: a name and a value for every retrieval knob."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, description="Preset name, e.g. phase3 or tuned.")
    description: str = Field(default="", description="What this configuration is for.")
    bm25_stemming: bool = Field(
        default=False, description="Porter-stem corpus and query words in BM25."
    )
    query_expansion: bool = Field(
        default=False, description="Add FOMC synonym-map terms to BM25 queries (BM25 only)."
    )
    expansion_weight: float = Field(
        default=DEFAULT_EXPANSION_WEIGHT,
        gt=0,
        le=1,
        description="BM25 weight of an expansion term relative to a query term.",
    )
    date_prior: float = Field(
        default=1.0,
        gt=0,
        description="Score factor for chunks of a meeting in a month or on a day the question "
        "names; 1 is off.",
    )
    year_prior: float = Field(
        default=1.0,
        gt=0,
        description="Score factor for chunks of a meeting in a year the question names without "
        "a month; 1 is off.",
    )
    recency_prior: float = Field(
        default=0.0,
        ge=0,
        description="Extra score factor for a chunk released on as_of when the question asks "
        "about the current state, halving every recency_half_life_days; 0 is off.",
    )
    recency_half_life_days: float = Field(
        default=180.0, gt=0, description="Days over which the recency factor halves."
    )
    statement_boost: float = Field(
        default=1.0, gt=0, description="Score factor for statement chunks; 1 is off."
    )
    rrf_k: int = Field(default=RRF_K, ge=1, description="RRF constant k in 1 / (k + rank).")
    rrf_weights: tuple[float, float] = Field(
        default=(1.0, 1.0), description="RRF weights of the BM25 and the dense ranking."
    )
    min_candidates: int = Field(
        default=MIN_CANDIDATES,
        ge=1,
        description="Candidates per ranking before fusing, reranking or applying a prior: "
        "max(min_candidates, 4 * top_k).",
    )
    reranker_model: str | None = Field(
        default=None,
        min_length=1,
        description="Cross-encoder for hybrid_rerank; null uses settings.reranker_model.",
    )
    reranker_revision: str | None = Field(
        default=None,
        min_length=1,
        description="Hugging Face commit of reranker_model (null: its default branch); only "
        "with reranker_model.",
    )

    @field_validator("rrf_weights")
    @classmethod
    def _weights_rank_something(cls, value: tuple[float, float]) -> tuple[float, float]:
        if min(value) < 0 or max(value) <= 0:
            raise ValueError("rrf_weights must be non-negative and not both 0")
        return value

    @model_validator(mode="after")
    def _revision_needs_a_model(self) -> "RetrieverConfig":
        if self.reranker_revision is not None and self.reranker_model is None:
            raise ValueError("reranker_revision needs reranker_model")
        return self

    @property
    def behaviour(self) -> dict[str, Any]:
        """The fields that change rankings: all but name and description."""
        return self.model_dump(mode="json", exclude=set(_LABEL_FIELDS))

    @property
    def fingerprint(self) -> str:
        """A short hash of `behaviour`, for cache keys. It covers this config only: a reranker
        left null follows the settings, so key on resolved(settings) to include it."""
        canonical = json.dumps(self.behaviour, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def same_behaviour(self, other: "RetrieverConfig") -> bool:
        return self.behaviour == other.behaviour

    def resolved(self, settings: Settings) -> "RetrieverConfig":
        """This config with the reranker it uses under `settings` written in."""
        if self.reranker_model is not None:
            return self
        return self.model_copy(
            update={
                "reranker_model": settings.reranker_model,
                "reranker_revision": settings.reranker_revision,
            }
        )

    def reranker(self, settings: Settings) -> tuple[str, str | None]:
        """The cross-encoder (model, revision) hybrid_rerank uses under `settings`."""
        if self.reranker_model is None:
            return settings.reranker_model, settings.reranker_revision
        return self.reranker_model, self.reranker_revision

    def prior(self) -> MetadataPrior | None:
        """The metadata prior of fields 3-5, or None when all are off."""
        prior = MetadataPrior(
            date_boost=self.date_prior,
            year_boost=self.year_prior,
            recency_boost=self.recency_prior,
            recency_half_life_days=self.recency_half_life_days,
            statement_boost=self.statement_boost,
        )
        return None if prior.neutral else prior


PHASE3_CONFIG = RetrieverConfig(
    name=PHASE3_PRESET,
    description="Phase 3 retrieval: BM25 without stemming or expansion, RRF k=60 with equal "
    "weights, the settings' cross-encoder, no priors.",
)


def load_presets(path: Path = PRESETS_FILE) -> dict[str, RetrieverConfig]:
    """phase3 and the presets in `path`, a JSON object of name -> fields that differ from the
    defaults (the fields of RetrieverConfig other than name)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name}: expected a JSON object of preset name -> fields")
    presets = {PHASE3_PRESET: PHASE3_CONFIG}
    for name, fields in raw.items():
        if name == PHASE3_PRESET:
            raise ValueError(f"{path.name}: {PHASE3_PRESET} is fixed in code and cannot be set")
        if not isinstance(fields, dict) or "name" in fields:
            raise ValueError(f"{path.name}: preset {name} must be an object without a name field")
        presets[name] = RetrieverConfig.model_validate({"name": name, **fields})
    return presets


def get_preset(name: str, path: Path = PRESETS_FILE) -> RetrieverConfig:
    """The named preset; ValueError lists the known names."""
    presets = load_presets(path)
    if name not in presets:
        raise ValueError(f"unknown retriever preset {name!r}: one of {', '.join(presets)}")
    return presets[name]


def production_config(settings: Settings, path: Path = PRESETS_FILE) -> RetrieverConfig:
    """The retriever config Ask the Fed uses: the preset named by settings.retriever_preset
    (RETRIEVER_PRESET, "tuned" unless set). ValueError names the known presets."""
    try:
        return get_preset(settings.retriever_preset, path)
    except ValueError as exc:
        raise ValueError(f"RETRIEVER_PRESET: {exc}") from None
