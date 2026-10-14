"""Live check of the CrossEncoder reranker (downloads the model from Hugging Face on first run):
run with RUN_LIVE_TESTS=1."""

import pytest

from app.core.config import Settings
from app.rag.retrievers import CrossEncoderScorer

pytestmark = pytest.mark.live


def test_live_cross_encoder_ranks_the_relevant_passage_first() -> None:
    scorer = CrossEncoderScorer(Settings(_env_file=None).reranker_model)
    passages = [
        "Participants discussed developments in the housing market and residential investment.",
        "The Committee decided to lower the target range for the federal funds rate by 1/2 "
        "percentage point to 4-3/4 to 5 percent.",
    ]

    scores = scorer("How much did the Fed cut rates in September 2024?", passages)

    print(scores)
    assert len(scores) == 2 and all(0.0 < s < 1.0 for s in scores)
    assert scores[1] > scores[0]
