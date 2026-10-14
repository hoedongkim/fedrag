"""The Trillion Dollar Words hawkish-dovish sentence dataset (Shah, Paturi and Chava, ACL 2023).

Human-labelled sentences from FOMC minutes, press conferences and speeches, 1996-2022, with the
authors' three train/test splits. Released under CC BY-NC 4.0: it is downloaded on demand into
data/external/tdw (git-ignored) and never committed. Files are pinned to one commit and checked
against their SHA-256.
"""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx2
import openpyxl

from app.core.config import Settings, get_settings
from app.schemas.common import Stance

REPO = "gtfintechlab/fomc-hawkish-dovish"
COMMIT = "98646987452b326507479cf641571b33814bb73f"
SEEDS = (5768, 78516, 944601)
LABELS: dict[int, Stance] = {0: "dovish", 1: "hawkish", 2: "neutral"}  # the authors' mapping
CITATION = (
    "Agam Shah, Suvan Paturi and Sudheer Chava. 2023. Trillion Dollar Words: A New Financial "
    "Dataset, Task & Market Analysis. ACL 2023. https://aclanthology.org/2023.acl-long.368/"
)

Split = Literal["train", "test"]

SHA256: dict[tuple[str, int], str] = {
    ("train", 5768): "00e688cbd4947afd67d7829c52e0911008582d01148d7f4752e9a2a1c8a4d1c1",
    ("test", 5768): "d2de30a82541dc08c852d225805f6d2eb29faccf5ca0167bf5031b3249fa0f9c",
    ("train", 78516): "463596c164a8638184bc438226a065947d2a47e37d11954ff8b787112783c7dc",
    ("test", 78516): "b3034c34fa45b843429e71e0b0b80faf2afddda4e5d85f156783c63493cdeb3e",
    ("train", 944601): "29fed263eaf88c5a2cc2733d8a91677613cdc12bc6d1bcff7ef9de074ce9153e",
    ("test", 944601): "218fa6f3e2110fc38619c5e074b6aa629d458bb34069c01a608e491165ffd69c",
}


@dataclass(frozen=True)
class LabeledSentence:
    """One annotated sentence. index is the authors' row id within its source document set, so it
    is NOT unique across minutes, speeches and press conferences: identify sentences by text. The
    published splits share 12-15 duplicate sentence texts between train and test per seed."""

    index: int
    sentence: str
    year: int
    stance: Stance


def file_name(split: Split, seed: int) -> str:
    return f"lab-manual-combine-{split}-{seed}.xlsx"


def data_dir(settings: Settings | None = None) -> Path:
    return (settings or get_settings()).data_dir / "external" / "tdw"


def _url(name: str) -> str:
    folder = "training_data" if "-train-" in name else "test_data"
    return (
        f"https://raw.githubusercontent.com/{REPO}/{COMMIT}"
        f"/training_data/test-and-training/{folder}/{name}"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(settings: Settings | None = None) -> Path:
    """Fetch every split that is missing or corrupt; return the directory holding them."""
    target = data_dir(settings)
    target.mkdir(parents=True, exist_ok=True)
    with httpx2.Client(timeout=60, follow_redirects=True) as client:
        for (split, seed), digest in SHA256.items():
            name = file_name(split, seed)
            path = target / name
            if path.is_file() and _sha256(path) == digest:
                continue
            response = client.get(_url(name))
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest() != digest:
                raise ValueError(f"{name}: checksum mismatch; refusing to use it")
            path.write_bytes(response.content)
    return target


def _rows(path: Path) -> Iterator[LabeledSentence]:
    sheet = openpyxl.load_workbook(path, read_only=True).active
    header, *rows = sheet.iter_rows(values_only=True)
    if tuple(header) != ("index", "sentence", "year", "label"):
        raise ValueError(f"{path.name}: unexpected header {header!r}")
    for index, sentence, year, label in rows:
        yield LabeledSentence(int(index), str(sentence).strip(), int(year), LABELS[int(label)])


def deduplicated_test(seed: int, settings: Settings | None = None) -> list[LabeledSentence]:
    """The seed's test split minus sentences whose text also appears in its train split."""
    seen = {s.sentence for s in load("train", seed, settings)}
    return [s for s in load("test", seed, settings) if s.sentence not in seen]


def load(split: Split, seed: int, settings: Settings | None = None) -> list[LabeledSentence]:
    """One split of one seed, downloading the files on first use."""
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    path = data_dir(settings) / file_name(split, seed)
    if not path.is_file() or _sha256(path) != SHA256[(split, seed)]:
        download(settings)
    return list(_rows(path))
