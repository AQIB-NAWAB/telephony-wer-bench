"""Source datasets: the bundled sample, LibriSpeech test-clean, or your own folder."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from .conditions import derive_seed

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
LIBRISPEECH_LICENSE = "CC BY 4.0"
LIBRISPEECH_CITATION = (
    "V. Panayotov, G. Chen, D. Povey, S. Khudanpur. LibriSpeech: an ASR corpus based on public "
    "domain audio books. ICASSP 2015. Data: https://www.openslr.org/12 (CC BY 4.0)."
)


@dataclass
class SourceItem:
    id: str
    audio: Path
    text: str
    speaker: str = ""


def bundled_sample_dir() -> Path:
    return Path(__file__).parent / "data" / "sample"


def load_source(path: str | Path | None) -> list[SourceItem]:
    """Load a source set. ``path`` is a directory holding ``source.jsonl`` or ``transcripts.tsv``.

    ``None`` loads the bundled sample. In ``source.jsonl`` each line has ``id``,
    ``audio`` (relative to the file) and ``text``. In ``transcripts.tsv`` each
    line is ``<audio filename>\\t<text>``.
    """
    root = Path(path) if path else bundled_sample_dir()
    if root.is_file():
        root = root.parent
    items: list[SourceItem] = []
    jsonl = root / "source.jsonl"
    tsv = root / "transcripts.tsv"
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            items.append(SourceItem(d["id"], root / d["audio"], d["text"], d.get("speaker", "")))
    elif tsv.exists():
        for line in tsv.read_text().splitlines():
            if not line.strip():
                continue
            fn, text = line.split("\t", 1)
            items.append(SourceItem(Path(fn).stem, root / fn, text.strip()))
    else:
        raise FileNotFoundError(f"{root} has neither source.jsonl nor transcripts.tsv")
    if not items:
        raise ValueError(f"no utterances found in {root}")
    return items


def download(url: str, dest: Path, quiet: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + f".{os.getpid()}.part")
    if not quiet:
        print(f"downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:  # noqa: S310
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dest)
    return dest


def fetch_librispeech(
    out: str | Path,
    n: int = 40,
    seed: int = 7,
    min_s: float = 3.0,
    max_s: float = 12.0,
    tarball: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> list[SourceItem]:
    """Download LibriSpeech test-clean (about 350 MB) and keep ``n`` utterances.

    Selection is deterministic: utterances between ``min_s`` and ``max_s``
    seconds are shuffled with ``seed`` and taken one per speaker first, so the
    sample covers as many speakers as possible. Audio is copied as FLAC.
    The corpus is CC BY 4.0, see LIBRISPEECH_CITATION.
    """
    out = Path(out)
    cache = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "twb"
    tar_path = Path(tarball) if tarball else download(LIBRISPEECH_URL, cache / "test-clean.tar.gz")
    texts: dict[str, str] = {}
    files: dict[str, bytes] = {}
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            if m.name.endswith(".trans.txt"):
                for line in tf.extractfile(m).read().decode().splitlines():  # type: ignore[union-attr]
                    uid, _, text = line.partition(" ")
                    texts[uid] = text
            elif m.name.endswith(".flac"):
                files[Path(m.name).stem] = tf.extractfile(m).read()  # type: ignore[union-attr]
    import io

    cands = []
    for uid, blob in files.items():
        info = sf.info(io.BytesIO(blob))
        dur = info.frames / info.samplerate
        if min_s <= dur <= max_s and uid in texts:
            cands.append((uid, dur))
    cands.sort(key=lambda c: derive_seed(seed, c[0]))
    chosen: list[tuple[str, float]] = []
    seen_speakers: set[str] = set()
    for uid, dur in cands:
        spk = uid.split("-")[0]
        if spk not in seen_speakers:
            seen_speakers.add(spk)
            chosen.append((uid, dur))
        if len(chosen) == n:
            break
    for uid, dur in cands:
        if len(chosen) >= n:
            break
        if (uid, dur) not in chosen:
            chosen.append((uid, dur))
    chosen.sort(key=lambda c: c[0])
    out.mkdir(parents=True, exist_ok=True)
    items = []
    lines = []
    for uid, _ in chosen:
        fn = f"{uid}.flac"
        (out / fn).write_bytes(files[uid])
        lines.append(json.dumps({"id": uid, "audio": fn, "text": texts[uid], "speaker": uid.split("-")[0]}))
        items.append(SourceItem(uid, out / fn, texts[uid], uid.split("-")[0]))
    (out / "source.jsonl").write_text("\n".join(lines) + "\n")
    (out / "LICENSE.txt").write_text(
        f"Audio and transcripts are from LibriSpeech test-clean, licensed {LIBRISPEECH_LICENSE}.\n"
        f"{LIBRISPEECH_CITATION}\n"
    )
    return items
