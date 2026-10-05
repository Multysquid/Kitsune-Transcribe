"""Build the background bank (kitsune/noise_bank.py) from MUSAN: its music without vocals and its noise, never its
speech, as one int16 array at 16 kHz mono plus an index (DECISIONS H8; the P-0.3B review's issue A).

MUSAN (Snyder, Chen, Povey 2015; https://www.openslr.org/17/, CC BY 4.0): music/{fma,fma-western-art,hd-classical,
jamendo,rfm}/*.wav with an ANNOTATIONS file per folder (one line per file: name, genre, vocals Y/N, ...) and
noise/{free-sound,sound-bible}/*.wav, all 16 kHz mono. Music with vocals is left out (the student would learn to leave
sung words unwritten, next to speech it must write); speech is left out (a second voice is mix_p's business).

Writes <out>/audio.npy and <out>/index.json (kitsune.noise_bank's layout): every clip's samples back to back, in
sorted path order; per clip its id (folder/name), kind (music | noise), offset, length, folder and vocals flag; the
source and licence; and audio_sha256. index.json's own sha256 is printed: augment.noise_bank_sha256 pins it.

Usage (CPU, a few minutes on the full corpus; --music-hours caps the music, a fixed hash-ordered choice of whole clips):
  python tools/build_noise_bank.py --musan D:/kitsune-tmp/musan/musan --out D:/kitsune-tmp/musan/bank/musan-bg-v1 \
      --music-hours 15
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kitsune import noise_bank  # noqa: E402

SOURCE = ("MUSAN: A Music, Speech, and Noise Corpus (David Snyder, Guoguo Chen, Daniel Povey, 2015), "
          "https://www.openslr.org/17/, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); "
          "music without vocals and noise only, converted to one int16 array")


def vocals_of(folder: Path) -> dict[str, bool]:
    """{file stem: has vocals} from a MUSAN music folder's ANNOTATIONS (name, genre, vocals Y/N, ...); {} without one."""
    f = folder / "ANNOTATIONS"
    out = {}
    if f.exists():
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[2] in ("Y", "N"):
                out[parts[0]] = parts[2] == "Y"
    return out


def clips_of(musan: Path) -> list[dict]:
    """The clips the bank takes, in sorted path order: music without vocals (a music file the ANNOTATIONS do not
    mark is left out: its vocals are unknown) and all noise."""
    out = []
    for kind in ("music", "noise"):
        for folder in sorted(p for p in (musan / kind).iterdir() if p.is_dir()):
            voc = vocals_of(folder) if kind == "music" else {}
            for wav in sorted(folder.glob("*.wav")):
                if kind == "music" and voc.get(wav.stem, True):
                    continue
                out.append(dict(path=wav, id=f"{kind}/{folder.name}/{wav.stem}", kind=kind, folder=folder.name,
                                vocals=False if kind == "music" else None))
    return out


def cap_music(clips: list[dict], lengths: list[int], hours: float | None) -> tuple[list[dict], list[int]]:
    """With a cap: the music clips in the order of the sha256 of their ids (a fixed shuffle, the same on every machine)
    until the next one would pass `hours`, then back in path order; noise always whole. None: everything."""
    if hours is None:
        return clips, lengths
    music = sorted((i for i, c in enumerate(clips) if c["kind"] == "music"),
                   key=lambda i: hashlib.sha256(clips[i]["id"].encode()).hexdigest())
    keep, total = set(), 0
    for i in music:
        if total + lengths[i] > hours * 3600 * noise_bank.SR:
            continue
        keep.add(i)
        total += lengths[i]
    idx = [i for i, c in enumerate(clips) if c["kind"] != "music" or i in keep]
    return [clips[i] for i in idx], [lengths[i] for i in idx]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--musan", required=True, help="the extracted musan/ dir (music/, noise/, speech/ below it)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--music-hours", type=float, default=None,
                    help="at most this much music (a fixed hash-ordered choice of whole clips); default all")
    a = ap.parse_args(argv)
    import soundfile as sf

    musan, out = Path(a.musan), Path(a.out)
    clips = clips_of(musan)
    if not clips:
        raise SystemExit(f"{musan}: no music without vocals and no noise found")
    lengths = []
    for c in clips:
        info = sf.info(str(c["path"]))
        if info.samplerate != noise_bank.SR:
            raise SystemExit(f"{c['path']}: {info.samplerate} Hz, the bank is {noise_bank.SR} Hz")
        lengths.append(int(info.frames))
    clips, lengths = cap_music(clips, lengths, a.music_hours)
    total = int(sum(lengths))
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / (noise_bank.AUDIO + ".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.int16, shape=(total,))
    off = 0
    index = []
    for c, n in zip(clips, lengths):
        x, _ = sf.read(str(c["path"]), dtype="int16", always_2d=True)
        x = x.mean(axis=1).astype(np.int16) if x.shape[1] > 1 else x[:, 0]
        if len(x) != n:
            raise SystemExit(f"{c['path']}: read {len(x)} samples, the header says {n}")
        arr[off:off + n] = x
        index.append(dict(id=c["id"], kind=c["kind"], folder=c["folder"], vocals=c["vocals"], offset=off, length=n))
        off += n
    arr.flush()
    del arr
    tmp.replace(out / noise_bank.AUDIO)
    audio_sha = noise_bank.sha256_file(out / noise_bank.AUDIO)
    hours = {k: round(sum(c["length"] for c in index if c["kind"] == k) / noise_bank.SR / 3600, 3)
             for k in ("music", "noise")}
    info = dict(version=noise_bank.VERSION, sr=noise_bank.SR, audio_bytes=(out / noise_bank.AUDIO).stat().st_size,
                audio_sha256=audio_sha, samples=total, hours=hours, music_hours_cap=a.music_hours, source=SOURCE,
                licence="CC BY 4.0",
                left_out="MUSAN speech/ and music with vocals (or without an ANNOTATIONS line)", clips=index)
    (out / noise_bank.INDEX).write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    sha = noise_bank.sha256_file(out / noise_bank.INDEX)
    print(f"wrote {out}: {len(index)} clips, music {hours['music']} h, noise {hours['noise']} h, "
          f"{info['audio_bytes'] / 2**30:.2f} GiB; index.json sha256 {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
