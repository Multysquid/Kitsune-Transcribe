"""Build the background bank (kitsune/noise_bank.py) from MUSAN: its music without vocals and its noise, and - asked
for - its music with vocals ("song") and its speech, as one int16 array at 16 kHz mono plus an index (DECISIONS H8;
H12 adds the songs and the speech).

MUSAN (Snyder, Chen, Povey 2015; https://www.openslr.org/17/, CC BY 4.0): music/{fma,fma-western-art,hd-classical,
jamendo,rfm}/*.wav with an ANNOTATIONS file per folder (one line per file: name, genre, vocals Y/N, ...),
noise/{free-sound,sound-bible}/*.wav and speech/{librivox,us-gov}/*.wav (librivox's ANNOTATIONS: name, gender,
language; 17 languages), all 16 kHz mono. The train loader mixes music, noise and songs under a row (augment.noise_p)
and the speech behind it (augment.speech_p), the targets always the teacher's on the clean row: so the student learns
to leave a song's lyrics and a background voice unwritten. A music file the ANNOTATIONS do not mark is left out (its
vocals are unknown).

Writes <out>/audio.npy and <out>/index.json (kitsune.noise_bank's layout): every clip's samples back to back, in
sorted path order; per clip its id (folder/name), kind (music | noise | song | speech), offset, length, folder, vocals
flag and language; the source and licence; and audio_sha256. index.json's own sha256 is printed:
augment.noise_bank_sha256 pins it.

Usage (CPU, a few minutes on the full corpus; each --*-hours caps that kind, a fixed hash-ordered choice of whole
clips; songs and speech only when asked for):
  python tools/build_noise_bank.py --musan D:/kitsune-tmp/musan/musan --out D:/kitsune-tmp/musan/bank/musan-bg-v2 \\
      --music-hours 15 --song-hours 10 --speech-hours 10
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
          "converted to one int16 array")


def annotations(folder: Path) -> dict[str, list[str]]:
    """{file stem: its ANNOTATIONS fields after the name} of a MUSAN folder; {} without the file."""
    f = folder / "ANNOTATIONS"
    out = {}
    if f.exists():
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                out[parts[0]] = parts[1:]
    return out


def vocals_of(folder: Path) -> dict[str, bool]:
    """{file stem: has vocals} from a MUSAN music folder's ANNOTATIONS (name, genre, vocals Y/N, ...); {} without
    one."""
    return {k: v[1] == "Y" for k, v in annotations(folder).items() if len(v) >= 2 and v[1] in ("Y", "N")}


def clips_of(musan: Path, kinds=("music", "noise")) -> list[dict]:
    """The clips the bank takes, in sorted path order: music without vocals ("music"), with vocals ("song"; a music
    file the ANNOTATIONS do not mark is neither), all noise and all speech - of the kinds asked for."""
    out = []
    for top in ("music", "noise", "speech"):
        if not (musan / top).is_dir():
            continue
        for folder in sorted(p for p in (musan / top).iterdir() if p.is_dir()):
            voc = vocals_of(folder) if top == "music" else {}
            ann = annotations(folder) if top == "speech" else {}
            for wav in sorted(folder.glob("*.wav")):
                if top == "music":
                    if wav.stem not in voc:
                        continue
                    kind = "song" if voc[wav.stem] else "music"
                else:
                    kind = top
                if kind not in kinds:
                    continue
                lang = (ann.get(wav.stem) or [None, None])[1] if top == "speech" else None
                if top == "speech" and lang is None and folder.name == "us-gov":
                    lang = "english"
                out.append(dict(path=wav, id=f"{top}/{folder.name}/{wav.stem}", kind=kind, folder=folder.name,
                                vocals=voc.get(wav.stem) if top == "music" else None, language=lang))
    return out


def cap_kind(clips: list[dict], lengths: list[int], kind: str, hours: float | None) -> tuple[list[dict], list[int]]:
    """With a cap: the clips of `kind` in the order of the sha256 of their ids (a fixed shuffle, the same on every
    machine), each taken unless it would pass `hours`, then back in path order; the other kinds whole. None: all."""
    if hours is None:
        return clips, lengths
    mine = sorted((i for i, c in enumerate(clips) if c["kind"] == kind),
                  key=lambda i: hashlib.sha256(clips[i]["id"].encode()).hexdigest())
    keep, total = set(), 0
    for i in mine:
        if total + lengths[i] > hours * 3600 * noise_bank.SR:
            continue
        keep.add(i)
        total += lengths[i]
    idx = [i for i, c in enumerate(clips) if c["kind"] != kind or i in keep]
    return [clips[i] for i in idx], [lengths[i] for i in idx]


def cap_music(clips: list[dict], lengths: list[int], hours: float | None) -> tuple[list[dict], list[int]]:
    return cap_kind(clips, lengths, "music", hours)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--musan", required=True, help="the extracted musan/ dir (music/, noise/, speech/ below it)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--music-hours", type=float, default=None,
                    help="at most this much music without vocals (a fixed hash-ordered choice of whole clips); default "
                         "all")
    ap.add_argument("--song-hours", type=float, default=0.0,
                    help="songs (music with vocals) up to this many hours; default 0: none")
    ap.add_argument("--speech-hours", type=float, default=0.0,
                    help="speech up to this many hours; default 0: none")
    a = ap.parse_args(argv)
    import soundfile as sf

    musan, out = Path(a.musan), Path(a.out)
    kinds = ["music", "noise"] + (["song"] if a.song_hours > 0 else []) + (["speech"] if a.speech_hours > 0 else [])
    clips = clips_of(musan, kinds)
    if not clips:
        raise SystemExit(f"{musan}: no clips of {kinds} found")
    lengths = []
    for c in clips:
        info = sf.info(str(c["path"]))
        if info.samplerate != noise_bank.SR:
            raise SystemExit(f"{c['path']}: {info.samplerate} Hz, the bank is {noise_bank.SR} Hz")
        lengths.append(int(info.frames))
    clips, lengths = cap_kind(clips, lengths, "music", a.music_hours)
    if a.song_hours > 0:
        clips, lengths = cap_kind(clips, lengths, "song", a.song_hours)
    if a.speech_hours > 0:
        clips, lengths = cap_kind(clips, lengths, "speech", a.speech_hours)
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
        entry = dict(id=c["id"], kind=c["kind"], folder=c["folder"], vocals=c["vocals"], offset=off, length=n)
        if c.get("language"):
            entry["language"] = c["language"]
        index.append(entry)
        off += n
    arr.flush()
    del arr
    tmp.replace(out / noise_bank.AUDIO)
    audio_sha = noise_bank.sha256_file(out / noise_bank.AUDIO)
    hours = {k: round(sum(c["length"] for c in index if c["kind"] == k) / noise_bank.SR / 3600, 3) for k in kinds}
    left = "MUSAN music without an ANNOTATIONS line" + ("" if "song" in kinds else ", music with vocals") + \
        ("" if "speech" in kinds else ", speech")
    info = dict(version=noise_bank.VERSION, sr=noise_bank.SR, dtype="int16",
                audio_bytes=(out / noise_bank.AUDIO).stat().st_size, audio_sha256=audio_sha, samples=total,
                hours=hours, music_hours_cap=a.music_hours, song_hours_cap=a.song_hours or None,
                speech_hours_cap=a.speech_hours or None, source=SOURCE, licence="CC BY 4.0", left_out=left,
                clips=index)
    (out / noise_bank.INDEX).write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    sha = noise_bank.sha256_file(out / noise_bank.INDEX)
    print(f"wrote {out}: {len(index)} clips, " + ", ".join(f"{k} {h} h" for k, h in hours.items())
          + f", {info['audio_bytes'] / 2**30:.2f} GiB; index.json sha256 {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
