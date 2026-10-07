"""Build the room impulse response (RIR) bank of the reverb step (kitsune/noise_bank.py's layout, float32, kind rir;
kitsune.trainset's "acoustic steps", DECISIONS H12) from OpenSLR 28 (RIRS_NOISES, Apache 2.0), read straight from its
zip: --per-size simulated RIRs of each room size in --sets (small 1-10 m, medium 10-30 m, large 30-50 m wide; 20,000
each, 200 rooms x 100 positions; a fixed sha256 order of their rir ids picks them) and every real RIR of its rir_list
(325: RWCP, REVERB 2014, Aachen AIR; a multichannel file's first channel). The default sets are the small and medium
rooms, as Kaldi's reverberation recipes take them: the large ones are halls (RT60 up to ~2 s), far from a streamer's
room; the real RIRs keep a few large rooms.

Each RIR is cut to start at its direct path - its largest sample, the first of the clip (so a reverberated row keeps
its timing: a CTC row's frames still hold their sounds) - and to end where its remaining energy falls 60 dB below the
whole (Schroeder's backward integral), at most --max-s seconds, then scaled so the direct path is 1.0. The index
records per clip its rir id, room, set (smallroom | mediumroom | largeroom | real), length and an RT60 estimate (the
backward integral's -5 to -25 dB slope, x 3). index.json's own sha256 is printed: augment.rir_bank_sha256 pins it.

Usage (CPU, a few minutes):
  python tools/build_rir_bank.py --zip D:/kitsune-tmp/rirs/rirs_noises.zip --out D:/kitsune-tmp/rirs/bank/rirs-v1
"""
import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kitsune import noise_bank  # noqa: E402

SOURCE = ("OpenSLR 28, Room Impulse Response and Noise Database (RIRS_NOISES; Tom Ko, Vijayaditya Peddinti, Daniel "
          "Povey, Michael L. Seltzer, Sanjeev Khudanpur, 'A study on data augmentation of reverberant speech for robust "
          "speech recognition', ICASSP 2017), https://www.openslr.org/28/, Apache License 2.0: its simulated RIRs and "
          "its real ones from the RWCP sound scene database, the REVERB 2014 challenge and the Aachen impulse "
          "response (AIR) database; cut to their direct path and -60 dB tail, as one float32 array")
SETS = ("smallroom", "mediumroom", "largeroom")


def rir_list(z: zipfile.ZipFile, path: str) -> list[tuple[str, str, str]]:
    """(rir id, room id, member path) of a RIRS_NOISES rir_list ("--rir-id X --room-id Y path" per line)."""
    out = []
    for line in z.read(path).decode("utf-8", "replace").splitlines():
        p = line.split()
        if len(p) >= 5 and p[0] == "--rir-id" and p[2] == "--room-id":
            out.append((p[1], p[3], p[4]))
    return out


def prepare(h: np.ndarray, max_s: float) -> tuple[np.ndarray, float] | None:
    """h from its direct path (largest |sample|, now index 0) to its -60 dB point (at most max_s), the direct path
    scaled to 1.0, and an RT60 estimate (s; nan when the decay is too short to fit). None for a silent RIR."""
    h = np.asarray(h, dtype=np.float64)
    if h.ndim > 1:
        h = h[:, 0]
    if not np.any(h):
        return None
    d = int(np.argmax(np.abs(h)))
    h = h[d:]
    e = np.cumsum((h ** 2)[::-1])[::-1]  # the energy still to come at each sample
    edc = 10 * np.log10(np.maximum(e / e[0], 1e-30))
    end = int(np.searchsorted(-edc, 60.0))  # the first sample 60 dB down
    end = max(1, min(end, len(h), int(max_s * noise_bank.SR)))
    t5, t25 = np.searchsorted(-edc, 5.0), np.searchsorted(-edc, 25.0)
    rt60 = float(3.0 * (t25 - t5) / noise_bank.SR) if t25 < len(edc) and t25 > t5 else float("nan")
    out = (h[:end] / h[0]).astype(np.float32)
    return out, rt60


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--zip", required=True, help="OpenSLR 28's rirs_noises.zip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-size", type=int, default=600, help="simulated RIRs per room size (default 600)")
    ap.add_argument("--max-s", type=float, default=1.5, help="the longest RIR kept, seconds (default 1.5)")
    ap.add_argument("--sets", default="smallroom,mediumroom",
                    help=f"the simulated room sizes, comma-separated, of {','.join(SETS)} (default smallroom,mediumroom)")
    a = ap.parse_args(argv)
    import soundfile as sf

    z = zipfile.ZipFile(a.zip)
    sets = [x for x in a.sets.split(",") if x]
    if not sets or not set(sets) <= set(SETS):
        raise SystemExit(f"--sets {a.sets!r}: comma-separated room sizes of {SETS}")
    picks = []
    for s in sets:
        rows = rir_list(z, f"RIRS_NOISES/simulated_rirs/{s}/rir_list")
        rows.sort(key=lambda r: hashlib.sha256(r[0].encode()).hexdigest())
        picks += [(s, *r) for r in rows[:a.per_size]]
    picks += [("real", *r) for r in rir_list(z, "RIRS_NOISES/real_rirs_isotropic_noises/rir_list")]
    clips, parts, off, skipped = [], [], 0, []
    for kind_set, rid, room, path in picks:
        x, sr = sf.read(io.BytesIO(z.read(path)), dtype="float64", always_2d=True)
        if sr != noise_bank.SR:
            skipped.append(f"{path}: {sr} Hz")
            continue
        got = prepare(x[:, 0], a.max_s)
        if got is None:
            skipped.append(f"{path}: silent")
            continue
        h, rt60 = got
        clips.append(dict(id=f"rir/{kind_set}/{rid}", kind="rir", set=kind_set, room=room, offset=off,
                          length=len(h), rt60_s=None if np.isnan(rt60) else round(rt60, 3)))
        parts.append(h)
        off += len(h)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    audio = np.concatenate(parts).astype(np.float32)
    np.save(out / noise_bank.AUDIO, audio)
    by_set = {s: sum(1 for c in clips if c["set"] == s) for s in (*SETS, "real")}
    rts = [c["rt60_s"] for c in clips if c["rt60_s"] is not None]
    info = dict(version=noise_bank.VERSION, sr=noise_bank.SR, dtype="float32",
                audio_bytes=(out / noise_bank.AUDIO).stat().st_size,
                audio_sha256=noise_bank.sha256_file(out / noise_bank.AUDIO), samples=len(audio), sets=by_set,
                per_size=a.per_size, max_s=a.max_s, simulated_sets=sets,
                rt60_s_quantiles=[round(float(q), 3) for q in np.quantile(rts, [0.1, 0.5, 0.9])] if rts else None,
                skipped=skipped, source=SOURCE, licence="Apache License 2.0", clips=clips)
    (out / noise_bank.INDEX).write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    sha = noise_bank.sha256_file(out / noise_bank.INDEX)
    print(f"wrote {out}: {len(clips)} RIRs {by_set}, {len(audio) / noise_bank.SR / 60:.1f} min, "
          f"{info['audio_bytes'] / 2**20:.0f} MiB, RT60 p10/50/90 {info['rt60_s_quantiles']} s, {len(skipped)} skipped; "
          f"index.json sha256 {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
