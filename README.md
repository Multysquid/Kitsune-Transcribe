# Kitsune-Transcribe

Distilling [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) (2B, Apache-2.0)
into a small **Japanese-only** speech-to-text model that trains and runs on an 8 GB consumer GPU.

## Why this teacher

Open-weight Japanese ASR in 2026 is a three-way tie between Whisper large-v3, Qwen3-ASR-1.7B and Cohere
Transcribe depending on the test set (see the [HEROZ](https://techblog.heroz.jp/entry/2026/08/18/120000) and
[Neosophie](https://neosophie.com/en/blog/20260226-japanese-asr-benchmark) benchmarks). Cohere Transcribe has the
best FLEURS-ja / Common Voice CER, a permissive license, and is natively supported in `transformers >= 5.4`
(`model_type = cohere_asr`).

Architecture, which drives the student design:

| block | shape | params |
|---|---|---|
| encoder | FastConformer, 48 layers, d=1280, FFN 5120, 128 mel, 8x subsampling | ~1.8B (90 %) |
| decoder | Transformer, 8 layers, d=1024, FFN 4096 | ~130M |
| vocab | 16,384 SentencePiece tokens shared by 14 languages | |

So the student prunes the **encoder** (48 -> 20 layers, FFN 5120 -> 2560 by activation importance) and halves the
**decoder** (8 -> 4 layers). It keeps the teacher's full 16,384-token vocabulary, tied head and decoder prompt, so
token ids are unchanged: with the head tied to the embeddings, pruning the vocabulary would save at most ~15M
parameters and no compute, and it would add id-remapping risk.

## Pipeline

```
01_prepare_data.py   download sources -> data/shards/<source>/<split>-NNNNN.parquet + data/manifest.jsonl
02_teacher_pass.py   teacher forward passes -> teacher_out/<source>/<split>-NNNNN.{npz,jsonl}
02b_second_opinion.py  second ASR opinion per utterance -> second_out/ (agreement-based label filter)
make_selection.py    which utterances train / evaluate, and why -> selection/*.parquet
03_build_student.py  prune the teacher to the student (20 enc layers, FFN 2560, 4 dec layers), init from teacher
04_distill.py        KL on the stored top-16 + CE on the teacher tokens + L2-SP; TensorBoard (cards in three groups:
                     1_operational, 2_loss_accuracy, 3_misc; first the combined loss KL + 0.8 CE per step and on the
                     gate sets, under 2_loss_accuracy/00_combined and as one Custom Scalars chart, then every eval's
                     val / train CER and KL under 2_loss_accuracy/00_summary) + open-format logs
tools/               export_run.py: a run -> parquet/CSV tables + README; regroup_tb.py: rebuild an older run's
                     TensorBoard files in the three groups
vast/                training image, CI build and the vast.ai run scripts (see vast/README.md)
```

### Data

| source | domain | hours | license |
|---|---|---|---|
| `japanese-asr/whisper_transcriptions.reazonspeech.small` (ReazonSpeech v2 mirror) | TV / broadcast | ~100 | CDLA-Sharing-1.0 (use under Japanese Copyright Act Art. 30-4) |
| `TTS-AGI/emilia-yodas` JA (mirror of Emilia-YODAS, Amphion) | YouTube CC-BY talk, vlogs, streams | ~300 | CC BY 4.0 (from YODAS, CC BY 3.0 uploads) |
| `laion/Emolia` JA `*_standard` (`emilia_nc`: Emilia's non-YODAS part, Amphion) | podcasts, talk shows | optional, in no config | CC BY-NC 4.0: **non-commercial** models only (the repo's cc-by-4.0 tag does not relicense it) |
| `litagin/Galgame_Speech_ASR_16kHz`, first 6 of 115 tars | game / anime voices | ~280 | GPL-3 + **non-commercial**, trained models **must be open-sourced** |
| Common Voice ja (manual download from [Mozilla Data Collective](https://datacollective.mozillafoundation.org)) | read, diverse mics | optional | CC-0 |
| `japanese-asr/ja_asr.{jsut_basic5000,common_voice_8_0,reazonspeech_test}` | eval only | ~22 | JSUT: text CC BY-SA 4.0, audio not redistributable; CV8: CC-0, but Mozilla asks that it not be mirrored; ReazonSpeech test: CDLA-Sharing-1.0 (Art. 30-4). Eval only, so no term on the model; keep refs and samples private |

Only the Emilia hold-out is kept disjoint from training (by video). Nothing dedups training against the gate sets, so
the eval_reazon CER carries up to ~0.3 pp from re-aired broadcasts that are also in reazon_small. The galgame hold-out
is a random utterance sample of the first tar, so it measures seen games and voices, not unseen games.

Every upstream dataset repo is read at a pinned commit (`REVISIONS` in `scripts/01_prepare_data.py`). The label
passes pin theirs too: the teacher at `b1eacc2` (`MODEL_REVISION` in `02_teacher_pass.py`, recorded in
`teacher_out/meta.json`), and kotoba-whisper, the whisper-large-v3 tokenizer and the ReazonSpeech mirrors in
`02b_second_opinion.py` (recorded in `second_out/meta.json`). `03_build_student.py` reads the teacher at the same
pin (`TEACHER_REVISION`, `--teacher-revision`), keys its FFN-importance cache on that commit, refuses a teacher commit
other than the one `teacher_out/meta.json` records, and writes the commit to `student_meta.json` (the student built
before that record was added comes from `b1eacc2` too).

### License of the trained model

This is a non-commercial hobby project. Because Galgame is in the training mix, the student model is
**non-commercial** and its weights are published openly (Galgame's terms require trained models to be open-sourced),
under a non-commercial license (e.g. CC BY-NC 4.0). The model card, [MODEL_CARD.md](MODEL_CARD.md), which every saved
student and checkpoint carries as its README.md, records these terms, credits the training data: ReazonSpeech
(CDLA-Sharing-1.0), Emilia-YODAS (Amphion, CC BY 4.0, built on ESPnet's YODAS, CC BY 3.0) and Galgame_Speech_ASR
(litagin, [GPL-3](https://www.gnu.org/licenses/gpl-3.0.txt) + non-commercial; the dataset's own `license_link` points
to a missing file), and says that the model is modified from the teacher, Cohere Transcribe (Apache-2.0). A release
must also include the Apache-2.0 licence text next to that modified-from notice. Drop `galgame` (and keep `emilia_nc`
out) of the run config's `sources` for a model free of non-commercial terms. The open release is a separate public
repo holding only a chosen `checkpoints/step_N` (weights, config, generation config, tokenizer, processor) plus the
license, notices and model card: the run repos (`kitsune-runs`, `kitsune-data`) stay private, because they carry the
datasets' reference transcripts (`ref` in the teacher outputs, eval tables and samples), and `vast/launch.py` refuses
them if they are not.

### Teacher pass output

Per utterance the teacher pass stores the greedy token sequence, the top-k (default 16) logits at every step
together with the full-vocab log-sum-exp (so exact teacher probabilities are recoverable, `p = exp(logit - lse)`),
the CER of the teacher hypothesis against the dataset transcript (for pseudo-label filtering), and optionally the
final encoder states (`--save-encoder`, ~32 KB per audio second). The format is documented at the top of
[scripts/02_teacher_pass.py](scripts/02_teacher_pass.py).

## Setup

```bash
pip install -r requirements.txt
# CUDA build of torch (the default PyPI wheel on Windows is CPU-only):
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
hf auth login          # accept the model terms on its HF page first
python scripts/00_smoke_test.py
python scripts/01_prepare_data.py --sources reazon_small galgame eval --galgame-shards 6
python scripts/02_teacher_pass.py
```

Hardware target: RTX 4070 Laptop (8 GB). The teacher runs in bf16 with an fp32 LM head (~4.7 GB reserved); measured
RTF is ~0.006-0.008 with length-bucketed batching, so the pass over ~390 h of audio takes roughly 2.5-3.5 h and is
resumable per shard. Close other GPU-heavy apps first: if less than ~6 GB is free the pass spills to shared memory.
