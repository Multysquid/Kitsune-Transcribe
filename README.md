# Kitsune-Transcribe

Small **Japanese-only** speech recognition models, made by pruning two open teachers and distilling them on 11,533 hours
of Japanese speech, and the pipeline that trained them. Four models were released on Hugging Face on 2026-10-12.

| Family | Teacher (pinned revision, licence) | Students | Architecture |
|---|---|---|---|
| **T** (encoder-decoder) | [Cohere Transcribe 03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) @ `b1eacc2`, 2.07B, Apache-2.0 | T-0.6B | the teacher pruned: encoder 20 of 48 FastConformer layers with FFN 5120 -> 2560, decoder 4 of 8 layers; the teacher's 16,384-token vocabulary, tied head and decoder prompt |
| **P** (CTC) | [NVIDIA Parakeet TDT-CTC 0.6B ja](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) @ `44edb27`, CC BY 4.0 (its CTC head, 611M) | P-0.3B, P-0.1B, P-0.05B | Parakeet's encoder pruned (16, 8 or 4 of 24 layers; FFN 4096 -> 2560, 768, 768) with its CTC head; greedy CTC over 80 ms frames |

Each student starts from its pruned teacher and learns from the teacher's stored soft targets plus its greedy
transcript. All training data is Japanese, so every parameter serves Japanese.

## Released models

All five repositories are public on Hugging Face under **CC BY-NC 4.0**: the overview
[Multy123/kitsune-transcribe-ja](https://huggingface.co/Multy123/kitsune-transcribe-ja) and one repository per model.
Each model repository holds the bfloat16 weights at its root, seven quantised folders (see
[Quantised folders](#quantised-folders)), an `onnx/` folder with its own README (exported outside this repository),
`student_meta.json` (the exact pruning) and the licence files. The cards on the Hub are the reference for each model.

| Model | Repository | Type | Params | M4 CER | Speed (RTFx) | bf16 size |
|---|---|---|---|---|---|---|
| T-0.6B | [Multy123/kitsune-transcribe-t0.6b-ja](https://huggingface.co/Multy123/kitsune-transcribe-t0.6b-ja) | encoder-decoder (`CohereAsrForConditionalGeneration`) | 617M | 9.42 % | 763x | 1.23 GB |
| P-0.3B | [Multy123/kitsune-transcribe-p0.3b-ja](https://huggingface.co/Multy123/kitsune-transcribe-p0.3b-ja) | CTC (`ParakeetForCTC`) | 309M | 9.73 % | 5,848x | 617 MB |
| P-0.1B | [Multy123/kitsune-transcribe-p0.1b-ja](https://huggingface.co/Multy123/kitsune-transcribe-p0.1b-ja) | CTC | 98M | 11.45 % | 9,544x | 197 MB |
| P-0.05B | [Multy123/kitsune-transcribe-p0.05b-ja](https://huggingface.co/Multy123/kitsune-transcribe-p0.05b-ja) | CTC | 52M | 16.35 % | 11,800x | 104 MB |

Character error rate (CER, %) per test set, greedy decoding, lower is better:

| Model | Params | JSUT | CV8 | ReazonSpeech | Galgame (neutral) | Galgame (all) | Emilia | **M4** | RTFx |
|---|---|---|---|---|---|---|---|---|---|
| Kitsune T-0.6B | 617M | 8.09 | 8.62 | 7.41 | 13.56 | 19.90 | 39.76 | **9.42** | 763x |
| Kitsune P-0.3B | 309M | 7.19 | 8.62 | 10.12 | 12.99 | 19.29 | 40.32 | **9.73** | 5,848x |
| Kitsune P-0.1B | 98M | 8.99 | 10.78 | 12.62 | 13.42 | 19.80 | 41.56 | **11.45** | 9,544x |
| Kitsune P-0.05B | 52M | 13.24 | 16.59 | 20.19 | 15.38 | 21.61 | 47.73 | **16.35** | 11,800x |
| Cohere Transcribe 03-2026 (T's teacher) | 2.07B | 8.30 | 4.07 | 6.28 | 14.75 | 21.42 | 36.96 | 8.35 | 444x |
| Parakeet TDT-CTC 0.6B ja, CTC head (P's teacher) | 611M | 6.71 | 7.58 | 9.71 | 13.32 | 19.64 | 40.61 | 9.33 | 3,989x |
| Whisper large-v3 | 1.54B | 7.19 | 8.51 | 14.50 | 12.41 | 21.90 | 30.65 | 10.65 | 119x |
| Whisper large-v3-turbo | 809M | 7.30 | 9.30 | 10.70 | 13.07 | 21.52 | 39.34 | 10.09 | 240x |
| Kotoba-Whisper v2.0 | 756M | 8.39 | 9.23 | 11.66 | 13.17 | 20.02 | 39.18 | 10.61 | 267x |
| Whisper small | 242M | 14.24 | 15.06 | 27.46 | 18.50 | 28.25 | 37.76 | 18.82 | 346x |

How these were measured:
- **Test sets**, all complete, in one frozen eval manifest (sha256 `ef56dec2…`,
  `kitsune.fullrun.FROZEN_MANIFEST_SHA256`): JSUT basic5000 (5,000 clips), Common Voice 8 ja (4,483), ReazonSpeech
  test (5,263), a 1,000-clip Galgame hold-out, and a 1,000-clip Emilia hold-out from videos not used for training.
  Text is normalised with `kitsune.text.normalize_ja` before scoring; each number is the corpus CER of its set.
- **M4** is the mean of JSUT, CV8, ReazonSpeech and Galgame (neutral). Galgame (neutral) keeps the 810 hold-out clips
  whose subtitle-style reference Kotoba-Whisper v2.0 matched within CER 0.5; Galgame (all) scores every clip with a
  reference (969).
- **Every system** (students, teachers, Whisper) was decoded greedily by this repository's code on the same
  utterances: the students by `scripts/05_evaluate.py`; the teachers by their label passes (`02_teacher_pass.py`,
  `02p_parakeet_pass.py`; their scores are registered in `study/PREREG.json` `baselines`); Whisper by
  `tools/whisper_eval.py`.
- **Speed** is RTFx (seconds of audio per second of processing): one run on one RTX 5090, bf16, PyTorch eager, 200
  test utterances in batches of up to 400 s of audio (`tools/speed_probe.py`). The Kitsune models were timed on their
  1,000-hour study checkpoints of the same architectures. A second RTX 5090 host ran up to about 2x faster; the
  numbers are not comparable with the Open ASR Leaderboard's.

Caveats:
- **Whisper large-v3 is clearly better on spontaneous long-form speech.** It leads on Emilia (podcast and YouTube
  talk) and on the owner's own video tests (VRChat recordings: T-0.6B 21.6 %, P-0.3B 22.7 %, large-v3 15.6 %; a
  vlog: 22.2 / 19.1 / 10.0 %). The teachers trail it there as well, so the gap looks set by the teachers and the
  training data. Emilia's references are themselves Whisper-medium transcripts, which favours the Whisper family on
  that column.
- **T-0.6B's M4 lead over P-0.3B comes from ReazonSpeech.** Most of that gap is P-0.3B transcribing speech at the
  clip edges that the ReazonSpeech references leave out; on JSUT T-0.6B trails P-0.3B and on CV8 they tie.
- **In-domain sets.** The Galgame hold-out comes from the same games and voices as the training clips, and the
  training data is ReazonSpeech, Galgame and Emilia: no read speech, so JSUT and CV8 are the out-of-domain sets.
  Kotoba-Whisper chose the Galgame (neutral) clips, which favours Kotoba-Whisper on that column.
- **T-0.6B writes text for non-speech audio** (songs, music, foreign speech): run a voice activity detector first.
- One training run per model; the tables carry no confidence intervals.

## Using the models

Install (tested with Python 3.12, torch 2.14.0, transformers 5.13.1 and torchao 0.18.0; input is 16 kHz mono audio):

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130   # CPU only: .../whl/cpu
pip install transformers==5.13.1 librosa
pip install torchao==0.18.0   # only for the quantised folders' CUDA kernels; not needed on CPU
```

The bf16 root needs only transformers. T-0.6B, from its card:

```python
import librosa
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

repo = "Multy123/kitsune-transcribe-t0.6b-ja"
processor = AutoProcessor.from_pretrained(repo)
model = CohereAsrForConditionalGeneration.from_pretrained(repo, dtype=torch.bfloat16).to("cuda").eval()  # float32 on CPU

audio, _ = librosa.load("speech.wav", sr=16000, mono=True)  # resample to 16 kHz
# language="ja", punctuation=True: the decoder prompt the model was trained with
inputs = processor(audio, sampling_rate=16000, language="ja", punctuation=True,
                   return_tensors="pt").to(model.device, model.dtype)
with torch.no_grad():
    ids = model.generate(**inputs, max_new_tokens=16 + int(10 * min(len(audio) / 16000, 35)))
prompt_len = inputs["decoder_input_ids"].shape[1]
print("".join(processor.batch_decode(ids[:, prompt_len:], skip_special_tokens=True)))  # audio over 30 s: one row per piece
```

The P models, from the P-0.3B card (P-0.1B and P-0.05B differ only in the repository id):

```python
import librosa
import torch
from transformers import AutoProcessor, ParakeetForCTC

repo = "Multy123/kitsune-transcribe-p0.3b-ja"
processor = AutoProcessor.from_pretrained(repo)
model = ParakeetForCTC.from_pretrained(repo, dtype=torch.bfloat16).to("cuda").eval()  # or dtype=torch.float32 on CPU

audio, _ = librosa.load("speech.wav", sr=16000, mono=True)  # resample to 16 kHz
inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(model.device, model.dtype)
with torch.no_grad():
    ids = model.generate(**inputs)
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

**Repetition stop (T-0.6B).** The reported T scores used the output cap above plus `kitsune.generation.RepetitionStop`,
which ends a row once its last 24 generated tokens repeat with a period of at most 12. Continuing the T snippet, with
a clone of this repository:

```python
# git clone https://github.com/Multysquid/Kitsune-Transcribe
import sys
sys.path.insert(0, "Kitsune-Transcribe")
from transformers import StoppingCriteriaList
from kitsune.generation import RepetitionStop

prompt_len = inputs["decoder_input_ids"].shape[1]
with torch.no_grad():
    ids = model.generate(**inputs, max_new_tokens=16 + int(10 * min(len(audio) / 16000, 35)),
                         stopping_criteria=StoppingCriteriaList([RepetitionStop(prompt_len)]))
```

**Long audio.** Every model was trained on utterances of up to 30 s, and the long-form tests used 8 s chunks with
1.5 s overlap. Cut chunks at pauses where you can: T-0.6B tends to drop a phrase that a chunk boundary splits, where a
CTC model writes it up to the cut. A P model's chunk that ends inside a sentence can still end with 。: strip sentence
marks at chunk ends that are not pauses. P-0.1B was trained on single whole utterances (no cut or joined inputs, unlike
P-0.3B and P-0.05B): it writes 。 at the end of almost every input, and one input holding several utterances costs it
a lot.

### Quantised folders

| Folder | Weights / activations | T-0.6B | P-0.3B | P-0.1B | P-0.05B |
|---|---|---|---|---|---|
| (root) | bfloat16 | 1.23 GB, 9.42 | 617 MB, 9.73 | 197 MB, 11.45 | 104 MB, 16.35 |
| `fp16/` | float16 (encoder self-attention in float32) | 1.23 GB, 9.42 | 617 MB, not scored | 197 MB, not scored | 104 MB, not scored |
| `fp8-w8a8/` | FP8 E4M3 / FP8 E4M3 | 638 MB, 9.45 | 314 MB, 9.70 | 103 MB, not scored | 56 MB, 16.29 |
| `int8-w8a16/` | int8 / bfloat16 | 638 MB, 9.42 | 314 MB, 9.70 | 103 MB, not scored | 56 MB, 16.34 |
| `int8-w8a8/` | int8 / int8 | 638 MB, 9.44 | 314 MB, 9.73 | 103 MB, not scored | 56 MB, 16.55 |
| `nvfp4-w4a16/` | NVFP4 / bfloat16 | 375 MB, 9.44 | 179 MB, 9.74 | 61 MB, not scored | 34 MB, 17.18 |
| `nvfp4-w4a4/` | NVFP4 / NVFP4 | 375 MB, 9.48 | 179 MB, 9.85 | 61 MB, not scored | 34 MB, 19.51 |
| `mxfp4-w4a4/` | MXFP4 / MXFP4 | 356 MB, 9.69 | 170 MB, 10.90 | 58 MB, not scored | 33 MB, 26.24 |

Size and M4 CER (%) per folder. T-0.6B stays within +2.9 % of its bf16 root in every format; MXFP4 costs P-0.3B +12 %
and P-0.05B +61 %, and NVFP4 with 4-bit activations costs P-0.05B +19 %. GPU support (tested on an RTX 5090 with
torchao 0.18): `fp8-w8a8/` needs compute capability 8.9 or newer, `nvfp4-w4a4/` 10.0 or newer (Blackwell); on older
GPUs torchao stops with an AssertionError. MXFP4 has no kernel and always runs emulated. In PyTorch eager mode the
quantised folders ran slower than the bf16 root in the release tests: they save disk and download size, not time
(and an emulated layer, as on CPU, holds its weight dequantised in float32).

The quantised layers are stored as packed tensors (`qweight`, `qscale`, ...) next to a `quantization.json` recipe, so
load every folder with `kitsune.quant.load_quantized`: `from_pretrained` would not find their `.weight` tensors (and
would run `fp16/`'s encoder self-attention in float16, which overflows). The cards pin commit `76b28a1` for the loader;
`main` keeps the same loader and format (`tests/test_release_contract.py` guards its signature and constants). From
the T-0.6B card:

```python
# git clone https://github.com/Multysquid/Kitsune-Transcribe && git -C Kitsune-Transcribe checkout 76b28a1
# pip install torchao==0.18.0  # the CUDA kernels; not needed on CPU (tested with torch 2.14.0, transformers 5.13.1)
import sys
sys.path.insert(0, "Kitsune-Transcribe")

import librosa
import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor
from kitsune.quant import load_quantized

repo, folder, device = "Multy123/kitsune-transcribe-t0.6b-ja", "int8-w8a16", "cuda"  # any folder (GPU support: see above); or device = "cpu"
path = f"{snapshot_download(repo, allow_patterns=[f'{folder}/*'])}/{folder}"
processor = AutoProcessor.from_pretrained(repo)
model, recipe = load_quantized(path, device)
# each format runs under the autocast its recipe was scored with: float16 for fp16/, bfloat16 for the others
amp = torch.float16 if recipe["format"] == "fp16" else torch.bfloat16

audio, _ = librosa.load("speech.wav", sr=16000, mono=True)
inputs = processor(audio, sampling_rate=16000, language="ja", punctuation=True,
                   return_tensors="pt").to(device)
with torch.no_grad(), torch.autocast(device, dtype=amp):
    ids = model.generate(**inputs, max_new_tokens=16 + int(10 * min(len(audio) / 16000, 35)))
prompt_len = inputs["decoder_input_ids"].shape[1]
print("".join(processor.batch_decode(ids[:, prompt_len:], skip_special_tokens=True)))  # audio over 30 s: one row per piece
```

For a P model, drop `language`, `punctuation` and `prompt_len`, as its card does:

```python
inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
with torch.no_grad(), torch.autocast(device, dtype=amp):
    ids = model.generate(**inputs)
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

`python -m kitsune.quant inspect <folder>` prints a folder's recipe summary and any problems with its files.

## Licences and data credits

- **Code** in this repository: MIT ([LICENSE](LICENSE)).
- **Released weights**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Credit "Kitsune-Transcribe
  \<model\> by Multy123". Each model repository's `LICENSE.md` has the full terms and notices.
- **Base-model material** keeps its licence: T-0.6B's tokenizer and processor files are Cohere's (Apache-2.0; Cohere
  does not endorse the derivative), the P models' are NVIDIA's (CC BY 4.0; NVIDIA does not endorse it).
- [MODEL_CARD.md](MODEL_CARD.md) is not the release card: it is the template `kitsune/student.py` writes as
  `README.md` into every T-family student and checkpoint the pipeline saves. P-family checkpoints get a generated card
  with NVIDIA's CC BY 4.0 attribution (`kitsune/ctc_student.py`); trained ones also carry the training data's
  non-commercial terms, as that card says.

| Data | Hugging Face source (pinned in `scripts/01_prepare_data.py` `REVISIONS`) | Used for | Licence |
|---|---|---|---|
| ReazonSpeech (TV) | `japanese-asr/whisper_transcriptions.reazonspeech.small` and `.large`, mirrors of `reazon-research/reazonspeech` | training | CDLA-Sharing-1.0, used for training under Article 30-4 of the Japanese Copyright Act |
| Emilia-YODAS, Japanese (YouTube talk) | `TTS-AGI/emilia-yodas` | training, Emilia hold-out | CC BY 4.0, built from `espnet/yodas2` (CC BY 3.0) |
| Emolia, Japanese (podcasts, talk shows) | `laion/Emolia` (`*_standard` tars: Emilia's non-YODAS part) | training | CC BY 4.0 by its card; Amphion's Emilia, its upstream, is CC BY-NC 4.0 for this part |
| Galgame_Speech_ASR (visual-novel voices) | `litagin/Galgame_Speech_ASR_16kHz` @ `3fb8665` | training, Galgame hold-out | GPL-3.0 with two added conditions that carry over to trained models: non-commercial use only, and the models must be open-sourced |
| JSUT basic5000, Common Voice 8 ja, ReazonSpeech test | `japanese-asr/ja_asr.jsut_basic5000`, `.common_voice_8_0`, `.reazonspeech_test` | evaluation only | JSUT: text CC BY-SA 4.0, audio not redistributable; CV8: CC0, Mozilla asks that it not be mirrored; ReazonSpeech test: CDLA-Sharing-1.0 (Art. 30-4) |
| MUSAN (noise, music, songs, speech) | openslr.org/17, via `tools/build_noise_bank.py` | augmentation of T-0.6B and P-0.05B | CC BY 4.0 |
| OpenSLR 28 RIRS_NOISES (room impulse responses) | openslr.org/28, via `tools/build_rir_bank.py` | augmentation of T-0.6B and P-0.05B | Apache-2.0 |

Galgame's conditions and Emolia's upstream licence are why the weights are non-commercial. The private repositories
the pipeline uses (`Multy123/kitsune-data`, `Multy123/kitsune-runs`, `Multy123/kitsune-scratch`) stay private because
they hold the datasets' reference transcripts; `vast/launch.py` refuses a public one.

## Repository map

| Path | What it is |
|---|---|
| `kitsune/` | The library. Release contract: `quant.py` (quantised folders: export, `load_quantized`, the readouts; `python -m kitsune.quant export\|readout\|inspect\|compare\|selftest`) and `generation.py` (`RepetitionStop`). Models: `student.py` (T students), `ctc_student.py` (P students), `patches.py`, `features.py`, `audio.py`, `text.py` (`normalize_ja`, CER), `parakeet.py` (pinned Parakeet files, TDT decoder). Data and labels: `store.py`, `extent.py`, `fetchahead.py`, `labelpass.py`, `parakeet_targets.py`, `ctc_targets.py`, `ctc_preflight.py`, `devslice.py`, `trainset.py`, `aed_cuts.py`, `acoustics.py`, `noise_bank.py`. Training: `kd.py`, `ctc_kd.py`, `runlog.py`, `profiling.py`, `scratch.py`, `heartbeat.py`. Evaluation: `evaluate.py`, `ctc_eval.py`, `study_stats.py`, `whisper.py`, `prereg.py`. Boxes: `fullrun.py` (box registry and shared names), `full_queue.py` (a box's queue; `python -m kitsune.full_queue run\|plan\|build-stores\|resume-pull\|check-resume`), `study_queue.py` (its base), `netgate.py` (download gate) |
| `scripts/` | The numbered pipeline: `01_prepare_data.py` (audio shards from the pinned sources), `02_teacher_pass.py` (Cohere labels), `02p_parakeet_pass.py` (Parakeet labels), `02b_second_opinion.py` (agreement filter), `make_selection.py` (which rows train and evaluate, and why), `03_build_student.py` (T students), `03c_build_ctc_student.py` (P students), `04_distill.py` (the trainer), `05_evaluate.py` (a checkpoint's complete eval) |
| `tools/` | `make_full_configs.py` (generates `configs/full/*.json`, checks the box registry), `make_study_configs.py` (the run-config layer it builds on; no CLI), `full_plan.py` (step plans), `aed_cut_table.py`, `build_noise_bank.py`, `build_rir_bank.py` (augmentation inputs), `speed_probe.py`, `whisper_eval.py`, `full_report.py` (the report: offer table, quantised variants, Whisper, error-vs-speed chart), `export_run.py` (a run to parquet/CSV tables), `publish_parakeet.py` (the converted Parakeet teacher and its pins) |
| `vast/` | Renting and running vast.ai boxes: `launch.py` (on the laptop), `onstart_stub.sh`, `onstart.sh`, `bootstrap.sh`, `watchdog.sh`, `supervise.py`, `finish.py` (on the box), `label.py` and `label_sync.py` (the label box), `blocklist.json`, and [vast/README.md](vast/README.md), the operator guide |
| `configs/` | `full.json` (the label box's extent and selection), `next_run_template.json` (generator input), `full/boxes.json` (the hand-written box registry), `full/data-*.json`, `full/full-*.json` and `full/smoke-*.json` (generated; the configs the boxes ran, byte for byte), `full/plan/` (the step-plan and speed records), `study/study-t06.json` (box smoke-b's eval-store config) |
| `study/` | `PREREG.json` (the size study's frozen pre-registration: the eval manifest and the teachers' baselines that `kitsune/evaluate.py` reads) and `data.json` (the size study's data config: `make_selection.py --config study/data.json` built the study selection and the frozen eval manifest, and box smoke-b's data config copies it). Both frozen |
| `docker/`, `.github/` | The training image (`Dockerfile`, `smoke_import.py`) and its CI build (`workflows/image.yml`) |
| `tests/` | The pytest suite (see [Tests](#tests)) |
| `MODEL_CARD.md`, `LICENSE`, `AGENTS.md` | The T checkpoint card template, the MIT licence, notes for coding agents |
| `requirements*.txt` | `requirements.txt`: the laptop's packages, unpinned (the models, scripts, tools and tests on CPU). `requirements-dev.txt`: the test runner. `requirements-train.txt`: the training image's exact pins |

## How the released models were made

1. **Audio.** `scripts/01_prepare_data.py` reads every source at a pinned commit and writes 16 kHz shards plus an id
   sidecar, so any machine rebuilds byte-identical audio; `--extent-config` ingests a config's whole extent in the
   canonical order of `kitsune/extent.py`.
2. **Labels (the label box).** One rented RTX 5090 (`python vast/launch.py --job label ...`, run by `vast/label.py`)
   rebuilt the audio of `configs/full.json`'s extent and labelled about 13,000 h with both teachers: Cohere
   (`02_teacher_pass.py`: greedy tokens, top-16 log-probs and the log-sum-exp per step) and Parakeet
   (`02p_parakeet_pass.py`: TDT and CTC soft targets), plus the agreement filter (`02b_second_opinion.py`). The labels
   were uploaded write-once (`vast/label_sync.py`) and sealed as `labels/full` in `Multy123/kitsune-data` at commit
   `34aa704`. The Parakeet teacher was converted once on the laptop (`tools/publish_parakeet.py`) and is sha256-pinned
   in `kitsune/parakeet.py`.
3. **Selection.** The frozen eval manifest (`ef56dec2…`) comes from the size study
   (`make_selection.py --config study/data.json`) and is registered in `study/PREREG.json`. The training selection
   `labels/full/selections/full_study/full.parquet` (sha256 `e9a0695a…`) filters 13,002.7 h of labels: the 02b
   agreement filter, Cohere's and Parakeet's transcripts within CER 0.5 of each other, no text of 15 or more characters
   equal to an eval reference, and a held-out dev slice leave **11,533.2 h, 7,027,930 utterances**. It was built with
   `configs/full/data-full.json`, which the cleanup removed; `configs/full/data-p01.json` builds the same bytes
   (`--labels-root` and `--manifest` are local pulls of `labels/full` and
   `labels/full/selections/study_manifest.json` from the private data repo):
   ```bash
   python scripts/make_selection.py --config configs/full/data-p01.json --labels-root D:/kitsune-labels/full \
       --manifest D:/kitsune-study/selections/study_manifest.json --skip-audio-check \
       --out D:/stage/labels/full/selections/full_study/full.parquet
   ```
4. **Students** (on the laptop CPU; uploaded to `students/study/` in the data repo). T-0.6B: `03_build_student.py
   --device cpu --teacher-dtype fp32 --step0-eval-utts 20 --bn keep --ffn-from students/b20x2560-d4 --out
   students/study/t06` (the FFN selection of the viability run's student). P models: `03c_build_ctc_student.py
   --enc-layers 16 --ffn 2560 --name P-0.3B ...`, then 8/768 and 4/768, with the FFN neurons ranked on the viability
   run's 1,000 calibration ids (`prereg.CALIB_IDS_SHA256`). The full commands are in each script's docstring.
5. **Configs.** `tools/full_plan.py` measures the step plans on the selection; `tools/make_full_configs.py` turns the
   size study's run configs into `configs/full/*.json` (`--import-launch-plan`, `--import-speed`), and its `--check`
   checks them and the hand-written registry `configs/full/boxes.json`. The augmentation inputs come from
   `tools/build_noise_bank.py`, `tools/build_rir_bank.py` and `tools/aed_cut_table.py`.
6. **Training**, one 1x RTX 5090 box per model: `vast/launch.py --job full --box <box>` rents it; on the box
   `onstart.sh` starts `watchdog.sh` (the hard cost cap) and `bootstrap.sh` (download gate, pulls, audio rebuild,
   coverage check), then `supervise.py` runs `python -m kitsune.full_queue run`. The queue builds the stores, trains
   with `scripts/04_distill.py`, scores the final weights (`scripts/05_evaluate.py`: M4 on the frozen manifest), scores
   and exports seven quantised variants (`python -m kitsune.quant readout`), and runs the Whisper and speed items;
   `finish.py` uploads everything to `Multy123/kitsune-runs`, verifies it and destroys the box.
7. **Baselines and report.** The Whisper models were scored on box full-p with `tools/whisper_eval.py` (pinned in
   `kitsune/whisper.py`); `tools/full_report.py` builds the tables and the error-vs-speed chart on the laptop.
8. **Release** (outside this repository, by hand): the chosen checkpoints were staged, missing or outdated quantised
   folders exported with `python -m kitsune.quant export`, the ONNX folders and the cards made, and everything
   uploaded. The cards are maintained by hand on the Hub.

| | T-0.6B | P-0.3B | P-0.1B | P-0.05B |
|---|---|---|---|---|
| Box (`configs/full/boxes.json`) | `full-t` | `full-p` | `p01` (box 1) | `p005` |
| Code commit | `23081bb`, resumed at `bde0f36` | `30b13f3` | `7715f3f` | `ad0861f` |
| Trainer / data config | `full-t06.json` / `data-t.json` | `full-p03.json` / `data-p.json` | `full-p01.json` / `data-p01.json` | `full-p005.json` / `data-p01.json` |
| Init (data repo) | `students/study/t06` | `students/study/p03` | `students/study/p01` | `students/study/p005` |
| Run id | `full-t06-20261010T055818Z` | `full-p03-20261003T230143Z` | `full-p01-20261001T184145Z` | `full-p005-20261008T160232Z` |
| Released step | 143,891 | 211,720 | 107,910 | 80,939 |
| `kitsune-runs` revision | `d1c468e0` | `61afd417` | `c4604304` | `d2caddfc` |
| Weights sha256 | `96bf1f57…` | `1dfd38a5…` | `2e25a59d…` | `6dbbdb62…` |
| Loss | KL on the top-16 + 0.8 x CE | frame KL + 0.8 x CTC | frame KL + 0.8 x CTC | frame KL + 0.8 x CTC |
| Schedule | 6 epochs, peak LR 2e-4, 300 warm-up steps, last 20 % decayed | 6 epochs, 2e-4, 300 warm-up | 4 epochs, 1e-3, 1,000 warm-up | planned 10 epochs at 1e-3; the early stop started the decay at step 67,449 |
| Augmentation (`tools/make_full_configs.py`) | `RECIPE_AED_H15`: cuts, joins, noise, music and speech, reverb, gain, codecs, voiced end trim | `RECIPE` (v2): cuts and joins | SpecAugment only | `RECIPE_V4`: v2 plus the acoustics |

The configs in `configs/full/` are byte-identical to what each box trained with. Later re-runs of the three P runs (box
p01's recipe test and box p-cool's cooldown re-runs, neither released) wrote into the same run directories, so the
heads of those directories on `kitsune-runs` are not the released models: use the revisions above.

## Reproducing, or running a new box

Everything a box needs is in three **private** Hugging Face repositories: `Multy123/kitsune-data` (dataset: the sealed
labels, selections, student inits, augmentation banks, the converted Parakeet teacher), `Multy123/kitsune-runs` (run
directories, readouts, queue summaries) and `Multy123/kitsune-scratch` (timed training states). Reproducing starts
from those frozen artefacts, not from a fresh label run: the 02b Kotoba judge that chose the Galgame-neutral rows is
gone, and its stored hypotheses are a frozen input (`tools/full_report.py` `KOTOBA_JSONL_SHA256`).

[vast/README.md](vast/README.md) is the operator guide: one-time setup (image, Hub repos and token, vast.ai account),
the full-data boxes (launch, what launch checks, on the box, how it ends, resume on a new host), the label box and
`onstart.sh`. A launch without `--yes` only searches and prints (PowerShell, on the laptop):

```powershell
python vast\launch.py --job full --box full-t --image-tag main --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --scratch-repo Multy123/kitsune-scratch
```

Today this look-only run prints the offers and then two refusals. Box full-t already ran: its Hub queue summary has
its training done, so a fresh launch needs `--fresh-over-done` (a new run is cleaner as a new registry entry). And
launch refuses a box with quantised readouts unless box smoke-b passed at an ancestor commit with the same quant code
(`QUANT_CODE` in `vast/launch.py`); that code changed after the last smoke-b, so the next such box needs a new smoke-b
or `--allow-unverified-quant`. The vast/README.md section "Commands" explains both.

Renting (`--yes`), stopping and destroying boxes, Hub uploads and deletions, and tokens are the owner's steps. CI
builds the image `ghcr.io/multysquid/kitsune-train` for each pushed commit that changes its inputs.

## Laptop data prep and labelling

These still run on a single machine (paths as in each script's docstring). A local corpus, about 400 h (the laptop's
first one):

```bash
python scripts/01_prepare_data.py --sources reazon_small galgame eval --galgame-shards 6 --whisper-dir data/whisper
python scripts/02_teacher_pass.py                                            # Cohere labels -> teacher_out/
python scripts/02p_parakeet_pass.py --model-dir models/parakeet-tdt_ctc-0.6b-ja-hf   # -> parakeet_out/
python scripts/02b_second_opinion.py --whisper-dir data/whisper --parakeet-out parakeet_out   # -> second_out/
```

`--whisper-dir` keeps the ReazonSpeech mirrors' Whisper transcripts while each file is local; 02b reads them as the
second opinion of the ReazonSpeech rows. The label box's whole extent (about 571 GB of downloads and 1.4 TB of disk)
and its selection, the same scripts at full size:

```bash
python scripts/01_prepare_data.py --extent-config configs/full.json --whisper-dir data/whisper
python scripts/make_selection.py --config configs/full.json
```

The output formats are documented at the top of `scripts/02_teacher_pass.py` and `kitsune/parakeet_targets.py`.
`--limit-rows` on 01, 02 and 02p writes a small smoke run to separate `*_smoke` folders. `tools/publish_parakeet.py`
converted the Parakeet teacher once; it is the only generator of the `PARAKEET_FILES` pins and
`kitsune/parakeet_golden.json`, so run it only to replace them on purpose.

## Tests

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu   # the tests run on CPU
pip install -r requirements.txt -r requirements-dev.txt -c requirements-train.txt
python -m pytest tests -n 8 --dist loadfile -m "not slow" -p no:cacheprovider --basetemp <a dir on a disk with 30 GB free>
```

`requirements.txt` is unpinned. `-c requirements-train.txt` (the image's pins, as a constraints file: it installs
nothing of its own) holds each package to the exact version the image, the boxes and the tests used. Some tests pin
library behaviour (`GOLDEN` in `tests/test_recipe_v5.py` pins numpy 2.5.3's random streams), so other versions can
fail them. On the owner's laptop
use the command in [AGENTS.md](AGENTS.md) instead (one pytest process, `-n 3`, `OMP_NUM_THREADS=2`,
`KITSUNE_REAL_DATA_ROOT` pointing at an empty directory).

- **CPU only.** `tests/conftest.py` hides the GPU (`CUDA_VISIBLE_DEVICES=-1`) and gives each xdist worker
  `KITSUNE_TEST_THREADS` threads (default 1). `--dist loadfile` keeps a file's module fixtures on one worker.
  `--basetemp` should be on a disk with room: the extent tests refuse one with under 30 GB free.
- **Markers.** `slow` marks the tests that load the real 2B teacher; deselect them with `-m "not slow"`.
- **Real data.** Tests that read real labels or shards skip when the data is absent (a fresh clone has none).
  `KITSUNE_REAL_DATA_ROOT=<checkout with data>` points them at one, `KITSUNE_REQUIRE_REAL_DATA=1` turns those skips
  into failures, and a few take their own variable (`KITSUNE_PARAKEET_OUT`, `KITSUNE_LABELS_ROOT`,
  `KITSUNE_STUDY_STUDENTS`, `KITSUNE_FULL_SELECTION_DIR`, `KITSUNE_STUDY_SELECTION`, `KITSUNE_STUDY_RESULTS`).
- **Shell scripts** are tested with Git Bash on Windows (WSL's `bash.exe` is skipped on purpose); the tests that need
  a Linux bash with `flock` take `KITSUNE_LINUX_BASH` or skip.
- **torchao** is not in `requirements.txt`: without it the quant tests emulate and `tests/test_quant_torchao.py`
  skips; the image CI runs that file with torchao.
- **No pytest CI.** The only workflow is the image build (`.github/workflows/image.yml`), which runs
  `docker/smoke_import.py`, `tests/test_quant.py` and `tests/test_quant_torchao.py` inside the image. Run the suite
  locally before merging, and `python tools/make_full_configs.py --check` (exit 0 and
  "<checkout>/configs/full: up to date", with backslashes on Windows).

## History

| Phase | Dates (2026) | What happened | Outcome |
|---|---|---|---|
| Laptop | 09-20 to 09-24 | Data download; the Cohere teacher pass on an RTX 4070 Laptop (402.5 h of audio in 2.06 h); the agreement filter; overfit tests and a model bake-off | Parakeet set as the bar; the laptop labels were later adopted by the label box |
| Viability run | 09-25 | T-0.6B's shape on 519 h, 4 h on one A100 (PRs #2-#5, #7, #9) | 12.15 % pooled CER, inconclusive, $4.38. The limit was generalisation, not fitting, so the whole download was labelled |
| Label box | 09-25 to 09-26 | One RTX 5090 for ~11 h labelled ~13,000 h with both teachers (PRs #8, #10) | `labels/full` sealed at `kitsune-data` `34aa704`; ~$9 |
| Size study | 09-25 to 09-27 | Seven students and a bridge (a from-scratch control) on the same 1,000 h at equal A100 compute, pre-registered in `study/PREREG.json` (PRs #11-#17) | Parakeet halves cheaply (P-0.3B at 1.12x its teacher's M4), and its limit lies between P-0.3B and P-0.1B; Transcribe loses most at the first cut; ~$63.7 |
| Full-data runs | 10-01 to 10-12 | One RTX 5090 box per model on 11,533 h: box 1 P-0.1B (10-01), smoke-b's quant check (10-02), P-0.3B (10-03), P-0.05B (10-08), cooldown re-runs of the three P runs on box p-cool (10-09, not released), T-0.6B (10-10 to 10-12) (PRs #18-#54) | The four released models; about $213 of vast.ai time over all phases (the recorded box costs added up) |
| Release | 10-12 | Four model repositories on Hugging Face (bf16, seven quantised folders, ONNX) and an overview | Public, CC BY-NC 4.0 |
| Cleanup | after the release | The code of the finished phases removed | The last commit with all of it is `76b28a1` |

The removed code stays in git history at `76b28a1`, the commit the release cards pin. Read a removed file with
`git show 76b28a1:<path>` (for example `configs/viability.json`, `study/PREREG.md`, `tools/study_report.py`) and list
them with `git diff --stat --diff-filter=D 76b28a1 HEAD`. Removed with them: the laptop launchers and supervisors, the
viability (`--job train`) and size-study (`--job study`) box paths, the study's config-generator CLI
(`tools/make_study_configs.py` keeps only `run_config`) and its configs except `configs/study/study-t06.json`, the study
queue's phases (`kitsune/study_queue.py` keeps the queue base), the chained box p01-chain and box p-cool with their registry entries, the trainer's
study-only modes (old run configs still load: see `RETIRED_KEYS` in `scripts/04_distill.py`), the 02b Kotoba judge,
and `tools/asr_bakeoff.py`, `tools/box1_go.py`, `tools/judge_report.py`, `tools/label_checks.py` and
`tools/regroup_tb.py`.
