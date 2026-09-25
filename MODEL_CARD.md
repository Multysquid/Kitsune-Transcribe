---
license: other
license_name: kitsune-noncommercial
license_link: https://github.com/Multysquid/Kitsune-Transcribe/blob/main/MODEL_CARD.md
base_model: CohereLabs/cohere-transcribe-03-2026
language:
- ja
pipeline_tag: automatic-speech-recognition
datasets:
- litagin/Galgame_Speech_ASR_16kHz
- TTS-AGI/emilia-yodas
- japanese-asr/whisper_transcriptions.reazonspeech.small
---

# Kitsune-Transcribe student

A Japanese-only speech-to-text model distilled from Cohere Transcribe by
[Kitsune-Transcribe](https://github.com/Multysquid/Kitsune-Transcribe).

**Status.** The checkpoints a training run keeps, locally and in its private run repo, are working copies. As
Galgame's terms require (see Terms), the checkpoint chosen for release is published openly under a non-commercial
licence; the exact licence is fixed at release. The terms below apply to every copy of these files, private or public.

## Terms

The model is trained on Galgame_Speech_ASR ([litagin/Galgame_Speech_ASR_16kHz](https://huggingface.co/datasets/litagin/Galgame_Speech_ASR_16kHz)),
which is GPL-3.0 with two added conditions that carry over to models trained on it:

- **non-commercial**: no commercial use of this model;
- **open source**: models trained on the dataset must be open-sourced.

## Modification notice (Apache-2.0)

This model is a modified version of
[CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) at commit
`b1eacc2686a3d08ceaae5f24a88b1d519620bc09`, licensed under the
[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0). It was pruned (encoder: 20 of 48 layers,
FFN 5120 -> 2560; decoder: 4 of 8 layers), its output head tied to the token embedding and its BatchNorm statistics
recalibrated, then trained by distillation from the teacher's outputs. The tokenizer and processor files are Cohere's,
unmodified. `student_meta.json` records the exact layers and FFN neurons kept.

## Training data

- Galgame_Speech_ASR (litagin): GPL-3.0, non-commercial, trained models must be open-sourced (see Terms).
- Emilia-YODAS, Japanese part ([TTS-AGI/emilia-yodas](https://huggingface.co/datasets/TTS-AGI/emilia-yodas), a mirror
  of [amphion/Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset) Emilia-YODAS): CC BY 4.0, built
  from [espnet/yodas2](https://huggingface.co/datasets/espnet/yodas2) (CC BY 3.0).
- ReazonSpeech ([japanese-asr/whisper_transcriptions.reazonspeech.small](https://huggingface.co/datasets/japanese-asr/whisper_transcriptions.reazonspeech.small),
  a mirror of [reazon-research/reazonspeech](https://huggingface.co/datasets/reazon-research/reazonspeech)):
  CDLA-Sharing-1.0, used for training under Article 30-4 of the Japanese Copyright Act.

The run's exact sources and settings are in its `config.json` (`runs/<run_id>/` in the run repo).
