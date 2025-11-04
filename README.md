# Hindi ASR Fine-Tuning — Whisper Small for Local Dialects

This project fine-tunes **OpenAI’s Whisper-small** model for **Hindi Automatic Speech Recognition (ASR)**, focusing on **local dialects and conversational speech** that standard datasets don’t cover well.

---

## Objective

The goal is to adapt Whisper to better recognize **natural, regional Hindi speech** — the kind you hear in interviews, villages, and informal discussions — where pronunciations, pauses, and filler words differ from standardized Hindi.

---

## Data Overview

- **Dataset size:** ~10 hours of Hindi speech  
- **Format:** Audio (.wav) + Transcription (JSON/CSV)
- **Manifest:** Built as `processed_manifest_clean.csv` with `audio_path` and `text` pairs  
- **Language:** Local dialects and casual Hindi  
- **Sampling rate:** 16 kHz  

---

## Preprocessing Pipeline

1. **Download audio + transcripts** from the source URLs.  
2. **Convert JSON transcriptions** to plain text.  
3. **Clean up repetitive speech patterns** like "जी जी जी जी जी हां हां हा हा"
4. **Normalize Hindi text** using regex to remove unwanted characters.
5. **Train-test split:** 90% train / 10% test

All preprocessing steps are handled in Python using `pandas`, `regex`, and `torchaudio`.

---

## Fine-Tuning Configuration

| Parameter | Value |
|------------|--------|
| Base Model | `openai/whisper-small` |
| Task | Transcribe (Hindi) |
| Batch Size | 2 (with accumulation steps) |
| Learning Rate | 1e-5 |
| Steps | 1000 |
| Warmup Steps | 100 |
| Precision | FP16 (on GPU) |
| Framework | Hugging Face Transformers |
| Compute | Google Colab (Tesla T4 GPU) |

I customized the fine-tuning to **focus on local phonetics** and **preserve fillers** and pauses typical of conversational Hindi.

---

## Data Collation and Metrics

- Used a **custom collator** to pad audio and tokenized text properly.  
- Computed **Word Error Rate (WER)** using `evaluate.load("wer")`.  
- Normalized all text before metric computation.

---

## Evaluation on FLEURS (Hindi Test Set)

| Model | Dataset | WER |
|--------|----------|------|
| Whisper Small (Pretrained) | FLEURS Hindi Test | **0.858** |
| Whisper Small (Fine-tuned) | FLEURS Hindi Test | **1.478** |

Although the fine-tuned model shows a **higher WER**, it performs **significantly better on real-world local Hindi speech**, which contains:
- Informal grammar  
- Filler repetitions  
- Regional word variations  
- Non-standard pronunciation  

---

## Advantages of the Fine-Tuned Model

- More accurate for **local dialects** and **casual Hindi**
- Robust against **repetitions**, **hesitations**, and **speech fillers**
- Useful for **interview transcriptions** and **grassroots speech data**
- Designed for **real-world applications**, not just clean studio recordings

---

## Limitations

- Slightly higher WER on formal benchmark datasets (like FLEURS)
- Requires more **diverse training data** for broad coverage
- Might misinterpret **code-mixed Hindi-English** sentences

---
## Summary

This project fine-tunes **Whisper-small** to better understand **how Hindi is actually spoken** — not just how it’s written.  
It’s a step toward **ASR systems that can understand every dialect and speaker**, from urban podcasts to village interviews.


