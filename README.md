# MeetingMind

Meeting intelligence pipeline: transcription, speaker diarization,
summarization, decision/action-item extraction, and retrieval-based Q&A.

**Status:** in development. Currently at the dataset and experimentation stage.

## Goal
Take a meeting recording and produce a speaker-labelled transcript,
summary, decisions, action items, and grounded answers to questions
with speaker/timestamp references.

## Current progress
- [x] Project design and component analysis
- [x] AMI corpus annotation exploration (notebook 01)
- [x] Label construction for a decision/action utterance detector,
      split by session to avoid leakage (notebook 02)
- [ ] Baseline and fine-tuned classifier
- [ ] Speech-to-text (Whisper), diarization
- [ ] Summarization, extraction, RAG, API, UI

## Dataset
AMI Meeting Corpus (Carletta et al., 2005), CC BY 4.0.
https://groups.inf.ed.ac.uk/ami/download/
Data is downloaded by the notebooks and is not included in this repo.

## Design decisions
- Only one component is trained (decision/action detector). Whisper,
  pyannote, and embedding models are used pretrained.
- Train/val/test split is by meeting session, not by utterance.
- Known label limitation: only utterances chosen for the extractive
  summary are linked, so labels are incomplete.

## Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
