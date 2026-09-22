# 🔍 Google Photos Discovery Engine

An AI-powered research tool that analyzes real user feedback about **photo retrieval problems** in Google Photos. Built to uncover how people remember old photos, where the existing search/retrieval experience breaks down, and identify opportunity areas for improvement.

## Problem Statement

Users accumulate thousands of photos over years but retrieval becomes hard when memory is incomplete:
- *"That small café we went to during our Goa trip"*
- *"The picture of the medicine I took when I was sick last year"*

The user **knows** the photo exists but can't remember when, where, or the exact keywords needed to find it. This tool analyzes public user feedback at scale to understand these retrieval failures.

## What It Does

| Page | Description |
|---|---|
| **Dashboard** | Overview metrics, platform breakdown, affinity mapping methodology, opportunity areas ranked by impact |
| **Ask Insights (Reviews)** | Natural language search over 148 public reviews → TF-IDF retrieval → Groq LLM synthesis |
| **Ask Insights (Survey)** | Same RAG pipeline over primary survey data (add your own) |
| **Review Explorer** | Filter and browse all reviews by platform, theme, sentiment, user segment |
| **Comparison Matrix** | Interactive Altair scatter plot — Impact vs Evidence Volume |

## Tech Stack

- **Frontend**: Streamlit (Python)
- **Search**: Custom TF-IDF (zero dependencies — uses `re`, `math.log`, `collections.Counter`)
- **AI Synthesis**: Groq API (free tier, LLaMA 3.3 70B)
- **Charts**: Altair + Pandas
- **Deployment**: Streamlit Cloud (free)

## Theme Taxonomy

8 retrieval-focused themes emerged from the data:

1. **Vague Memory Retrieval** — Users remember context but can't formulate keywords
2. **Search Accuracy** — Wrong, incomplete, or too many results
3. **AI Search Regression** — Gemini/Ask Photos degrading classic search
4. **Face Recognition** — Grouping errors, misidentification
5. **Temporal Navigation** — Wrong timestamps, no date-range filtering
6. **Screenshot/Document Search** — Can't find receipts, medicine photos, documents
7. **Location Search** — Missing GPS, no indoor location inference
8. **Organization Friction** — Albums, tagging, manual effort

## Setup

1. Clone this repo
2. Get a free Groq API key at [console.groq.com](https://console.groq.com)
3. Deploy on Streamlit Cloud → Add secret: `GROQ_API_KEY = "gsk_..."`

## Data Sources

Reviews collected and analyzed from: Google Play Store, Apple App Store, Reddit, Google Help Community, Twitter/X, YouTube, Trustpilot.

## Adding Survey Data

Replace `data/survey.json` with your survey responses in the format described in the app's survey page.
