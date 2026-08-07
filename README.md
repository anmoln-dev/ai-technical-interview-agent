# AI Technical Interview Agent 🎯

> An adaptive, accessible AI Technical Interview Agent designed for candidates completing the **31-Day Enterprise AI Engineering Cohort**.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![Google AI Studio](https://img.shields.io/badge/Google_AI_Studio-BYOK-4285F4.svg)](https://aistudio.google.com)
[![Accessibility](https://img.shields.io/badge/WCAG_2.1_AA-Compliant-success.svg)](https://www.w3.org/WAI/WCAG21/quickref/)

---

## 🌟 Overview

Preparing for enterprise AI engineering interviews requires more than memorizing static Q&A. Candidates must demonstrate deep architectural understanding, explain trade-offs, solve live coding challenges, and defend design decisions made across modern AI disciplines.

The **AI Technical Interview Agent** simulates a real technical interview with a Lead AI Architect:
- **Personalized Candidate Context**: Analyzes cohort candidate profiles (role, experience, completed missions, skipped days, attempt counts) to start interviews at candidate-specific entry points.
- **Adaptive Deep-Dives**: Asks a minimum of **8 questions** covering at least **4 distinct curriculum days**, triggering dynamic follow-up questions when responses lack architectural detail.
- **Live Code Testing Tool**: Dynamic coding challenges triggered during the interview with interactive problem statement, timer, and solution evaluator.
- **Google AI Studio BYOK Architecture**: Powered by Google AI Studio (default model `gemma-4-31b-it`) with optional model switching (`gemini-2.5-flash`, `gemini-2.5-pro`) and BYOK API key configuration.
- **Ground-Up Accessibility**: Fully WCAG 2.1 AA compliant with high-contrast modes, font scaling, speech dictation (STT), voice playback (TTS), complete keyboard shortcuts, and screen reader announcements.
- **Actionable Evaluation Report**: Generates structured performance reports with domain radar mastery scores, readiness levels, engineering strengths, growth areas, and targeted curriculum study roadmaps.

---

## 🚀 Features

- **20 Cohort Candidate Profiles**: Pre-loaded candidate data across senior engineers, junior developers, DevOps specialists, and business analysts.
- **31-Day Curriculum Mapping**: 8 modules covering Vector Databases, RAG, Prompt Engineering, Agentic AI, MCP Protocol, Guardrails, and Docker/Kubernetes Deployment.
- **Live Code Testing Tool**: Interactive live code editor with countdown timer and automated test evaluation.
- **Accessibility Suite**:
  - `Alt + S` : Toggle Voice Dictation (Microphone Input)
  - `Alt + R` : Read Current Question Aloud (Text-to-Speech)
  - `Alt + C` : High Contrast Theme Toggle
  - `Alt + F` : Finish & View Feedback Report
  - `Esc` : Close Modals / Navigation
- **Export & Print**: Export formatted PDF evaluation report with domain mastery charts and transcript breakdowns.

---

## ⚙️ Quick Start

### 1. Install Dependencies
```bash
pip install fastapi uvicorn pydantic httpx
```

### 2. Run Backend Server
```bash
python app.py
```

### 3. Open Portal in Browser
Navigate to `http://localhost:8000` to launch the interactive interview portal.

---

## 📡 API Specification

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/health` | `GET` | Health check endpoint |
| `/api/candidates` | `GET` | Returns list of cohort candidates |
| `/api/curriculum` | `GET` | Returns 31-day curriculum data |
| `/api/interview/start` | `POST` | Starts a new candidate interview session |
| `/api/interview/chat` | `POST` | Processes candidate message & returns adaptive AI response |
| `/api/interview/live-test/submit` | `POST` | Evaluates submitted live coding solution |
| `/api/interview/end` | `POST` | Concludes interview & returns feedback report |
| `/api/interview/session/{id}/feedback` | `GET` | Retrieves formatted evaluation report |

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
