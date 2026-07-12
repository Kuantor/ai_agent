# Mykola Flask Web App

Welcome to **Mykola**, the web-based English learning assistant with a personal touch!

## Quick Start

### 1. Ensure your environment is set up
From the project directory:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Add your API key to `.env`
Copy or edit `.env` and add your Anthropic API key:

```text
ANTHROPIC_API_KEY=sk-ant-api03-...
```

Get a key from: https://console.anthropic.com/account/api-keys

### 3. Run the Flask app
```powershell
.\venv\Scripts\python flask_app.py
```

The app will start at: **http://localhost:5000**

## Features

### 💬 Chat Page (`/`)
- Interactive chat with Mykola
- Ask questions about English grammar, vocabulary, and learning strategies
- Retrieves relevant knowledge from the local knowledge base
- Streams responses from Claude
- Shows which knowledge sources were used for your answer

### 🎨 About Page (`/about`)
- Meet Mykola and learn his story
- Gallery with clickable images (click to see full-size poster)
- Watch Mykola dance in the embedded video
- Browse key features and what Mykola offers

## File Structure

```
├── flask_app.py              # Main Flask application
├── rag.py                    # Knowledge base retrieval
├── templates/
│   ├── base.html            # Base layout
│   ├── index.html           # Chat page
│   └── about.html           # About/gallery page
├── static/
│   ├── style.css            # Global styling
│   └── img/                 # Mykola's media files
│       ├── mykola_avatar.jpg
│       ├── mykola_poster.jpg
│       ├── mykola_poster.jpg
│       └── mykola_choir.mp4
└── knowledge/               # Knowledge base markdown files
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'flask'`
Run: `pip install Flask`

### `Error: Your credit balance is too low`
Add credits to your Anthropic account at:
https://console.anthropic.com/account/billing/overview

### Port 5000 already in use
Change the port in `flask_app.py`:
```python
app.run(debug=True, port=8000)  # Use 8000 instead
```

### Images/videos not loading
Ensure all files exist in `static/img/`:
- `mykola_avatar.jpg`
- `mykola_poster.jpg`
- `mykola_poster.jpg`
- `mykola_choir.mp4`

## Tech Stack

- **Backend**: Flask (Python web framework)
- **RAG**: TF-IDF cosine similarity over markdown documents
- **AI Model**: Claude (Anthropic API)
- **Frontend**: HTML5, CSS3, JavaScript (vanilla)
- **Styling**: Custom CSS with modern design

---

Enjoy chatting with Mykola! 🚀
