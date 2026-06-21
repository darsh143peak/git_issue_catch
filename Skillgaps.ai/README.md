# GitMatch.AI

GitMatch.AI is a Streamlit dashboard that aligns a developer's skills with open GitHub issues. It can fetch live issues through the official GitHub MCP server over stdio, then ranks those issues with `sentence-transformers/all-MiniLM-L6-v2` semantic similarity.

## Setup

1. Use Python 3.12. The previous Python 3.14 virtualenv can fail during Streamlit startup because parts of the Streamlit/uvicorn stack are not stable there yet.

```bash
uv python install 3.12
uv venv --python 3.12
uv pip install -r requirements.txt
```

2. Install Node.js so `npx` is available.

3. Create a local `.env` file:

```env
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_your_classic_token_here
```

4. Run the dashboard:

```bash
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

If MCP, Node, GitHub authentication, or the network is unavailable, the app shows the specific error and falls back to local simulated issues so the ranking workflow still works.
