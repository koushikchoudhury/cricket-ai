# Sharing the app

There are two ways to share the Streamlit app with other people.

| | Laptop tunnel | Hosted on Streamlit Community Cloud |
|---|---|---|
| Link | Changes every time you start it | Fixed, e.g. `https://<your-name>-cricket-ai.streamlit.app` |
| Works when | Your laptop and the app are running | Always (wakes up on first visit after 12 idle hours) |
| Language model | Ollama on your GPU (free) | Groq's free tier (no card), with a daily question cap |
| Setup | Two commands | About 30 minutes, once |
| Good for | Live demos and interviews | Your CV, LinkedIn and GitHub README |

## 1. Laptop tunnel (no code, works today)

1. Install `cloudflared`: download `cloudflared-windows-amd64.msi` from
   <https://github.com/cloudflare/cloudflared/releases/latest>, run it, then open a new terminal.
2. Start the app: `cricket-ai app`
3. In a second terminal: `cloudflared tunnel --url http://localhost:8501`
4. Share the `https://....trycloudflare.com` link it prints.

`Ctrl+C` in the second terminal stops sharing. Cloudflare's quick tunnels need no account,
are meant for testing, and come with no uptime guarantee. Anyone with the link can use the
app, including the agent on your GPU.

## 2. Hosted on Streamlit Community Cloud

The hosted app runs from this GitHub repository. It has no GPU and no Ollama, and your data
and models are not in Git, so three things are needed: a data bundle, a free Groq API key and
the app settings.

### Step 1: build the data bundle (on your laptop)

```bash
cricket-ai bundle ipl
```

This writes `dist/cricket-ai-ipl.zip`: the tables, search index, trained models and reports
the app reads, with a checksum for every file. It lists anything not built yet; those pages
then show how to build them.

### Step 2: upload it as a GitHub Release

1. Open <https://github.com/koushikchoudhury/cricket-ai/releases> and click **Draft a new release**.
2. Tag: `data-v1` (create it). Title: `Data bundle v1`.
3. Drag `dist/cricket-ai-ipl.zip` into the attachments area, then click **Publish release**.
4. The download link is
   `https://github.com/koushikchoudhury/cricket-ai/releases/download/data-v1/cricket-ai-ipl.zip`
   (right-click the file on the release page and copy the link to check).

The repository is public, so the app can download it without a token. The bundle contains
only derived Cricsheet data and your models, no keys.

### Step 3: a free Groq API key

Ollama is not available on the host, so the Ask and Agent pages use an API. Groq's free tier
needs no credit card and cannot run up a bill.

1. Sign in at <https://console.groq.com> (Google or GitHub works).
2. Open **API Keys**, click **Create API Key**, name it `cricket-ai`, and copy it. It starts
   with `gsk_` and is shown only once.
3. Paste it **only** into the Streamlit Secrets box in step 4. Never put it in a chat, a
   commit or any file in this repository. If it leaks, delete it in the Groq console and
   create a new one.

The app uses `openai/gpt-oss-120b` (Groq retired Llama 3.3 70B from the free tier on
2026-08-16). Its free limits are about 30 requests a minute, 1,000 a day, 8,000 tokens a
minute and 200,000 tokens a day, shared by all visitors. One question uses roughly 3,000 to
5,000 tokens, so the app's own limit of 30 questions a day stays well inside it. When Groq
says it is busy (the per-minute limit), Ask still answers from the retrieved sources and
Agent asks the visitor to try again in a minute.

Paid alternatives work the same way: set `ANTHROPIC_API_KEY` with `CRICKET_AI_LLM =
"anthropic"`, or `OPENAI_API_KEY` with `CRICKET_AI_LLM = "openai"`, and set a spend limit in
that provider's console.

### Step 4: create the app

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Create an app from an existing repository:
   - Repository: `koushikchoudhury/cricket-ai`, branch `main`
   - Main file path: `streamlit_app.py`
   - App URL: choose a name, e.g. `koushik-cricket-ai`
3. Open **Advanced settings**:
   - Python version: **3.12**
   - Secrets: paste the contents of `.streamlit/secrets.toml.example` with your values:

```toml
CRICKET_AI_BUNDLE_URL = "https://github.com/koushikchoudhury/cricket-ai/releases/download/data-v1/cricket-ai-ipl.zip"
GROQ_API_KEY = "gsk_..."
CRICKET_AI_LLM = "groq"
CRICKET_AI_DAILY_LIMIT = "30"      # model questions per day, all visitors together
CRICKET_AI_SESSION_LIMIT = "10"    # model questions per visitor
```

4. Click **Deploy**. The first build installs PyTorch (CPU) and the other packages and takes
   several minutes. On first start the app downloads the bundle and the embedding model.

When the limits are used up, Ask keeps working from the retrieved sources and Agent says to
come back tomorrow. The counts reset daily and when the app restarts.

### Updating the hosted app

- **Code:** merge to `main`. Streamlit redeploys automatically.
- **Data or models** (after retraining): run `cricket-ai bundle ipl`, publish a new release
  (`data-v2`), update `CRICKET_AI_BUNDLE_URL` in the app's secrets, then **Reboot** the app
  from its menu so it downloads the new bundle.

### Troubleshooting

| Problem | What to do |
|---|---|
| Build fails while installing `torch` | Remove the `--extra-index-url` line from `requirements.txt`: the build gets larger and slower but uses the standard PyTorch package |
| "Could not download the data bundle" | Open the bundle URL in a browser: it must download the zip without logging in |
| App restarts with "over its resource limits" | Community Cloud allows about 2.7 GB of memory; avoid generating very large synthetic samples on the hosted app |
| Ask and Agent offer only "none" | `GROQ_API_KEY` is missing or misspelled; check the Secrets box, then Reboot |
| "The free language model is busy right now" | Groq's per-minute limit was reached; it clears within a minute. If it happens often, lower `CRICKET_AI_DAILY_LIMIT` |
| Groq says the model is not found | Groq retires models from time to time; pick a current one at <https://console.groq.com/docs/models> and set `CRICKET_AI_LLM_MODEL` in Secrets |

## Data licence

Ball-by-ball data comes from [Cricsheet](https://cricsheet.org) under the Open Data Commons
Attribution License. The app credits it in the sidebar.
