# Sharing the app

There are two ways to share the Streamlit app with other people.

| | Laptop tunnel | Hosted on Streamlit Community Cloud |
|---|---|---|
| Link | Changes every time you start it | Fixed, e.g. `https://<your-name>-cricket-ai.streamlit.app` |
| Works when | Your laptop and the app are running | Always (wakes up on first visit after 12 idle hours) |
| Language model | Ollama on your GPU (free) | Your Anthropic or OpenAI API key, with a daily question cap |
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
and models are not in Git, so three things are needed: a data bundle, an API key and the app
settings.

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

### Step 3: an API key with a spending limit

Ollama is not available on the host, so the Ask and Agent pages use an API.

- **Anthropic:** create a key at <https://console.anthropic.com>, then set a monthly spend
  limit in the console's billing or limits settings. The app's default model is
  `claude-haiku-5-5`, the low-cost tier.
- **OpenAI:** create a key at <https://platform.openai.com> and set a usage limit there.

The provider's spend limit is the hard cap. The app's own question limits (step 4) stop
visitors well before that.

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
ANTHROPIC_API_KEY = "sk-ant-..."
CRICKET_AI_LLM = "anthropic"
CRICKET_AI_DAILY_LIMIT = "100"     # model questions per day, all visitors together
CRICKET_AI_SESSION_LIMIT = "15"    # model questions per visitor
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
| Ask and Agent offer only "none" | The API key secret is missing or misspelled; check the Secrets box, then Reboot |

## Data licence

Ball-by-ball data comes from [Cricsheet](https://cricsheet.org) under the Open Data Commons
Attribution License. The app credits it in the sidebar.
