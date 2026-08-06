"""
LLM-Assisted Thematic Analysis (LATA) - Full Posts (title + selftext)

Features:
- Resume-safe execution
- Post ID tracking for audit and reproducibility
- Automatic rate-limit backoff
- Token-safe rolling memory
- Exact research prompts preserved

Important:
The research prompts are preserved exactly from the original analysis script.
Do not edit them if reproducibility is required.
"""

import json
import os
import random
import re
import time

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError, RateLimitError
from tqdm import tqdm

# ========== CONFIG ==========
INPUT_FILE = "reddit_rebatched.csv"
OUTDIR = "LATA_FULL_ANALYSIS"
os.makedirs(OUTDIR, exist_ok=True)

MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.0

# --- Safety controls ---
MAX_REQUEST_TOKENS = 24000   # keep well below TPM limits
MAX_MEMORY_SEGMENTS = 25     # how many prior batch themes we carry
BASE_RETRY_DELAY = 15        # seconds
PER_BATCH_COOLDOWN = 3.0     # seconds spacing to smooth TPM

PROGRESS_FILE = f"{OUTDIR}/progress.json"
MEMORY_FILE = f"{OUTDIR}/memory_checkpoint.txt"
THEMES_FILE = f"{OUTDIR}/02_batch_themes.csv"

# ========== API KEY ==========
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit(
        "OPENAI_API_KEY was not found. Copy .env.example to .env and add your API key."
    )
client = OpenAI(api_key=api_key)

enc = tiktoken.encoding_for_model(MODEL_NAME)

def tok(x):
    return len(enc.encode(x)) if isinstance(x, str) else 0


def trim(x, n):
    return enc.decode(enc.encode(x)[:n]) if tok(x) > n else x


# ========== SAFE CHAT (with proper backoff) ==========
def safe_chat(messages):
    attempt = 0
    while True:
        try:
            return client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
            )
        except RateLimitError as e:
            attempt += 1
            msg = str(e)
            m = re.search(r"try again in\s*([0-9]+(?:\.[0-9]+)?)s", msg)
            if m:
                wait = float(m.group(1)) + random.uniform(1, 4)
            else:
                wait = min(55, BASE_RETRY_DELAY * (1.4 ** attempt)) + random.uniform(1, 4)
            print(f"Rate limit reached. Waiting {wait:.1f} seconds.")
            time.sleep(wait)
        except OpenAIError:
            attempt += 1
            wait = min(45, BASE_RETRY_DELAY * (1.3 ** attempt)) + random.uniform(1, 5)
            print(f"OpenAI API error. Waiting {wait:.1f} seconds.")
            time.sleep(wait)
        except Exception as e:
            attempt += 1
            wait = min(45, BASE_RETRY_DELAY * (1.2 ** attempt)) + random.uniform(1, 5)
            print(f"Unexpected error: {e}\nWaiting {wait:.1f} seconds.")
            time.sleep(wait)


# ========== LOAD DATA ==========
df = pd.read_csv(INPUT_FILE)
required_columns = {"batch_id", "post_id", "title", "selftext"}
missing_columns = sorted(required_columns - set(df.columns))
if missing_columns:
    raise SystemExit(f"Dataset is missing required columns: {missing_columns}")

df["full_input"] = (
    df["title"].fillna("") + " " + df["selftext"].fillna("")
).str.strip()
batch_ids = sorted(df["batch_id"].unique())

print(f"Loaded: {INPUT_FILE}")
print(f"{len(df):,} posts across {len(batch_ids)} batches.\n")

# ========== RESUME CHECKPOINT ==========
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        last_done = json.load(f).get("last_completed_batch", 0)
    print(f"Resume mode: starting at batch {last_done + 1}.\n")
else:
    last_done = 0
    print("Starting a fresh run.\n")

# ========== EXACT PROMPTS (ABSOLUTELY DO NOT CHANGE!) ==========
PROMPT_1 = """
Keep in mind the following background material and contextual information for the subsequent requests. Confirm your understanding by summarizing the information below.

This is an academic research project where you are assisting in conducting inductive thematic analysis on Reddit posts.

Overarching goal of project: to enhance the impact and effectiveness of VA’s campaigns for lethal means safety (LMS), with a particular focus on providing insights to reduce suicide by firearms in rural Veterans.

Specific aims of the project: 
Aim 1: Quantify the volume of conversation on lethal means safety (LMS). 
Aim 2: Uncover themes and sentiment contained within social media conversations. 
Aim 3: Analyze rural-urban differences in volume, themes, and sentiment of conversations on LMS.  

You will assist with just Aim 2. More specifically, you will uncover themes contained within social media posts related to our research topic of lethal means safety, and you will uncover themes related our overarching goal to reduce suicide by firearms in rural veterans.

Your role: You will act as a qualitative data analyst applying a non-reflexive thematic approach to interpretation.

Theoretical frameworks and methods: For this analysis, use inductive thematic analytic methods. Do not apply any specific theoretical model. Focus on content related to how Veterans talk about firearms safety and about their experiences with suicidality.

Data type/sources: We extracted all available posts from nine military and veteran-related subreddits between 2015 and 2024. To be included in the dataset, the original post had to contain at least one keyword related to suicide and one keyword related to firearms. The dataset comprised 1,330 unique users (posters) and 1,569 posts. Each post includes a title, post content, post ID, author username, score (upvotes minus downvotes), upvote ratio, number of comments, and post creation date.

Never make up information that is not in the input text you receive. If the text of a post is very short, empty, or says nothing meaningful, do not make up new things.

Additional input provided with prompt: None.
"""

def P2(txt): return f"""
Considering all the posts provided, generate a list of themes relevant to the overarching goal of the project and Aim 2. For each theme provide a brief explanation of the theme and two to five exemplar quotes from the data. Themes should be distinct from each other yet may be interrelated. Quotes should be direct quotations from the posts.

Additional input provided with prompt: Batch 1 of posts.

Posts:
{txt}
"""

def P3(txt): return f"""
Considering the preliminary themes and quotes provided, as well as the additional posts provided, generate a revised list of themes relevant to the overarching goal of the project and Aim 2. For each theme provide a brief explanation of the theme and two to five exemplar quotes. Exemplar quotes include ones already provided and/or quotes from the additional posts provided. Themes should be distinct from each other yet may be interrelated. Quotes should be direct quotations from the posts.

Additional input provided with prompt: 1) Preliminary themes and quotes from Prompt 2;  2) Batch 2 of posts.

Preliminary themes and quotes:
{txt}
"""

def P3B(txt): return f"""
Give me a list of sub-themes that tells more about this theme. For each sub-theme, provide two to five exemplar quotes. Quotes should be direct quotations from the posts.

Additional input provided with prompt: Theme #1 and associated quotes.

Themes and quotes:
{txt}
"""

# ========== Familiarization (batch 0) ==========
if last_done == 0:
    r1 = safe_chat([
        {"role":"system","content":"You are a helpful academic research assistant."},
        {"role":"user","content":PROMPT_1}
    ])
    familiarization = r1.choices[0].message.content
    open(f"{OUTDIR}/01_familiarization.txt","w",encoding="utf-8").write(familiarization)
    themes_all = []
    memory_texts = []
else:
    familiarization = open(f"{OUTDIR}/01_familiarization.txt","r",encoding="utf-8").read()
    if os.path.exists(THEMES_FILE):
        themes_all = pd.read_csv(THEMES_FILE).to_dict("records")
        print(f"Restored {len(themes_all)} batch themes.")
    else:
        themes_all = []
    memory_texts = []
    if os.path.exists(MEMORY_FILE):
        mem = open(MEMORY_FILE,"r",encoding="utf-8").read().split("\n\n")
        memory_texts.extend(mem[-MAX_MEMORY_SEGMENTS:])
        print(f"Restored {len(memory_texts)} memory segments.")


# ========== Batch Loop ==========
print("\nPass 2: batch theming.\n")

for i, b in enumerate(tqdm(batch_ids), start=1):
    if i <= last_done:
        continue

    posts = df[df.batch_id == b]["full_input"].tolist()
    post_ids = df[df.batch_id == b]["post_id"].astype(str).tolist()
    batch_text = "\n\n".join(posts)

    # Build memory representation
    memory = ""
    if memory_texts:
        # recent memory only
        mem_blob = "\n".join(memory_texts[-MAX_MEMORY_SEGMENTS:])
        memory = trim(mem_blob, MAX_REQUEST_TOKENS // 2)

    prompt2_body = P2(batch_text)

    messages = [{"role":"system","content":"You are a qualitative data analyst conducting inductive thematic analysis."}]
    if memory:
        messages.append({"role":"user","content":memory})
    messages.append({"role":"user","content":prompt2_body})

    r2 = safe_chat(messages)
    out = r2.choices[0].message.content

    themes_all.append({
        "batch_id": b,
        "post_ids": ",".join(post_ids),
        "themes": out
    })
    memory_texts.append(out)
    memory_texts = memory_texts[-MAX_MEMORY_SEGMENTS:]  # trim

    pd.DataFrame(themes_all).to_csv(THEMES_FILE, index=False)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_completed_batch": i}, f)
    with open(MEMORY_FILE,"w",encoding="utf-8") as f:
        f.write("\n\n".join(memory_texts))

    time.sleep(PER_BATCH_COOLDOWN)

print("\nAll batch theming is complete.")

# If just finished batching, require a restart for refinement
if last_done < len(batch_ids):
    print("Run the script again to complete refinement and sub-theme generation.")
    raise SystemExit

# ========== Prompt 3 ==========
print("Pass 3: refining themes.")
prelim = "\n\n".join([x["themes"] for x in themes_all])
r3 = safe_chat([
    {"role":"system","content":"You are a qualitative data analyst refining themes."},
    {"role":"user","content":P3(prelim)}
])
refined = r3.choices[0].message.content
open(f"{OUTDIR}/03_refined_themes.txt","w",encoding="utf-8").write(refined)

# ========== Prompt 3B ==========
print("Pass 3B: generating sub-themes.")
r4 = safe_chat([
    {"role":"system","content":"You expand on themes."},
    {"role":"user","content":P3B(refined)}
])
subthemes = r4.choices[0].message.content
open(f"{OUTDIR}/04_subthemes.txt","w",encoding="utf-8").write(subthemes)

# Final CSV for reporting
rows = [{"Stage":"Familiarization","Content":familiarization}]
rows += [{"Stage":f"Batch_{x['batch_id']}","Content":x["themes"]} for x in themes_all]
rows.append({"Stage":"Refined Themes","Content":refined})
rows.append({"Stage":"Sub-Themes","Content":subthemes})
pd.DataFrame(rows).to_csv(f"{OUTDIR}/LATA_Output.csv",index=False,encoding="utf-8")

print("\nComplete. All outputs were saved in:", OUTDIR)