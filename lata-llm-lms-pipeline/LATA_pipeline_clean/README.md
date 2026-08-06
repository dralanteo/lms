# LLM-Assisted Thematic Analysis (LATA)

This project runs a multi-stage, LLM-assisted thematic analysis of Reddit posts using fixed research prompts. The prompt wording in `run_lata.py` is intentionally preserved from the original analysis workflow and should not be edited when reproducibility is required.

## Workflow

1. Familiarization with the project context
2. Batch-level theme generation with rolling memory
3. Refinement of accumulated themes
4. Generation of sub-themes
5. Combined CSV export

## Required input

Place `reddit_rebatched.csv` in the project folder. It must contain:

- `batch_id`
- `post_id`
- `title`
- `selftext`

The script combines `title` and `selftext` for analysis and retains the batch post IDs in `02_batch_themes.csv` for audit purposes.

## Environment setup

From PowerShell in the project folder:

```powershell
python -m venv lata_env
.\lata_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder with your API key:

```text
OPENAI_API_KEY=your_actual_api_key_here
```

Do not commit `.env`. It is excluded by `.gitignore`.

## Run

```powershell
python run_lata.py
```

The batch-theming stage is resume-safe. The script writes a checkpoint after every completed batch. If the run is interrupted, run the same command again.

After the final batch is completed, the script intentionally exits and asks you to run it once more. The second run produces theme refinement, sub-themes, and the final combined CSV.

## Outputs

Outputs are written to `LATA_FULL_ANALYSIS/`:

- `01_familiarization.txt`
- `02_batch_themes.csv`
- `03_refined_themes.txt`
- `04_subthemes.txt`
- `LATA_Output.csv`
- `progress.json`
- `memory_checkpoint.txt`

## Analysis settings

- Model: `gpt-4o`
- Temperature: `0.0`
- Maximum rolling request target: `24,000` tokens
- Rolling-memory segments retained: `25`
- Per-batch cooldown: `3` seconds

## Reproducibility note

The research prompts are marked in `run_lata.py` under:

```python
# ========== EXACT PROMPTS (ABSOLUTELY DO NOT CHANGE!) ==========
```

The cleanup in this version only changes configuration, API-key loading, console messages, validation, documentation, and formatting. The prompt block itself is preserved exactly.
