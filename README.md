# WikiGPT-12M

A small GPT-style language model built from scratch, trained on English Wikipedia on a free Kaggle GPU, tuned to hold a simple conversation, and served by a local chat app with saved history.

The project follows the "build an LLM from scratch" path (tokenizer, attention, GPT block, pretraining, generation, finetuning) and keeps every stage in plain, readable PyTorch.

| | |
|---|---|
| Parameters | 12,649,600 |
| Architecture | GPT-2 style decoder, 8 layers, 5 heads, width 320, pre-LayerNorm, tied embeddings |
| Context length | 512 tokens |
| Tokenizer | Byte-level BPE, 8,192 tokens, trained on Wikipedia |
| Pretraining data | About 500M tokens of English Wikipedia (`wikimedia/wikipedia`, `20231101.en`) |
| Pretraining | 3,500 steps x 131,072 tokens = about 460M tokens |
| Chat tuning | `yahma/alpaca-cleaned` + `HuggingFaceTB/everyday-conversations-llama3.1-2k` |
| Exported size | About 25 MB (fp16) |

**What to expect.** The model writes fluent, Wikipedia-flavoured English and answers in the right shape, but it is often wrong on facts. That is normal at 12M parameters (GPT-2 small is 124M). The point of the project is the full pipeline, not a capable assistant.

**Can it run on a laptop?** Yes. 25 MB of weights runs comfortably on any small NVIDIA GPU (such as an RTX 3050) or on the CPU alone.

---

## Contents

1. [How it works](#1-how-it-works)
2. [Project structure](#2-project-structure)
3. [Requirements](#3-requirements)
4. [Part A: one-time Kaggle setup](#4-part-a-one-time-kaggle-setup)
5. [Part B: notebook 01, prepare data](#5-part-b-notebook-01-prepare-data)
6. [Part C: notebook 02, pretrain](#6-part-c-notebook-02-pretrain)
7. [Part D: notebook 03, chat tuning and export](#7-part-d-notebook-03-chat-tuning-and-export)
8. [Part E: run the chat app on your computer](#8-part-e-run-the-chat-app-on-your-computer)
9. [Updating the code on Kaggle](#9-updating-the-code-on-kaggle)
10. [Configuration reference](#10-configuration-reference)
11. [Tests](#11-tests)
12. [Troubleshooting](#12-troubleshooting)
13. [FAQ](#13-faq)
14. [References and licenses](#14-references-and-licenses)

---

## 1. How it works

```
English Wikipedia (parquet shards on Hugging Face)
    |   scripts/prepare_data.py    clean articles, train the BPE tokenizer, encode to uint16
    v
data/train.bin, val.bin, tokenizer.json                    notebook 01 (Kaggle CPU)
    |   scripts/pretrain.py        next-token prediction on random 512-token windows
    v
runs/pretrain/ckpt.pt                                      notebook 02 (Kaggle T4 GPU)
    |   scripts/finetune.py        chat format, loss on assistant replies only
    v
runs/finetune/ckpt.pt                                      notebook 03 (Kaggle T4 GPU)
    |   scripts/export.py          fp16 weights + config + tokenizer, verified on reload
    v
wikigpt-12m.zip  ->  models/wikigpt-12m/  ->  python -m app    (your computer)
```

**Where each concept from the playlists lives:**

| Concept | File |
|---|---|
| Byte pair encoding tokenizer | `wikigpt/tokenizer.py` |
| Data loading with sliding windows (input and target shifted by one) | `wikigpt/data.py` |
| Causal multi-head self-attention | `wikigpt/model.py` (`CausalSelfAttention`) |
| Transformer block, GPT model, weight tying | `wikigpt/model.py` (`Block`, `GPT`) |
| Pretraining loop, learning rate warmup and cosine decay, evaluation | `wikigpt/trainer.py` |
| Text generation with temperature and top-k | `wikigpt/generate.py` |
| Instruction finetuning with a prompt template and loss masking | `wikigpt/chat.py`, `wikigpt/sft.py` |

**Chat format.** Chat tuning and the app use the same template, defined once in `wikigpt/chat.py`:

```
<|user|>
What is the capital of France?<|end|><|assistant|>
Paris is the capital of France.<|end|>
```

The app ends the prompt with `<|assistant|>` and a newline. The model writes the reply and stops at `<|end|>`. During chat tuning the loss is computed only on reply tokens, so the model learns to answer, not to imitate the user. If someone types a marker such as `<|end|>` into a message, it is kept as plain text, so it cannot fake a turn.

---

## 2. Project structure

```
LLM/
├── wikigpt/                  core package, used by the Kaggle scripts and the app
│   ├── config.py             ModelConfig, TrainConfig, FinetuneConfig and CLI flag helpers
│   ├── tokenizer.py          train, save and load the BPE tokenizer; special token ids
│   ├── model.py              the GPT model
│   ├── data.py               Wikipedia shard download, article cleaning, token files, batches
│   ├── chat.py               chat template, training examples, prompt building, chat_reply
│   ├── sft.py                chat dataset download, padding and batching
│   ├── trainer.py            training loop: fp16, gradient accumulation, LR schedule, logging
│   ├── checkpoint.py         training checkpoints and the exported bundle
│   └── generate.py           sampling
├── scripts/                  entry points run by the notebooks
│   ├── prepare_data.py
│   ├── pretrain.py
│   ├── finetune.py
│   └── export.py
├── tests/                    pytest suite, runs on Kaggle at the top of every notebook
├── notebooks/                import these into Kaggle
│   ├── 01-prepare-data.ipynb
│   ├── 02-pretrain.ipynb
│   └── 03-finetune-export.ipynb
├── app/                      local chat app
│   ├── __main__.py           `python -m app`
│   ├── settings.py           paths and generation settings, overridable by environment variables
│   ├── server.py             FastAPI app, loads the model once at startup
│   ├── api.py                REST endpoints
│   ├── store.py              SQLite chat history
│   ├── engine.py             turns a chat history into a reply
│   └── static/               index.html, style.css, app.js (no frameworks, works offline)
├── models/                   put the exported model here (created in Part E)
├── requirements.txt          packages for Kaggle (all preinstalled there)
├── requirements-app.txt      packages for the local app (plus torch, see Part E)
└── README.md
```

---

## 3. Requirements

**Kaggle**
- A Kaggle account with a **verified phone number**. GPU and internet access in notebooks need it.
- GPU quota: Kaggle gives roughly 30 GPU hours per week (check your current quota on your Kaggle profile under Settings). This project uses about 3 to 5 GPU hours.
- Session limit: 12 hours per run. Pretraining fits well inside that, and can be resumed if needed.

**Your computer (only for the chat app)**
- Windows 10 or 11 (macOS and Linux work too, with the usual shell differences).
- Python 3.10 or newer. Check with `python --version`.
- About 1 GB of disk for Python packages and 25 MB for the model.
- An NVIDIA GPU is optional. On the CPU, replies take a few seconds longer.

---

## 4. Part A: one-time Kaggle setup

### A1. Verify your phone number
kaggle.com, then your profile picture, then **Settings**, then **Phone Verification**. Without this the GPU and Internet options are missing in notebooks.

### A2. Package the code
Open PowerShell in the project folder and run:

```powershell
cd C:\Users\nandu\Desktop\Rishabh\Rishabh\logs\LLM
tar.exe -a -c -f wikigpt-code.zip --exclude=__pycache__ --exclude=app/data wikigpt scripts tests app requirements.txt requirements-app.txt README.md
tar.exe -tf wikigpt-code.zip
```

The listing must show paths such as `wikigpt/model.py` and `scripts/pretrain.py` at the top level of the zip, not inside an extra folder.

Use `tar.exe`, not `Compress-Archive`. `tar.exe` ships with Windows 10 and 11 and writes zip paths with forward slashes. `Compress-Archive` in Windows PowerShell 5.1 can write backslashes, which Linux (and Kaggle) may not unpack into folders.

### A3. Create the code dataset
1. kaggle.com, then **Datasets** in the left menu, then **New Dataset**.
2. Drag in `wikigpt-code.zip`. Kaggle unpacks it automatically.
3. Title: `wikigpt-code`. Leave it **Private**. Click **Create**.

### A4. Import the three notebooks
For each file in `notebooks/`:
1. kaggle.com, then **Code**, then **New Notebook**.
2. In the editor menu: **File**, then **Import Notebook**, then choose the `.ipynb` file.
3. Rename the notebook at the top left, for example `wikigpt-01-prepare-data`, `wikigpt-02-pretrain`, `wikigpt-03-finetune-export`.

Every notebook starts with the same setup cell. It finds the code anywhere under `/kaggle/input`, so the exact dataset path does not matter, and it prints the attached inputs, the torch version, the GPU and the free disk space. Read those lines first whenever something looks wrong.

---

## 5. Part B: notebook 01, prepare data

**Settings** (right sidebar, Session options):
- Accelerator: **None**. This notebook only needs the CPU, which saves GPU quota.
- Internet: **On**.

**Inputs:** **Add Input**, then **Datasets**, then **Your Datasets**, then `wikigpt-code`.

**Run:** **Save Version**, then **Save & Run All (Commit)**, then **Save**. The run continues in the background, so you can close the tab. To watch it, open the notebook's version and view its log.

**What it does:**
1. Runs the test suite.
2. Downloads Wikipedia shards one at a time, in a fixed seeded order.
3. Cleans each article: drops articles under 500 characters, cuts trailing "See also / References / External links" sections, collapses blank lines.
4. Trains the 8,192-token tokenizer on the first 300M characters.
5. Encodes articles with `<|endoftext|>` between them. Every 200th article goes to validation (0.5%).
6. Stops after 500M training tokens and deletes the downloaded parquet files.

**Duration:** roughly 30 to 90 minutes.

**Expected result** (the last cell prints `meta.json`):
- `train_tokens` a little over 500,000,000
- `val_tokens` about 2,500,000
- `vocab_size` 8192
- `chars_per_token` about 3.5 to 4.5
- `shards` lists the 3 to 5 shards used

The last cell also prints a sample sentence split into tokens and checks that decoding gives back the exact text.

**Output:** `/kaggle/working/data/` with `train.bin` (about 1 GB), `val.bin`, `tokenizer.json`, `meta.json`.

---

## 6. Part C: notebook 02, pretrain

**Settings:**
- Accelerator: **GPU T4 x2** (the code uses one of the two GPUs; the model is too small to need both).
- Internet: **On**.

**Inputs:**
- `wikigpt-code`
- The output of notebook 01: **Add Input**, then **Notebooks** (or **Your Work**), then `wikigpt-01-prepare-data`.

**Run:** **Save Version**, then **Save & Run All (Commit)**.

**What it does:**
1. Runs the tests.
2. **Smoke run:** 50 quick steps in a throwaway folder. Checks that the loss falls and measures speed, then deletes its output.
3. **Full run:** 3,500 steps.
   - Prints a log line every 10 steps.
   - Evaluates on validation data every 250 steps.
   - Saves `ckpt.pt` and writes sample text every 500 steps.
4. Shows `loss.png` and the latest samples.

**Reading the log:**
```
parameters: 12,649,600
training on cuda from step 0 to 3500
step 250/3500 | train 4.6012 | val 4.5871 (ppl 98.2) | lr 8.37e-04 | 85.3k tok/s | 6.9 min
```
The numbers above are only an example of the format.

**What to expect:**
- **Starting loss:** about 9.0, which is ln(8192), a uniform guess over the vocabulary.
- **First smoke-run line:** slow, because `torch.compile` builds kernels during the first steps. Read tok/s from the later smoke-run lines.
- **Duration:** about 460,000,000 divided by tok/s seconds. Typically 2 to 4 hours on a T4.
- **Loss over the run:** both losses fall quickly at first and then slowly. A final validation loss somewhere around 3 to 4 is typical for this size; what matters is that it keeps falling without spikes.
- **Samples:** `samples.txt` moves from random fragments to grammatical, Wikipedia-like sentences.

**Output:** `/kaggle/working/runs/pretrain/` with:

| file | contents |
|---|---|
| `ckpt.pt` | latest checkpoint: model, optimizer, scaler, step and configs |
| `log.csv` | step, train_loss, val_loss, lr, tokens_per_sec, elapsed_min |
| `loss.png` | loss curve |
| `samples.txt` | sample text from each checkpoint |
| `tokenizer.json` | a copy, so notebook 03 needs only this output |

### Resuming after a session ends
A checkpoint is written every 500 steps, so at most 500 steps are lost.

1. Open the stopped version and check that its **Output** contains `runs/pretrain/ckpt.pt`.
2. Open the notebook editor. **Add Input**, then **Notebooks**, then select this same notebook (`wikigpt-02-pretrain`) so its previous output is attached. Keep the notebook 01 input attached too.
3. In the **Full run** cell, change `RESUME = False` to `RESUME = True`.
4. **Save & Run All (Commit)**.

Training continues from the saved step with the same model settings. The learning rate schedule continues where it stopped, and `log.csv` is carried over and trimmed to the checkpoint step. Set `RESUME` back to `False` before starting a fresh run.

### Passing extra options
Add flags to the `args` list in the **Full run** cell before `run(*args)`:

```python
args += ["--batch-size", 16, "--grad-accum", 16]
args += ["--no-compile"]
```

All flags are listed in [Configuration reference](#10-configuration-reference).

---

## 7. Part D: notebook 03, chat tuning and export

**Settings:**
- Accelerator: **GPU T4 x2**.
- Internet: **On**. The chat datasets are downloaded here.

**Inputs:**
- `wikigpt-code`
- The output of notebook 02 (`wikigpt-02-pretrain`). The tokenizer is included in it, so notebook 01 is not needed here.

**Run:** **Save Version**, then **Save & Run All (Commit)**.

**What it does:**
1. Runs the tests.
2. **Chat tuning:**
   - Downloads the two chat datasets and formats them with the chat template.
   - Drops examples longer than 512 tokens.
   - Holds out 2% for validation.
   - Trains for 2 epochs with dropout 0.1. The loss counts only the assistant's reply tokens.
   - Under an hour on a T4.
3. Prints `loss.png` and replies to 10 fixed prompts. They are also saved to `runs/finetune/replies.txt`.
4. **Export:**
   - Writes the bundle the app loads: fp16 weights, config and tokenizer.
   - Reloads the bundle and checks that it gives exactly the same greedy reply as the checkpoint.
   - Zips the bundle.

**What the replies look like:** short answers in the right shape that stop on their own. Greetings get greetings back. Facts are often wrong or vague. If replies are fragments of Wikipedia text rather than answers, see [Troubleshooting](#12-troubleshooting).

**Download the model:**
1. Open the finished version of the notebook and go to the **Output** tab.
2. Find `export/wikigpt-12m.zip` (about 25 MB) and click the download icon.

In an interactive session, you can use the link printed by the last cell instead.

The zip contains one folder:
```
wikigpt-12m/
├── model.pt          fp16 weights
├── config.json       model settings
└── tokenizer.json    tokenizer
```

---

## 8. Part E: run the chat app on your computer

All commands are for PowerShell, run in the project folder.

### E1. Create a virtual environment
```powershell
cd C:\Users\nandu\Desktop\Rishabh\Rishabh\logs\LLM
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
If activation is blocked with "running scripts is disabled on this system", run this once and then activate again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### E2. Install PyTorch
Pick one.

**With an NVIDIA GPU:**
1. Open https://pytorch.org/get-started/locally/.
2. Select Stable, Windows, Pip, Python, and the newest CUDA version listed.
3. Copy the command, keeping only `torch`. It looks like this, though the `cu` number may differ:
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   ```

**CPU only:**
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Check it:
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
`True` means the GPU will be used. `False` is fine too; the app will run on the CPU.

### E3. Install the app packages
```powershell
pip install -r requirements-app.txt
```

### E4. Put the model in place
```powershell
Expand-Archive -Path "$HOME\Downloads\wikigpt-12m.zip" -DestinationPath models
Get-ChildItem models\wikigpt-12m
```
You should see `model.pt`, `config.json` and `tokenizer.json` inside `models\wikigpt-12m\`.

### E5. Start the app
```powershell
python -m app
```
Open http://127.0.0.1:8000 in your browser. Stop the app with Ctrl+C.

**Using it:**
- **New chat** starts a conversation. The chat appears in the sidebar after the first message, titled with that message.
- Enter sends. Shift+Enter adds a new line.
- Click a chat in the sidebar to reopen it. Hover over it and click **Delete** to remove it.
- The model sees the most recent turns of the conversation that fit in its 512-token window.

**Next time**, from the project folder:
```powershell
.\.venv\Scripts\Activate.ps1
python -m app
```

**Chat history** is stored in `app\data\chats.db` (SQLite). To clear all history, stop the app and delete that file.

**Changing settings**, for example forcing the CPU or another port:
```powershell
$env:WIKIGPT_DEVICE = "cpu"
$env:WIKIGPT_PORT = "8080"
python -m app
```
These last for the current PowerShell window. All variables are listed in [Configuration reference](#10-configuration-reference).

---

## 9. Updating the code on Kaggle

After changing any file:
1. Re-create the zip with the command in [A2](#a2-package-the-code).
2. Open the `wikigpt-code` dataset on Kaggle, click **New Version**, upload the new zip, and save.
3. In each notebook editor, open the input panel. If `wikigpt-code` shows an update notice, click it to use the latest version. If it does not, remove the input and add it again.
4. Rerun the notebook. The setup cell prints the code path, and the tests run against the new code.

---

## 10. Configuration reference

### Model (`ModelConfig`, flags accepted by `pretrain.py`)

| Setting | Flag | Default | Meaning |
|---|---|---|---|
| vocab_size | (none) | 8192 | Taken from the tokenizer automatically |
| context_length | `--context-length` | 512 | Maximum tokens the model sees at once |
| d_model | `--d-model` | 320 | Width of every layer |
| n_layers | `--n-layers` | 8 | Number of transformer blocks |
| n_heads | `--n-heads` | 5 | Attention heads; `d_model` must be divisible by it |
| dropout | `--dropout` | 0.0 | Dropout during pretraining |

On resume, model settings come from the checkpoint and these flags are ignored.

### Pretraining (`TrainConfig`, flags for `pretrain.py`)

| Setting | Flag | Default | Meaning |
|---|---|---|---|
| batch_size | `--batch-size` | 32 | Sequences per micro-batch |
| grad_accum | `--grad-accum` | 8 | Micro-batches per optimizer step (32 x 8 x 512 = 131,072 tokens per step) |
| max_steps | `--max-steps` | 3500 | Optimizer steps |
| lr | `--lr` | 1e-3 | Peak learning rate |
| min_lr | `--min-lr` | 1e-4 | Learning rate at the end of the cosine decay |
| warmup_steps | `--warmup-steps` | 300 | Linear warmup steps |
| weight_decay | `--weight-decay` | 0.1 | AdamW weight decay (matrices only, not biases or norms) |
| grad_clip | `--grad-clip` | 1.0 | Gradient norm clipping |
| log_interval | `--log-interval` | 10 | Steps between log lines |
| eval_interval | `--eval-interval` | 250 | Steps between validation runs |
| eval_batches | `--eval-batches` | 40 | Batches per validation run |
| ckpt_interval | `--ckpt-interval` | 500 | Steps between checkpoints and samples |
| compile | `--compile` / `--no-compile` | on | `torch.compile`; switched off automatically on GPUs below compute capability 7.0, such as the P100 |
| seed | `--seed` | 1337 | Random seed |

Other `pretrain.py` flags: `--data` (folder with `train.bin`, `val.bin`, `tokenizer.json`, required), `--out` (default `/kaggle/working/runs/pretrain`), `--resume` (path to a `ckpt.pt`).

### Chat tuning (`FinetuneConfig`, flags for `finetune.py`)
These settings use the same flags as pretraining. The defaults that differ, and two extra settings:

| Setting | Flag | Default | Meaning |
|---|---|---|---|
| epochs | `--epochs` | 2 | Passes over the chat data |
| max_steps | `--max-steps` | 0 | 0 means epochs x batches per epoch |
| grad_accum | `--grad-accum` | 1 | |
| lr | `--lr` | 3e-4 | |
| min_lr | `--min-lr` | 3e-5 | |
| warmup_steps | `--warmup-steps` | 50 | |
| weight_decay | `--weight-decay` | 0.0 | |
| dropout | `--dropout` | 0.1 | Dropout during chat tuning |
| eval_interval | `--eval-interval` | 200 | |
| eval_batches | `--eval-batches` | 20 | |
| ckpt_interval | `--ckpt-interval` | 1000 | |
| compile | `--compile` / `--no-compile` | off | Sequence lengths vary, so compiling does not pay off |

Other `finetune.py` flags: `--checkpoint` (pretrain `ckpt.pt` with `tokenizer.json` next to it, required), `--out` (default `/kaggle/working/runs/finetune`).

### Data preparation (`prepare_data.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--out` | `/kaggle/working/data` | Output folder |
| `--target-tokens` | 500000000 | Stop after this many training tokens |
| `--vocab-size` | 8192 | Tokenizer size, at most 65535 |
| `--tokenizer-chars` | 300000000 | Characters used to train the tokenizer |
| `--encode-batch` | 512 | Articles encoded per batch; lower it if memory runs out |
| `--val-every` | 200 | Every Nth article goes to validation |
| `--seed` | 1337 | Shard order |

### Export (`export.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | required | Chat-tuned `ckpt.pt` with `tokenizer.json` next to it |
| `--out` | `/kaggle/working/export` | Output folder |
| `--name` | `wikigpt-12m` | Bundle folder and zip name |

### App environment variables

| Variable | Default | Meaning |
|---|---|---|
| `WIKIGPT_MODEL_DIR` | `models/wikigpt-12m` | Folder with `model.pt`, `config.json`, `tokenizer.json` |
| `WIKIGPT_DB` | `app/data/chats.db` | Chat history database |
| `WIKIGPT_DEVICE` | `auto` | `auto`, `cpu` or `cuda` |
| `WIKIGPT_HOST` | `127.0.0.1` | Address to listen on |
| `WIKIGPT_PORT` | `8000` | Port |
| `WIKIGPT_MAX_NEW_TOKENS` | `200` | Longest reply in tokens, 1 to 256 (half the context, so the conversation always fits) |
| `WIKIGPT_TEMPERATURE` | `0.8` | Higher is more varied, lower is more predictable; 0 always picks the likeliest token |
| `WIKIGPT_TOP_K` | `40` | Sample only from the K likeliest tokens |
| `WIKIGPT_REPETITION_PENALTY` | `1.1` | Values above 1 discourage reusing tokens the reply has already produced |

---

## 11. Tests

The tests use tiny models and a tiny tokenizer and finish in seconds on the CPU. Every notebook runs the full suite right after the setup cell, and the notebook stops if a test fails.

| File | Checks |
|---|---|
| `tests/test_tokenizer.py` | exact encode/decode round trip, including non-ASCII text; special tokens map to single ids; save and load |
| `tests/test_model.py` | output shapes; future tokens never affect earlier positions (causal mask); tied weights; the default model has exactly 12,649,600 parameters; loss falls when overfitting one batch; ignored targets; generation past the context window; stop tokens |
| `tests/test_data.py` | article cleaning; seeded shard order; `<|endoftext|>` after each article; targets are the inputs shifted by one |
| `tests/test_chat.py` | only reply tokens are trained; over-long examples are dropped; prompts end with the reply header; long histories keep the newest turns; an over-long message keeps its role marker; typed markers such as `<|end|>` stay plain text; padding and batching |
| `tests/test_store.py` | creating, titling, listing, ordering and deleting chats |

To run a single file or test, add a cell after the setup cell in any notebook:

```python
run("-m", "pytest", CODE / "tests" / "test_model.py", "-q", "-p", "no:cacheprovider")
run("-m", "pytest", CODE / "tests", "-q", "-p", "no:cacheprovider", "-k", "causal")
```

Keep `-p no:cacheprovider`. The code input on Kaggle is read-only, and without that flag pytest tries to write a cache there.

---

## 12. Troubleshooting

Start with the setup cell output: it shows which inputs are attached, whether a GPU is visible and how much disk is free.

### Kaggle setup

| Symptom | Cause | Fix |
|---|---|---|
| No GPU or Internet option in Session options | Phone number not verified | Verify it in Settings (step A1) |
| `FileNotFoundError: wikigpt/__init__.py not found under /kaggle/input` | Code dataset not attached, or the zip had an extra top-level folder | Attach `wikigpt-code`. Check `tar.exe -tf wikigpt-code.zip` lists `wikigpt/...` at the top level and re-upload |
| `FileNotFoundError: train.bin not found` (notebook 02) | Notebook 01 output not attached | Add Input, Notebooks, `wikigpt-01-prepare-data`. It must be a version that finished successfully |
| `FileNotFoundError: pretrain/ckpt.pt not found` (notebook 03) | Notebook 02 output not attached | Add Input, Notebooks, `wikigpt-02-pretrain` |
| Changes to the code seem ignored | Notebook still uses the old dataset version | Section 9, step 3 |
| `ModuleNotFoundError` for `tokenizers`, `datasets`, `pyarrow` or `matplotlib` | Rare change in the Kaggle image | Add a cell after setup: `run("-m", "pip", "install", "-q", "-r", CODE / "requirements.txt")` (Internet must be On) |
| Tests fail with "Read-only file system" | pytest cache writing into the input folder | Use the exact test command from the notebook, including `-p no:cacheprovider` |
| `RuntimeError: command failed with exit code 1` | A script crashed | Scroll up in that cell's output; the real error is printed above this line |

### Data preparation (notebook 01)

| Symptom | Cause | Fix |
|---|---|---|
| `download failed (...), retrying` | Temporary network or Hugging Face rate limit | It retries 3 times. If it still fails, run the cell again; finished shards are skipped |
| `LocalEntryNotFoundError` or connection errors | Internet is Off | Turn Internet On in Session options |
| `No space left on device` | Disk full | Lower the target: `run(..., "--target-tokens", 300_000_000)`. Check free disk in the setup output |
| Kernel restarts or dies during encoding | Out of memory | Pass `"--encode-batch", 128` |
| Kernel dies while training the tokenizer | Out of memory | Pass `"--tokenizer-chars", 100_000_000` |
| `train_tokens` far below 500M | The run stopped early | Run the cell again to continue, or check the log for errors |
| Want to start data prep over from scratch | Old files in `/kaggle/working/data` (interactive session) | Delete that folder first: `shutil.rmtree(DATA)` |

### Pretraining (notebook 02)

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | Batch too large for the GPU | `args += ["--batch-size", 16, "--grad-accum", 16]` keeps the same tokens per step |
| Errors that mention `torch._dynamo`, `inductor` or `triton` | `torch.compile` problem | `args += ["--no-compile"]` (somewhat slower, otherwise identical) |
| First steps very slow | `torch.compile` warming up | Normal; read tok/s from later log lines |
| `gpu: none` in the setup output | Accelerator not set | Session options, Accelerator, GPU T4 x2 |
| Got a P100 instead of a T4 | Accelerator choice | Pick GPU T4 x2. A P100 works but is slower, and compile is switched off on it automatically |
| Loss is `nan` | Unstable training | Lower the learning rate: `args += ["--lr", 6e-4]` and start fresh |
| Loss stays near 9 | Data or learning rate problem | Check `meta.json` from notebook 01. Make sure `--lr` was not set to a tiny value |
| Loss falls, then jumps up | Learning rate too high | `--lr 6e-4`, or resume from the checkpoint before the jump with a lower `--lr` |
| Session stopped at 12 hours | Kaggle time limit | [Resume](#resuming-after-a-session-ends) |
| Weekly GPU quota used up | Kaggle quota | Wait for the weekly reset; resume afterwards |
| No files in Output | The run was interactive, not committed | Only **Save & Run All (Commit)** versions keep their output |
| `--resume` fails with `size mismatch` | Checkpoint from a different model size | Resume only from checkpoints of the same run |

### Chat tuning and export (notebook 03)

| Symptom | Cause | Fix |
|---|---|---|
| Errors downloading `yahma/alpaca-cleaned` or `everyday-conversations` | Internet is Off or Hugging Face is busy | Turn Internet On and rerun |
| Replies never stop and run to 200 tokens | Tokenizer mismatch or too little tuning | Use the `tokenizer.json` copied by notebook 02 (the default). Try `--epochs 3` |
| Replies are Wikipedia fragments, not answers | Too little chat tuning | Rerun with `--epochs 3`. Check the chat-tuning val loss is falling |
| Gibberish replies | Pretraining too short or failed | Check notebook 02's `loss.png` and samples |
| Replies repeat the same phrase | Small-model looping | In the app raise `WIKIGPT_REPETITION_PENALTY` to 1.2 or 1.3, or lower `WIKIGPT_TEMPERATURE` |
| `export check failed` | Exported weights did not reload identically | Rerun the export cell. If it persists, make sure code and checkpoint come from the same version |

### Local app

| Symptom | Cause | Fix |
|---|---|---|
| `Model bundle not found in ...\models\wikigpt-12m` | Model not extracted, or extracted one folder too deep | Expected path: `models\wikigpt-12m\model.pt`. If you see `models\wikigpt-12m\wikigpt-12m\`, move the files up one level |
| `'python' is not recognized` | Python not on PATH | Use `py` instead of `python`, or reinstall Python with "Add to PATH" ticked |
| Activation script blocked | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `ModuleNotFoundError: No module named 'torch'` | torch not installed in the active environment | Activate `.venv`, then step E2 |
| `ModuleNotFoundError: No module named 'wikigpt'` or `'app'` | Started from the wrong folder | `cd` to the project folder, then `python -m app` |
| `torch.cuda.is_available()` is `False` with an NVIDIA GPU | CPU build of torch, or old driver | Reinstall torch with the CUDA command from E2. Update the NVIDIA driver. Or use the CPU, which works fine |
| `WIKIGPT_DEVICE is cuda, but this torch install cannot see a GPU` | Same as above | Same fix, or set `WIKIGPT_DEVICE` to `auto` |
| `size mismatch` or `Error(s) in loading state_dict` | Model bundle and code from different versions | Export the bundle again with the current code, or use matching code |
| Port already in use (error 10048 on Windows) | Another program uses port 8000 | `$env:WIKIGPT_PORT = "8080"`, then open http://127.0.0.1:8080 |
| Page loads but stays empty, or shows 404 | Old browser cache or wrong folder | Hard refresh with Ctrl+F5. Start the app from the project folder |
| "Could not get a reply" in the chat | Server error | Read the error in the terminal running the app. Nothing is saved for a failed reply, so just send the message again |
| `WIKIGPT_MAX_NEW_TOKENS must be between 1 and 256` | Setting too large for the 512-token context | Use a value up to 256 |
| "No reply generated." shown in grey | The model ended its reply immediately | Send the message again or rephrase; sampling differs each time |
| Replies are slow | Running on CPU | Normal on CPU (a few seconds). Lower `WIKIGPT_MAX_NEW_TOKENS`, or use the GPU |
| Want to clear all chats | | Stop the app and delete `app\data\chats.db` |

---

## 13. FAQ

**Why a custom 8,192-token tokenizer instead of GPT-2's?**
GPT-2's vocabulary has 50,257 tokens. At width 320 its embedding table alone would be 16M parameters, more than the whole model budget. An 8,192-token vocabulary trained on Wikipedia costs 2.6M parameters and still encodes English efficiently, at roughly 4 characters per token.

**Why are the answers often wrong?**
Facts take capacity. A 12M parameter model learns spelling, grammar and the style of Wikipedia long before it can store many facts. Chat tuning teaches the format of an answer, not new knowledge.

**How do I train longer or train a bigger model?**
- **More data:** pass `"--target-tokens", 1_000_000_000` to notebook 01, then raise `--max-steps` in notebook 02 to match. Steps = tokens / 131,072.
- **Bigger model:** pass model flags in notebook 02, for example `--d-model 384 --n-heads 6 --n-layers 10`. `d_model` must be divisible by `n_heads`, and the parameter count is printed at the start. The app reads `config.json`, so any size you export works without code changes.
- **Watch the budget:** bigger models need more tokens and more GPU time. Keep runs under the weekly quota.

**Does it use both T4 GPUs?**
No. One GPU is plenty for 12M parameters, and a single-GPU loop keeps the code simple.

**Can this run on a phone or a Raspberry Pi?**
The weights are only 25 MB, so memory is not the problem. A Raspberry Pi 4 or 5 can run the app with the CPU build of PyTorch, though replies will be slow. Phones would need the model converted to a mobile runtime, which this project does not cover.

**Why do replies take longer on the CPU?**
Generation re-reads the whole conversation for every new token. That keeps the code short, and on a GPU it is fast. A key-value cache would speed up CPU generation and is a natural next step.

---

## 14. References and licenses

**Learning material**
- Playlist: Build a Large Language Model (from Scratch), chapter by chapter: https://youtube.com/playlist?list=PLQRyiBCWmqp5twpd8Izmaxu5XRkxd5yC-
- Playlist: Build an LLM From Scratch (course): https://youtube.com/playlist?list=PLBwE3Q2KZ4kM
- Playlist: Building LLMs from scratch: https://youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu
- Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning, 2024. Code: https://github.com/rasbt/LLMs-from-scratch
- Andrej Karpathy, nanoGPT: https://github.com/karpathy/nanoGPT

**Data**
- Wikipedia text via `wikimedia/wikipedia` on Hugging Face. Wikipedia content is licensed CC BY-SA 4.0 (and GFDL).
- `yahma/alpaca-cleaned`. The dataset card lists CC BY 4.0 in its metadata, but its licensing section states CC BY-NC 4.0 (non-commercial), and the data was generated with an OpenAI model. Treat it as non-commercial.
- `HuggingFaceTB/everyday-conversations-llama3.1-2k`, Apache 2.0.

A model trained on this data carries those terms with it. Keep it a personal, educational project unless you have checked that your use is allowed.
