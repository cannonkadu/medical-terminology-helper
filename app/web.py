"""Zero-dependency local web studio for Medical Terminology Helper.

Launch with: ``python -m app.cli web`` (opens on http://127.0.0.1:8000).
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse

from app.core import (
    DEFAULT_CHECKPOINT_PATH,
    TrainingConfig,
    build_quiz_question,
    corpus_stats,
    ensure_trained_model,
    generate_completion,
    load_checkpoint,
    search_corpus,
    train_model,
)

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Medical Terminology Helper — GPT from Scratch</title>
<style>
  :root {
    --bg-0: #0a0d16;
    --bg-1: #111626;
    --bg-2: #171e31;
    --border: rgba(148, 163, 184, 0.16);
    --text: #f1f5f9;
    --muted: #94a3b8;
    --accent: #7c5cff;
    --accent-2: #2dd4bf;
    --accent-glow: rgba(124, 92, 255, 0.35);
    --ok: #34d399;
    --warn: #fbbf24;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
    color: var(--text);
    background:
      radial-gradient(900px 500px at 85% 10%, rgba(45, 212, 191, 0.12), transparent 60%),
      radial-gradient(800px 500px at 15% 0%, rgba(124, 92, 255, 0.18), transparent 60%),
      var(--bg-0);
    min-height: 100vh;
    line-height: 1.5;
  }
  .container {
    max-width: 1120px;
    margin: 0 auto;
    padding: 28px 20px 64px;
  }
  header {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 24px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
  }
  .eyebrow {
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent-2);
    font-weight: 600;
    margin-bottom: 6px;
  }
  h1 {
    margin: 0 0 8px;
    font-size: clamp(24px, 3.2vw, 34px);
    letter-spacing: -0.02em;
  }
  .subtitle {
    margin: 0;
    color: var(--muted);
    max-width: 680px;
    font-size: 14.5px;
  }
  .badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
  }
  .badge {
    font-size: 12px;
    padding: 4px 12px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.04);
    color: #cbd5e1;
  }
  .badge.primary {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
    font-weight: 600;
  }
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    border-radius: 12px;
    background: var(--bg-1);
    border: 1px solid var(--border);
    font-size: 13px;
    color: #cbd5e1;
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--ok);
    box-shadow: 0 0 10px var(--ok);
  }
  .tabs {
    display: flex;
    gap: 6px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 22px;
    overflow-x: auto;
  }
  .tab {
    background: none;
    border: none;
    color: var(--muted);
    padding: 10px 16px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: color 0.15s, border-color 0.15s;
    white-space: nowrap;
  }
  .tab:hover { color: var(--text); }
  .tab.active {
    color: var(--text);
    border-bottom-color: var(--accent);
  }
  .panel { display: none; }
  .panel.active { display: block; }
  .grid-2 {
    display: grid;
    grid-template-columns: 1.35fr 1fr;
    gap: 18px;
  }
  @media (max-width: 860px) {
    .grid-2 { grid-template-columns: 1fr; }
  }
  .card {
    background: rgba(17, 22, 38, 0.85);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    backdrop-filter: blur(8px);
  }
  .card h2 {
    margin: 0 0 12px;
    font-size: 16px;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 8px;
    display: block;
  }
  textarea, input[type="text"], input[type="number"] {
    width: 100%;
    background: var(--bg-0);
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--text);
    padding: 12px 14px;
    font-size: 15px;
    font-family: inherit;
    outline: none;
    transition: border-color 0.15s;
  }
  textarea:focus, input:focus {
    border-color: var(--accent);
  }
  .presets {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 12px 0 16px;
  }
  .chip {
    background: var(--bg-2);
    border: 1px solid var(--border);
    color: #cbd5e1;
    border-radius: 999px;
    padding: 5px 12px;
    font-size: 12.5px;
    cursor: pointer;
    transition: all 0.15s;
  }
  .chip:hover {
    border-color: var(--accent-2);
    color: #fff;
  }
  .controls {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }
  @media (max-width: 600px) {
    .controls { grid-template-columns: 1fr; }
  }
  .slider-box {
    background: var(--bg-0);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 12px;
  }
  .slider-head {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .slider-head strong { color: var(--accent-2); }
  input[type="range"] {
    width: 100%;
    accent-color: var(--accent);
  }
  .btn {
    background: linear-gradient(135deg, var(--accent), #5b3df5);
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 11px 18px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    box-shadow: 0 6px 20px var(--accent-glow);
    transition: transform 0.1s, opacity 0.15s;
  }
  .btn:hover { opacity: 0.94; }
  .btn:active { transform: translateY(1px); }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn.secondary {
    background: var(--bg-2);
    border: 1px solid var(--border);
    box-shadow: none;
  }
  .output-box {
    margin-top: 16px;
    background: var(--bg-0);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    min-height: 86px;
    font-size: 16px;
    line-height: 1.65;
  }
  .out-prompt { color: #e2e8f0; font-weight: 600; }
  .out-gen { color: var(--accent-2); }
  .meta-line {
    margin-top: 10px;
    font-size: 12px;
    color: var(--muted);
  }
  .tokens-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
  }
  .tok {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 8px;
    border-radius: 7px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12.5px;
    border: 1px solid rgba(255, 255, 255, 0.1);
  }
  .tok .tid {
    font-size: 10px;
    opacity: 0.7;
  }
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin-bottom: 18px;
  }
  .stat {
    background: var(--bg-0);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
  }
  .stat .k { font-size: 11.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
  .stat .v { font-size: 22px; font-weight: 700; margin-top: 4px; color: var(--text); }
  .ref-item {
    padding: 12px;
    border-radius: 10px;
    background: var(--bg-0);
    border: 1px solid var(--border);
    margin-bottom: 10px;
    font-size: 13.5px;
  }
  .ref-cat {
    font-size: 11px;
    color: var(--accent-2);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
  }
  svg.chart {
    width: 100%;
    height: 220px;
    background: var(--bg-0);
    border: 1px solid var(--border);
    border-radius: 12px;
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <div class="eyebrow">Build GPT From Scratch &bull; PyTorch + BPE</div>
      <h1>Medical Terminology Helper</h1>
      <p class="subtitle">
        Complete medical phrases and definitions using a decoder-only GPT trained from scratch on public-domain anatomy, physiology, pathology, and pharmacology texts.
      </p>
      <div class="badges">
        <span class="badge primary">Healthcare</span>
        <span class="badge">Runs on your laptop</span>
        <span class="badge">Byte-Pair Tokenizer</span>
        <span class="badge">Causal Self-Attention</span>
      </div>
    </div>
    <div class="status-pill" id="modelStatus">
      <span class="dot"></span>
      <span>Loading model status...</span>
    </div>
  </header>

  <nav class="tabs">
    <button class="tab active" data-tab="complete">Phrase Completion</button>
    <button class="tab" data-tab="tokenizer">BPE Tokenizer</button>
    <button class="tab" data-tab="quiz">Study Quiz</button>
    <button class="tab" data-tab="lookup">Corpus Lookup</button>
    <button class="tab" data-tab="training">Architecture &amp; Training</button>
  </nav>

  <!-- Tab 1: Completion -->
  <section id="tab-complete" class="panel active">
    <div class="grid-2">
      <div class="card">
        <h2>Complete a Medical Phrase</h2>
        <label class="label" for="promptInput">Partial medical term or definition</label>
        <textarea id="promptInput" rows="2" placeholder="e.g., Myocardium is ">Myocardium is </textarea>

        <div class="presets">
          <button class="chip" data-prompt="Myocardium is ">Myocardium is </button>
          <button class="chip" data-prompt="Alveoli are ">Alveoli are </button>
          <button class="chip" data-prompt="Bradycardia is ">Bradycardia is </button>
          <button class="chip" data-prompt="Nephron is ">Nephron is </button>
          <button class="chip" data-prompt="The suffix itis ">The suffix itis </button>
          <button class="chip" data-prompt="Homeostasis is ">Homeostasis is </button>
        </div>

        <div class="controls">
          <div class="slider-box">
            <div class="slider-head"><span>Temperature</span><strong id="tempVal">0.6</strong></div>
            <input id="tempSlider" type="range" min="0" max="1.4" step="0.1" value="0.6" />
          </div>
          <div class="slider-box">
            <div class="slider-head"><span>Top-K</span><strong id="topkVal">20</strong></div>
            <input id="topkSlider" type="range" min="1" max="60" step="1" value="20" />
          </div>
          <div class="slider-box">
            <div class="slider-head"><span>Max Tokens</span><strong id="maxTokVal">32</strong></div>
            <input id="maxTokSlider" type="range" min="8" max="64" step="4" value="32" />
          </div>
        </div>

        <button class="btn" id="completeBtn">Generate Completion</button>

        <div class="output-box" id="completionOutput">
          <span style="color: var(--muted)">Click "Generate Completion" to run autoregressive decoding with MedicalGPT.</span>
        </div>
        <div class="meta-line" id="completionMeta"></div>
      </div>

      <div class="card">
        <h2>Related Corpus References</h2>
        <div id="relatedRefs">
          <p style="color: var(--muted); font-size: 13.5px;">Matching public-domain definitions from the medical corpus will appear here for verification.</p>
        </div>
      </div>
    </div>
  </section>

  <!-- Tab 2: Tokenizer -->
  <section id="tab-tokenizer" class="panel">
    <div class="card">
      <h2>Byte-Pair Encoding (BPE) Inspector</h2>
      <label class="label" for="tokInput">Inspect how MedicalGPT splits medical terms into learned subwords</label>
      <input id="tokInput" type="text" value="Myocardial infarction is the irreversible necrosis of heart muscle tissue." />
      <div class="meta-line" id="tokStats"></div>
      <div class="tokens-wrap" id="tokPills"></div>
    </div>
  </section>

  <!-- Tab 3: Study Quiz -->
  <section id="tab-quiz" class="panel">
    <div class="card">
      <h2>Interactive Terminology Quiz <button class="btn secondary" id="newQuizBtn">New Question</button></h2>
      <div id="quizCategory" class="ref-cat"></div>
      <div id="quizPrompt" style="font-size: 20px; font-weight: 600; margin: 8px 0 14px;"></div>
      <input id="quizGuess" type="text" placeholder="Type your definition completion from memory..." />
      <div style="margin-top: 12px; display: flex; gap: 10px;">
        <button class="btn" id="revealQuizBtn">Compare with MedicalGPT &amp; Reference</button>
      </div>
      <div id="quizReveal" style="margin-top: 16px; display: none;">
        <div class="ref-item">
          <div class="ref-cat">MedicalGPT Prediction</div>
          <div id="quizGptOut"></div>
        </div>
        <div class="ref-item">
          <div class="ref-cat">Corpus Reference Definition</div>
          <div id="quizRefOut"></div>
        </div>
      </div>
    </div>
  </section>

  <!-- Tab 4: Corpus Lookup -->
  <section id="tab-lookup" class="panel">
    <div class="card">
      <h2>Search Public-Domain Medical Corpus</h2>
      <div style="display: flex; gap: 10px; margin-bottom: 16px;">
        <input id="lookupInput" type="text" placeholder="Search by term, organ, or prefix (e.g., cardio, neuron, itis, nephron)..." />
        <button class="btn" id="lookupBtn">Search</button>
      </div>
      <div id="lookupResults"></div>
    </div>
  </section>

  <!-- Tab 5: Architecture & Training -->
  <section id="tab-training" class="panel">
    <div class="stats-grid" id="archStats"></div>
    <div class="card">
      <h2>
        <span>Training &amp; Validation Loss Curve</span>
        <div style="display: flex; align-items: center; gap: 8px;">
          <input id="epochsInput" type="number" min="5" max="80" value="25" style="width: 82px; padding: 6px 10px; font-size: 13px;" />
          <button class="btn" id="retrainBtn">Train From Scratch</button>
        </div>
      </h2>
      <svg class="chart" id="lossChart" viewBox="0 0 700 220"></svg>
      <div class="meta-line" id="trainNote"></div>
    </div>
  </section>
</div>

<script>
const $ = (id) => document.getElementById(id);
const palette = [
  "rgba(124,92,255,0.22)", "rgba(45,212,191,0.20)", "rgba(56,189,248,0.20)",
  "rgba(244,114,182,0.20)", "rgba(251,191,36,0.18)", "rgba(52,211,153,0.20)"
];

document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    $("tab-" + btn.dataset.tab).classList.add("active");
  });
});

document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    $("promptInput").value = chip.dataset.prompt;
    runComplete();
  });
});

$("tempSlider").oninput = e => $("tempVal").textContent = e.target.value;
$("topkSlider").oninput = e => $("topkVal").textContent = e.target.value;
$("maxTokSlider").oninput = e => $("maxTokVal").textContent = e.target.value;

async function fetchStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  $("modelStatus").innerHTML = `<span class="dot"></span><span>${data.num_parameters.toLocaleString()} params &bull; ${data.vocab_size} BPE tokens</span>`;

  $("archStats").innerHTML = `
    <div class="stat"><div class="k">Parameters</div><div class="v">${data.num_parameters.toLocaleString()}</div></div>
    <div class="stat"><div class="k">Vocab Size</div><div class="v">${data.vocab_size}</div></div>
    <div class="stat"><div class="k">Layers / Heads</div><div class="v">${data.config.n_layers}L / ${data.config.n_heads}H</div></div>
    <div class="stat"><div class="k">Embedding Dim</div><div class="v">${data.config.d_model}</div></div>
    <div class="stat"><div class="k">Corpus Entries</div><div class="v">${data.corpus.num_entries}</div></div>
    <div class="stat"><div class="k">Final Loss</div><div class="v">${data.history.length ? data.history[data.history.length - 1].train_loss : "—"}</div></div>
  `;
  drawChart(data.history);
}

function drawChart(history) {
  const svg = $("lossChart");
  if (!history || history.length < 2) {
    svg.innerHTML = `<text x="350" y="115" fill="#94a3b8" text-anchor="middle" font-size="14">No training history yet. Click "Train From Scratch".</text>`;
    return;
  }
  const W = 700, H = 220, pad = 34;
  const maxL = Math.max(...history.map(h => Math.max(h.train_loss, h.val_loss)));
  const minL = Math.min(...history.map(h => Math.min(h.train_loss, h.val_loss))) * 0.9;
  const x = i => pad + (i / (history.length - 1)) * (W - 2 * pad);
  const y = v => H - pad - ((v - minL) / Math.max(0.01, maxL - minL)) * (H - 2 * pad);
  const trainPts = history.map((h, i) => `${x(i).toFixed(1)},${y(h.train_loss).toFixed(1)}`).join(" ");
  const valPts = history.map((h, i) => `${x(i).toFixed(1)},${y(h.val_loss).toFixed(1)}`).join(" ");
  svg.innerHTML = `
    <line x1="${pad}" y1="${H-pad}" x2="${W-pad}" y2="${H-pad}" stroke="rgba(148,163,184,0.2)" />
    <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H-pad}" stroke="rgba(148,163,184,0.2)" />
    <polyline fill="none" stroke="#7c5cff" stroke-width="2.5" points="${trainPts}" />
    <polyline fill="none" stroke="#2dd4bf" stroke-width="2" stroke-dasharray="5,4" points="${valPts}" />
    <text x="${W-pad}" y="20" fill="#7c5cff" text-anchor="end" font-size="12">Train Loss (${history[history.length-1].train_loss})</text>
    <text x="${W-pad-140}" y="20" fill="#2dd4bf" text-anchor="end" font-size="12">Val Loss (${history[history.length-1].val_loss})</text>
    <text x="${pad}" y="${H-10}" fill="#94a3b8" font-size="11">Epoch 1</text>
    <text x="${W-pad}" y="${H-10}" fill="#94a3b8" text-anchor="end" font-size="11">Epoch ${history.length}</text>
  `;
}

async function runComplete() {
  const btn = $("completeBtn");
  btn.disabled = true;
  btn.textContent = "Generating...";
  try {
    const res = await fetch("/api/complete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: $("promptInput").value,
        temperature: parseFloat($("tempSlider").value),
        top_k: parseInt($("topkSlider").value, 10),
        max_new_tokens: parseInt($("maxTokSlider").value, 10),
      }),
    });
    const data = await res.json();
    $("completionOutput").innerHTML = `<span class="out-prompt">${escapeHtml(data.prompt)}</span><span class="out-gen">${escapeHtml(data.completion)}</span>`;
    $("completionMeta").textContent = `Prompt tokens: ${data.prompt_tokens} | Generated tokens: ${data.generated_tokens}`;
    renderRefs(data.related || []);
  } finally {
    btn.disabled = false;
    btn.textContent = "Generate Completion";
  }
}

function renderRefs(items) {
  if (!items.length) {
    $("relatedRefs").innerHTML = `<p style="color: var(--muted); font-size: 13.5px;">No exact corpus match for the leading term.</p>`;
    return;
  }
  $("relatedRefs").innerHTML = items.map(m => `
    <div class="ref-item">
      <div class="ref-cat">${escapeHtml(m.category)}</div>
      <div>${escapeHtml(m.text)}</div>
    </div>
  `).join("");
}

async function runTokenize() {
  const text = $("tokInput").value;
  const res = await fetch("/api/tokenize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const data = await res.json();
  $("tokStats").textContent = `${data.tokens.length} BPE tokens from ${data.num_bytes} UTF-8 bytes (${data.compression_ratio}x compression)`;
  $("tokPills").innerHTML = data.tokens.map((t, i) => `
    <span class="tok" style="background:${palette[i % palette.length]}">
      <span>${escapeHtml(t.piece.replace(/ /g, "·"))}</span>
      <span class="tid">#${t.id}</span>
    </span>
  `).join("");
}

let currentQuiz = null;
async function loadQuiz() {
  const res = await fetch("/api/quiz");
  currentQuiz = await res.json();
  $("quizCategory").textContent = currentQuiz.category;
  $("quizPrompt").textContent = currentQuiz.prompt + " ___";
  $("quizGuess").value = "";
  $("quizReveal").style.display = "none";
}

$("revealQuizBtn").addEventListener("click", () => {
  if (!currentQuiz) return;
  $("quizGptOut").textContent = currentQuiz.gpt_completion;
  $("quizRefOut").textContent = currentQuiz.full_text;
  $("quizReveal").style.display = "block";
});

async function runLookup() {
  const q = $("lookupInput").value.trim();
  const res = await fetch("/api/lookup?q=" + encodeURIComponent(q));
  const data = await res.json();
  if (!data.matches.length) {
    $("lookupResults").innerHTML = `<p style="color: var(--muted)">No matching medical entries found.</p>`;
    return;
  }
  $("lookupResults").innerHTML = data.matches.map(m => `
    <div class="ref-item">
      <div class="ref-cat">${escapeHtml(m.category)}</div>
      <div>${escapeHtml(m.text)}</div>
    </div>
  `).join("");
}

$("retrainBtn").addEventListener("click", async () => {
  const btn = $("retrainBtn");
  btn.disabled = true;
  btn.textContent = "Training...";
  $("trainNote").textContent = "Training MedicalGPT from scratch on CPU...";
  try {
    const epochs = parseInt($("epochsInput").value, 10) || 25;
    const res = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs }),
    });
    const data = await res.json();
    $("trainNote").textContent = `Trained ${epochs} epochs in ${data.elapsed_seconds}s | loss ${data.initial_loss} -> ${data.final_loss} | perplexity ${data.final_perplexity}`;
    await fetchStatus();
  } finally {
    btn.disabled = false;
    btn.textContent = "Train From Scratch";
  }
});

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c]));
}

$("completeBtn").addEventListener("click", runComplete);
$("tokInput").addEventListener("input", runTokenize);
$("newQuizBtn").addEventListener("click", loadQuiz);
$("lookupBtn").addEventListener("click", runLookup);
$("lookupInput").addEventListener("keydown", e => { if (e.key === "Enter") runLookup(); });

fetchStatus().then(() => { runTokenize(); loadQuiz(); });
</script>
</body>
</html>
"""


class _StudioState:
    def __init__(self, checkpoint_path: str | Path):
        self.checkpoint_path = Path(checkpoint_path)
        self.lock = threading.Lock()
        self.model, self.tokenizer, self.history = ensure_trained_model(
            checkpoint_path=self.checkpoint_path,
            config=TrainingConfig(epochs=20),
        )


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
) -> ThreadingHTTPServer:
    state = _StudioState(checkpoint_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode("utf-8") or "{}")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/api/status":
                with state.lock:
                    self._send_json({
                        "num_parameters": state.model.num_parameters(),
                        "vocab_size": state.tokenizer.vocab_size,
                        "config": state.model.config.to_dict(),
                        "history": state.history,
                        "corpus": corpus_stats(),
                    })
                return

            if parsed.path == "/api/lookup":
                q = parse_qs(parsed.query).get("q", [""])[0]
                self._send_json({"matches": search_corpus(q, limit=10)})
                return

            if parsed.path == "/api/quiz":
                q = build_quiz_question()
                with state.lock:
                    out = generate_completion(
                        state.model,
                        state.tokenizer,
                        q["prompt"],
                        max_new_tokens=28,
                        temperature=0.4,
                    )
                q["gpt_completion"] = out["full_text"]
                self._send_json(q)
                return

            self._send_json({"error": "Not found"}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            data = self._read_json()

            if parsed.path == "/api/complete":
                prompt = str(data.get("prompt", ""))
                with state.lock:
                    out = generate_completion(
                        model=state.model,
                        tokenizer=state.tokenizer,
                        prompt=prompt,
                        max_new_tokens=int(data.get("max_new_tokens", 32)),
                        temperature=float(data.get("temperature", 0.6)),
                        top_k=int(data.get("top_k", 20)),
                        top_p=float(data.get("top_p", 0.9)),
                    )
                first_word = prompt.split()[0] if prompt.strip() else ""
                out["related"] = search_corpus(first_word, limit=3) if first_word else []
                self._send_json(out)
                return

            if parsed.path == "/api/tokenize":
                text = str(data.get("text", ""))
                with state.lock:
                    tokens = state.tokenizer.tokenize_with_pieces(text)
                num_bytes = len(text.encode("utf-8"))
                self._send_json({
                    "tokens": tokens,
                    "num_bytes": num_bytes,
                    "compression_ratio": round(num_bytes / max(1, len(tokens)), 2),
                })
                return

            if parsed.path == "/api/train":
                epochs = max(5, min(80, int(data.get("epochs", 25))))
                with state.lock:
                    result = train_model(
                        config=TrainingConfig(epochs=epochs),
                        checkpoint_path=state.checkpoint_path,
                    )
                    state.model = result["model"]
                    state.tokenizer = result["tokenizer"]
                    state.history = result["history"]
                self._send_json({
                    "initial_loss": result["initial_loss"],
                    "final_loss": result["final_loss"],
                    "final_perplexity": result["final_perplexity"],
                    "elapsed_seconds": result["elapsed_seconds"],
                })
                return

            self._send_json({"error": "Not found"}, status=404)

    return ThreadingHTTPServer((host, port), Handler)


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
) -> None:
    server = create_server(host=host, port=port, checkpoint_path=checkpoint_path)
    print(f"Medical Terminology Helper Studio running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()
