import math
import io
from pypdf import PdfReader
import os
import re
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
import nltk
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine as scipy_cosine
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import torch

load_dotenv()
API_KEY = os.getenv("API_KEY")

# --- NLTK SETUP ---
for pkg, path in [
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

STOPWORDS = set(stopwords.words("english"))

# --- PAGE CONFIG ---
st.set_page_config(page_title="Claim Matching & Validation Tool", layout="wide")
st.title("Document Claim Matching and Validation Tool")

st.markdown("""
**System Architecture:**
1. **Local NLP Layer (NLTK + SciPy):** Offline claim extraction, vector embeddings (`all-MiniLM-L6-v2`), and lexical Jaccard scoring against source text chunks.
2. **AI Verification Layer (Gemini):** Single pass contextual verification over the full document to evaluate semantic equivalence and mathematical validity.
""")

# --- SIDEBAR ---
st.sidebar.header("Settings")
gemini_api_key = st.sidebar.text_input(
    "Gemini API Key:", value=API_KEY if API_KEY else "", type="password"
)
model_name = st.sidebar.text_input("Model Name:", value="gemini-3.1-flash-lite")
max_rpm = st.sidebar.slider("Rate Limit (Max RPM)", 3, 15, 10)


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device=torch.device("cpu"))


with st.spinner("Initializing local embedding engine..."):
    embedder = load_embedder()

# --- STATE INITIALIZATION ---
for key in [
    "ocr_text",
    "generated_note_val",
    "existing_note_val",
    "call_timestamps",
    "claims",
    "local_results",
    "ai_results",
]:
    if key not in st.session_state:
        st.session_state[key] = (
            ""
            if "val" in key or "text" in key
            else []
            if key == "call_timestamps"
            else None
        )


# --- RATE LIMITER & API WRAPPER ---
def enforce_rate_limit(cap):
    now = time.time()
    st.session_state.call_timestamps = [
        t for t in st.session_state.call_timestamps if now - t < 60
    ]
    if len(st.session_state.call_timestamps) >= cap:
        wait = max(0, 60 - (now - st.session_state.call_timestamps[0])) + 0.5
        if wait > 0:
            st.info(f"Rate limit reached. Waiting {wait:.1f}s...")
            time.sleep(wait)
        now = time.time()
        st.session_state.call_timestamps = [
            t for t in st.session_state.call_timestamps if now - t < 60
        ]
    st.session_state.call_timestamps.append(time.time())


def call_gemini(client, model_name, contents, max_retries=3):
    for attempt in range(max_retries):
        enforce_rate_limit(max_rpm)
        try:
            return client.models.generate_content(model=model_name, contents=contents)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = 15 * (attempt + 1)
                st.warning(f"Rate limit hit. Retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Exceeded maximum retries due to rate limit errors.")


def extract_text_locally(file_bytes: bytes) -> str:
    """Extracts raw selectable text from a digital PDF without using any API."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        extracted = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted.append(text)
        return "\n".join(extracted).strip()
    except Exception:
        return ""


# --- UTILITY: NLP EXTRACTION & CLEANING ---
LIST_ITEM_RE = re.compile(
    r"(?m)^[ \t]*(?:\d+[.)]|[-*])\s+(.+(?:\n(?![ \t]*(?:\d+[.)]|[-*])\s+)(?!\s*$)[ \t]*\S.*)*)"
)


def is_noise_line(line: str) -> bool:
    s = line.strip()
    if re.match(r"^==.*OCR.*==$", s, re.IGNORECASE):
        return True
    if len(s) < 8 or s.startswith("#") or re.fullmatch(r"[\d.\-#*\s]+", s):
        return True
    if len(re.findall(r"[A-Za-z]", s)) < 6:
        return True
    if re.match(
        r"^(here is|here's|below is|sure[,!]?\s*here|certainly)",
        s,
        re.IGNORECASE,
    ):
        return True
    return False


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip().replace("**", "")
    s = re.sub(r"^[\d]+[.)]\s*", "", s)
    return s.strip()


def extract_claims(note_text: str):
    claims = []
    for block in re.split(r"\n\s*\n", note_text):
        block = block.strip()
        if not block:
            continue
        items = list(LIST_ITEM_RE.finditer(block))
        if items:
            for m in items:
                t = clean_text(m.group(1))
                if not is_noise_line(t):
                    claims.append(t)
        else:
            for line in block.split("\n"):
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                for s in nltk.sent_tokenize(line):
                    t = clean_text(s)
                    if not is_noise_line(t):
                        claims.append(t)
    return claims


def extract_source_chunks(src_text: str):
    chunks = []
    for block in re.split(r"\n\s*\n", src_text):
        lines = [ln.strip() for ln in block.strip().split("\n")]
        clean_lines = [ln for ln in lines if not is_noise_line(ln)]
        chunks.extend(clean_lines)
        for i in range(len(clean_lines) - 1):
            merged = clean_lines[i] + " " + clean_lines[i + 1]
            if merged not in chunks:
                chunks.append(merged)
    return chunks if chunks else [src_text.strip()]


def lexical_jaccard(a: str, b: str) -> float:
    wa = {
        w.lower()
        for w in nltk.word_tokenize(a)
        if w.isalpha() and w.lower() not in STOPWORDS
    }
    wb = {
        w.lower()
        for w in nltk.word_tokenize(b)
        if w.isalpha() and w.lower() not in STOPWORDS
    }
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb) * 100.0


def clamp_pct(val: float) -> float:
    if math.isnan(val):
        return 0.0
    return min(max(float(val) / 100.0, 0.0), 1.0)


# ==========================================
# STEP 1: SOURCE TEXT EXTRACTION
# ==========================================
st.subheader("1. Source Text Extraction")
uploaded_file = st.file_uploader(
    "Upload Source Image or PDF", type=["png", "jpg", "jpeg", "pdf"]
)

if uploaded_file and st.button("Extract Text"):
    file_bytes = uploaded_file.read()
    is_pdf = uploaded_file.type == "application/pdf"
    local_text = extract_text_locally(file_bytes) if is_pdf else ""

    # Case A: Digital PDF with direct selectable text (> 40 chars)
    if is_pdf and len(local_text) > 40:
        st.session_state.ocr_text = local_text
        st.success("Extracted native text directly from PDF (0 API calls used).")

    # Case B: Images or Scanned PDFs with no selectable text -> Use Gemini OCR
    else:
        if not gemini_api_key:
            st.error(
                "This document requires OCR. Please enter your Gemini API Key in the sidebar."
            )
        else:
            try:
                client = genai.Client(api_key=gemini_api_key)
                ocr_prompt = (
                    "Perform exact OCR on this document. Preserve math equations clearly.\n"
                    "Every LaTeX symbol/formula must be wrapped in $...$ (inline) or $$...$$ (display).\n"
                    "Never output bare LaTeX commands outside delimiters.\n"
                    "CRITICAL: Do NOT generate page markers, headers, or footers (e.g., no '==Start of OCR=='). "
                    "Output strictly the document content."
                )
                media_part = types.Part.from_bytes(
                    data=file_bytes, mime_type=uploaded_file.type
                )
                with st.spinner(
                    "Scanned document detected. Performing OCR via Gemini..."
                ):
                    response = call_gemini(client, model_name, [media_part, ocr_prompt])
                    st.session_state.ocr_text = response.text
                    st.success("OCR completed via Gemini.")
            except Exception as e:
                st.error(f"Extraction Error: {str(e)}")

col_s1, col_s2 = st.columns(2)
with col_s1:
    source_text = st.text_area(
        "Source Ground Truth Text:", value=st.session_state.ocr_text, height=200
    )
with col_s2:
    with st.expander("Formatted Text Preview", expanded=False):
        st.markdown(source_text if source_text.strip() else "No text loaded.")

st.divider()

# ==========================================
# STEP 2: TARGET NOTE PREPARATION
# ==========================================
st.subheader("2. Target Note Preparation")
col_n1, col_n2 = st.columns(2)
with col_n1:
    existing_unit_note = st.text_area(
        "Existing Baseline Note (Optional):",
        value=st.session_state.existing_note_val,
        height=120,
    )
    st.session_state.existing_note_val = existing_unit_note
    if st.button("Generate Merged Note"):
        if not gemini_api_key or not source_text.strip():
            st.error("API key and source text are required.")
        else:
            try:
                client = genai.Client(api_key=gemini_api_key)
                merge_prompt = f"Synthesize study notes from:\nExisting: {existing_unit_note}\nSource: {source_text}\nWrite clean Markdown. Wrap LaTeX in $...$. Output only the content without conversational preamble."
                with st.spinner("Synthesizing note..."):
                    res = call_gemini(client, model_name, merge_prompt)
                    st.session_state.generated_note_val = (
                        res.text if res and res.text else ""
                    )
            except Exception as e:
                st.error(f"Synthesis Error: {str(e)}")
with col_n2:
    generated_note = st.text_area(
        "Target Note to Verify:",
        value=st.session_state.generated_note_val,
        height=200,
    )

st.divider()

# ==========================================
# STEP 3: CLAIM MATCHING & VERIFICATION
# ==========================================
st.subheader("3. Claim Matching & Verification")
st.markdown(
    "Extracts claims locally, then runs local vector similarity and contextual AI verification across the same claim set."
)

col_v1, col_v2 = st.columns([0.5, 0.5])

with col_v1:
    if st.button("1. Run Local Similarity Search", type="primary"):
        if not source_text.strip() or not generated_note.strip():
            st.error("Source text and target note are both required.")
        else:
            source_chunks = extract_source_chunks(source_text)
            claims = extract_claims(generated_note)

            if not claims:
                st.warning("No valid claims found in the target note.")
            else:
                with st.spinner(f"Encoding {len(claims)} claims..."):
                    source_emb = embedder.encode(source_chunks, convert_to_tensor=True)
                    claim_emb = embedder.encode(claims, convert_to_tensor=True)
                    cosine_scores = util.cos_sim(claim_emb, source_emb)

                    local_results = []
                    for i, claim in enumerate(claims):
                        k = min(3, len(source_chunks))
                        top_scores, top_idxs = torch.topk(cosine_scores[i], k=k)
                        candidates = []
                        for rank in range(k):
                            idx = top_idxs[rank].item()
                            anchor = source_chunks[idx]
                            a_np = claim_emb[i].cpu().numpy()
                            b_np = source_emb[idx].cpu().numpy()

                            dist = scipy_cosine(a_np, b_np)
                            sem = (
                                float((1.0 - dist) * 100.0)
                                if not math.isnan(dist)
                                else 0.0
                            )
                            lex = float(lexical_jaccard(claim, anchor))
                            candidates.append((anchor, sem, lex))

                        local_results.append(candidates)

                st.session_state["claims"] = claims
                st.session_state["local_results"] = local_results
                st.session_state["ai_results"] = None
                st.rerun()

with col_v2:
    if st.session_state.get("claims"):
        if st.button("2. Run AI Contextual Check", type="primary"):
            if not gemini_api_key:
                st.error("API Key required.")
            else:
                claims_text = "\n".join(
                    [f"{i + 1}. {c}" for i, c in enumerate(st.session_state["claims"])]
                )
                prompt = rf"""You are a factual verification system for academic study notes.

SOURCE DOCUMENT:
{source_text}

CLAIMS TO VERIFY:
{claims_text}

Determine if each claim is:
- SUPPORTED: directly stated or logically equivalent to the source (including algebraic or notation rephrasing).
- NOT SUPPORTED: the source lacks sufficient context, definitions, or steps to verify this claim.
- CONTRADICTED: the source directly conflicts with or negates the claim.

Output EXACTLY one line per claim in this precise format:
[Claim Number] | [STATUS] | [1-2 sentence concise explanation of how the source aligns with or differs from the claim]

Examples of valid explanations:
1 | SUPPORTED | The source explicitly defines the continuous CDF $F(x)$ as the integral of $f(y)$ from $-\infty$ to $x$, matching this formula.
2 | NOT SUPPORTED | While the source discusses discrete expectation, it does not define or calculate the continuous integral form specified here.
3 | CONTRADICTED | The source states that $F(x)$ is strictly non-decreasing, whereas the claim asserts that $F(x)$ can decrease as $x$ increases.
"""
                try:
                    client = genai.Client(api_key=gemini_api_key)
                    with st.spinner("Running AI analysis..."):
                        res = call_gemini(client, model_name, prompt)
                        res_text = res.text if res and res.text else ""

                        ai_dict = {}
                        for line in res_text.split("\n"):
                            line = line.strip()
                            if not line or "|" not in line:
                                continue
                            parts = [p.strip() for p in line.split("|", 2)]
                            if len(parts) == 3:
                                num_str = re.sub(r"\D", "", parts[0])
                                if num_str.isdigit():
                                    idx = int(num_str) - 1
                                    ai_dict[idx] = {
                                        "status": parts[1],
                                        "reason": parts[2],
                                    }

                        st.session_state["ai_results"] = ai_dict
                        st.rerun()
                except Exception as e:
                    st.error(f"AI Check Error: {e}")

# Render Unified Analysis
if (
    st.session_state.get("claims")
    and st.session_state.get("local_results")
    and len(st.session_state["claims"]) == len(st.session_state["local_results"])
):
    st.caption(
        f"Analyzed **{len(st.session_state['claims'])}** claims. Local analysis checks vector/lexical match against top source chunks. AI analysis evaluates overall contextual faithfulness."
    )

    for i, claim in enumerate(st.session_state["claims"]):
        candidates = st.session_state["local_results"][i]
        anchor, semantic, lexical = candidates[0]
        alternates = candidates[1:]
        avg_score = round((semantic + lexical) / 2.0, 1)

        with st.container(border=True):
            st.markdown(f"**{i + 1}. {claim}**")

            c_loc, c_ai = st.columns(2)

            # Left Column: Local Search
            with c_loc:
                st.markdown("**Local Similarity Match**")
                if avg_score >= 75:
                    st.markdown(":green[**Well-Grounded**]")
                elif avg_score >= 50:
                    st.markdown(":orange[**Partial Match**]")
                else:
                    st.markdown(":red[**Weak Match**]")

                m1, m2, m3 = st.columns(3)
                m1.caption(f"Sem: {semantic:.1f}%")
                m1.progress(clamp_pct(semantic))
                m2.caption(f"Lex: {lexical:.1f}%")
                m2.progress(clamp_pct(lexical))
                m3.caption(f"Avg: {avg_score:.1f}%")
                m3.progress(clamp_pct(avg_score))

                with st.expander("Matched Source Chunks", expanded=False):
                    st.markdown(f"**Top Candidate:**\n{anchor}")
                    if alternates:
                        st.caption("Secondary Candidates:")
                        for alt_anchor, alt_sem, alt_lex in alternates:
                            st.markdown(
                                f"- *(Sem {alt_sem:.0f}%, Lex {alt_lex:.0f}%)* {alt_anchor}"
                            )

            # Right Column: AI Verification
            with c_ai:
                st.markdown("**AI Contextual Verdict**")
                if (
                    st.session_state.get("ai_results")
                    and i in st.session_state["ai_results"]
                ):
                    ai_res = st.session_state["ai_results"][i]
                    status = ai_res["status"].upper()

                    if "SUPPORTED" in status and "NOT" not in status:
                        st.success(f"**{status}**\n\n{ai_res['reason']}")
                    elif "CONTRADICTED" in status:
                        st.error(f"**{status}**\n\n{ai_res['reason']}")
                    else:
                        st.warning(f"**{status}**\n\n{ai_res['reason']}")
                else:
                    st.info("Pending AI Verification")

st.divider()

# ==========================================
# STEP 4: VERSION COMPARISON (OPTIONAL)
# ==========================================
st.subheader("4. Version Comparison (Optional)")
st.caption(
    "Compares the newly synthesized note against the baseline note to detect contradictions."
)

if st.button("Check Notes for Discrepancies"):
    if not st.session_state.existing_note_val.strip() or not generated_note.strip():
        st.warning(
            "Both an Existing Baseline Note and a Target Note are required for comparison."
        )
    elif not gemini_api_key:
        st.error("API Key required.")
    else:
        try:
            client = genai.Client(api_key=gemini_api_key)
            prompt = rf"""Compare these two versions of a study note.
EXISTING BASELINE NOTE:
{st.session_state.existing_note_val}

NEW TARGET NOTE:
{generated_note}

List any statement in the NEW TARGET NOTE that contradicts the EXISTING BASELINE NOTE as a short bulleted list. 
If there are no contradictions, output exactly: "No contradictions found." Plain English only."""
            with st.spinner("Comparing versions..."):
                res = call_gemini(client, model_name, prompt)
                result = res.text if res and res.text else ""
                if "no contradictions found" in result.lower():
                    st.success(
                        "No contradictions detected between existing and target notes."
                    )
                else:
                    st.warning("Potential contradictions found:")
                    st.markdown(result)
        except Exception as e:
            st.error(f"Comparison Error: {str(e)}")
