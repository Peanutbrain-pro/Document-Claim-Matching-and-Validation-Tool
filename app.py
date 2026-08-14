import streamlit as st
import torch
import nltk
import re
import time
from scipy.spatial.distance import cosine as scipy_cosine
from sentence_transformers import SentenceTransformer, util
from google import genai
from google.genai import types
from nltk.corpus import stopwords
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

# Download punkt (for tokenization) and stopwords dataset
for pkg, path in [("punkt", "tokenizers/punkt"), ("stopwords", "corpora/stopwords")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg)

STOPWORDS = set(stopwords.words("english"))

# --- PAGE CONFIG ---
st.set_page_config(page_title="Study Note Verification", layout="wide")
st.title("Study Note Faithfulness & Provenance Checker")

st.markdown("""
**Verification Architecture:**
1. **Local NLP Layer (NLTK + SciPy):** Offline extraction, semantic embeddings, and lexical matching. Finds exact locations in the source text.
2. **AI Faithfulness Layer (Gemini):** Holistic one-pass verification on the exact same claims extracted by the local layer to resolve contextual ambiguities.
""")

# --- SIDEBAR ---
st.sidebar.header("API Configuration")
gemini_api_key = st.sidebar.text_input(
    "Gemini API Key:", value=API_KEY, type="password"
)
model_name = st.sidebar.text_input("Model Name:", value="gemini-3.1-flash-lite")
max_rpm = st.sidebar.slider("Rate Limit (Max RPM)", 3, 15, 10)


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device=torch.device("cpu"))


with st.spinner("Initializing local embedding engine..."):
    embedder = load_embedder()

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
            st.info(f"Rate limit active. Waiting {wait:.1f}s...")
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
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 15 * (attempt + 1)
                st.warning(f"Rate limit hit. Backing off {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Exceeded max retries after repeated rate-limit errors.")


# To shorten bullet points into single line to avoid separation issues
bullet_point_shortener = re.compile(
    r"(?m)^[ \t]*(?:\d+[.)]|[-*])\s+(.+(?:\n(?![ \t]*(?:\d+[.)]|[-*])\s+)(?!\s*$)[ \t]*\S.*)*)"
)


# def is_noise_line(line: str) -> bool:
#     s = line.strip()
#     if len(s) < 8 or s.startswith("#") or re.fullmatch(r"[\d.\-#*\s]+", s):
#         return True
#     if len(re.findall(r"[A-Za-z]", s)) < 6:
#         return True
#     if re.match(
#         r"^(here is|here's|below is|sure[,!]?\s*here|certainly)", s, re.IGNORECASE
#     ):
#         return True
#     return False


def extract_claims(note_text: str):
    claims = []
    for block in re.split(r"\n\s*\n", note_text):
        block = block.strip()
        if not block:
            continue

        items = list(bullet_point_shortener.finditer(block))
        if items:
            for m in items:
                claims.append(m.group(1))
        else:
            for line in block.split("\n"):
                line = line.strip()
                # Only ignore standard markdown headers
                if line.startswith("#") or not line:
                    continue
                for s in nltk.sent_tokenize(line):
                    claims.append(s)
    return claims


def extract_source_chunks(src_text: str):
    chunks = []
    for block in re.split(r"\n\s*\n", src_text):
        clean_lines = [ln.strip() for ln in block.strip().split("\n") if ln.strip()]
        chunks.extend(clean_lines)

        # Create 2-line sliding windows
        for i in range(len(clean_lines) - 1):
            merged = clean_lines[i] + " " + clean_lines[i + 1]
            if merged not in chunks:
                chunks.append(merged)

    return chunks if chunks else [src_text.strip()]


def extract_source_chunks(src_text: str):
    chunks = []
    for paragraph in re.split(r"\n\s*\n", src_text):
        lines = [ln.strip() for ln in paragraph.strip().split("\n")]
        # clean_lines = [ln for ln in lines if not is_noise_line(ln)]
        clean_lines = lines
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
    return len(wa & wb) / len(wa | wb) * 100


# ==========================================
# STEP 1: SOURCE OCR
# ==========================================
st.subheader("1. Source Document OCR")
uploaded_file = st.file_uploader(
    "Upload Image or PDF Document", type=["png", "jpg", "jpeg", "pdf"]
)

if uploaded_file and st.button("Extract OCR Text"):
    if not gemini_api_key:
        st.error("Enter Gemini API Key in sidebar.")
    else:
        try:
            client = genai.Client(api_key=gemini_api_key)
            ocr_prompt = (
                "Perform exact OCR on this document. Preserve math equations clearly.\n"
                "Every LaTeX symbol/formula must be wrapped in $...$ (inline) or $$...$$ (display).\n"
                "Never output bare LaTeX commands outside delimiters.\n"
                "CRITICAL: Do NOT generate any page markers, headers, or footers (e.g., do not output '==Start of OCR==' or '==End of OCR=='). "
                "Output strictly the raw content of the document."
            )
            media_part = types.Part.from_bytes(
                data=uploaded_file.read(), mime_type=uploaded_file.type
            )
            with st.spinner("Extracting..."):
                response = call_gemini(client, model_name, [media_part, ocr_prompt])
                st.session_state.ocr_text = response.text
        except Exception as e:
            st.error(f"OCR Error: {str(e)}")

col_s1, col_s2 = st.columns(2)
with col_s1:
    source_text = st.text_area(
        "Source Ground Truth Text:", value=st.session_state.ocr_text, height=200
    )
with col_s2:
    with st.expander("Preview Formatted Source", expanded=False):
        st.markdown(source_text if source_text.strip() else "No text.")

st.divider()

# ==========================================
# STEP 2: CANDIDATE NOTE
# ==========================================
st.subheader("2. Candidate Study Note Synthesis")
col_n1, col_n2 = st.columns(2)
with col_n1:
    existing_unit_note = st.text_area(
        "Existing Master Note (Optional):",
        value=st.session_state.existing_note_val,
        height=120,
    )
    st.session_state.existing_note_val = existing_unit_note
    if st.button("Synthesize Merged Note"):
        if not gemini_api_key or not source_text.strip():
            st.error("API key and source text required.")
        else:
            try:
                client = genai.Client(api_key=gemini_api_key)
                merge_prompt = f"Synthesize study notes from:\nExisting: {existing_unit_note}\nSource: {source_text}\nWrite clean Markdown. Wrap LaTeX in $...$. No conversational preamble."
                with st.spinner("Synthesizing..."):
                    res = call_gemini(client, model_name, merge_prompt)
                    st.session_state.generated_note_val = res.text
            except Exception as e:
                st.error(f"Synthesis Error: {str(e)}")
with col_n2:
    generated_note = st.text_area(
        "Target Note to Verify:", value=st.session_state.generated_note_val, height=200
    )

st.divider()

# ==========================================
# STEP 3: UNIFIED CLAIMS VERIFICATION
# ==========================================
st.subheader("3. Unified Claims Verification")
st.markdown(
    "Extracts claims once locally, then applies both statistical NLP and holistic AI verification to the exact same list."
)

col_v1, col_v2 = st.columns([0.5, 0.5])

with col_v1:
    if st.button("1. Run Local NLP Extraction & Analysis", type="primary"):
        if not source_text.strip() or not generated_note.strip():
            st.error("Missing source text or candidate note!")
        else:
            source_chunks = extract_source_chunks(source_text)
            claims = extract_claims(generated_note)

            if not claims:
                st.warning("No valid claims detected in the text.")
            else:
                with st.spinner(f"Computing embeddings for {len(claims)} claims..."):
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
                            sem = float((1 - scipy_cosine(a_np, b_np)) * 100)
                            lex = float(lexical_jaccard(claim, anchor))
                            candidates.append((anchor, sem, lex))

                        local_results.append(candidates)

                st.session_state["claims"] = claims
                st.session_state["local_results"] = local_results
                st.session_state["ai_results"] = (
                    None  # Reset AI results if local is rerun
                )
                st.rerun()

with col_v2:
    if st.session_state["claims"]:
        if st.button("2. Run Holistic AI Check (Gemini)", type="primary"):
            if not gemini_api_key:
                st.error("API Key required.")
            else:
                claims_text = "\n".join(
                    [f"{i + 1}. {c}" for i, c in enumerate(st.session_state["claims"])]
                )
                prompt = rf"""You are a strict factual verification system for academic study notes.

SOURCE DOCUMENT:
{source_text}

CLAIMS TO VERIFY:
{claims_text}

Determine if each claim is:
- SUPPORTED: directly stated or logically equivalent to the source (including standard algebraic/notation rephrasing).
- NOT SUPPORTED: the source lacks sufficient context, definitions, or steps to verify this claim.
- CONTRADICTED: the source directly conflicts with or negates the claim.

Output EXACTLY one line per claim in this precise format:
[Claim Number] | [STATUS] | [1-2 sentence concise explanation of how the source aligns with or differs from the claim]

Examples of good explanations:
1 | SUPPORTED | The source explicitly defines the continuous CDF $F(x)$ as the integral of $f(y)$ from $-\infty$ to $x$, matching this formula.
2 | NOT SUPPORTED | While the source discusses discrete expectation, it does not define or calculate the continuous integral form specified here.
3 | CONTRADICTED | The source states that $F(x)$ is strictly non-decreasing, whereas the claim asserts that $F(x)$ can decrease as $x$ increases.
"""
                try:
                    client = genai.Client(api_key=gemini_api_key)
                    with st.spinner("AI analyzing claims..."):
                        res = call_gemini(client, model_name, prompt)

                        ai_dict = {}
                        for line in res.text.split("\n"):
                            line = line.strip()
                            if not line or "|" not in line:
                                continue
                            parts = [p.strip() for p in line.split("|", 2)]
                            if len(parts) == 3 and parts[0].isdigit():
                                idx = int(parts[0]) - 1
                                ai_dict[idx] = {"status": parts[1], "reason": parts[2]}

                        st.session_state["ai_results"] = ai_dict
                        st.rerun()
                except Exception as e:
                    st.error(f"AI Check Error: {e}")

# Render the Unified View
if st.session_state.get("claims"):
    st.caption(
        f"Analyzing **{len(st.session_state['claims'])}** extracted claims. Local analysis checks single-line exact matches. AI analysis checks full document context."
    )

    for i, claim in enumerate(st.session_state["claims"]):
        # Retrieve data
        candidates = st.session_state["local_results"][i]
        anchor, semantic, lexical = candidates[0]
        alternates = candidates[1:]
        avg_score = round((semantic + lexical) / 2, 1)

        with st.container(border=True):
            st.markdown(f"**{i + 1}. {claim}**")

            c_loc, c_ai = st.columns(2)

            # Left Column: Local NLP
            with c_loc:
                st.markdown("**Local Anchor Search**")
                if avg_score >= 75:
                    st.markdown(":green[**Well-grounded**]")
                elif avg_score >= 50:
                    st.markdown(":orange[**Partial Match**]")
                else:
                    st.markdown(":red[**Weak Match**]")

                m1, m2, m3 = st.columns(3)
                m1.caption(f"Sem: {semantic:.1f}%")
                m1.progress(min(max(semantic / 100, 0.0), 1.0))
                m2.caption(f"Lex: {lexical:.1f}%")
                m2.progress(min(max(lexical / 100, 0.0), 1.0))
                m3.caption(f"Avg: {avg_score:.1f}%")
                m3.progress(min(max(avg_score / 100, 0.0), 1.0))

                with st.expander("View Top Matched Text", expanded=False):
                    st.markdown(f"**Top Match:**\n{anchor}")
                    if alternates:
                        st.caption("Next Best Candidates:")
                        for alt_anchor, alt_sem, alt_lex in alternates:
                            st.markdown(
                                f"- *(Sem {alt_sem:.0f}%, Lex {alt_lex:.0f}%)* {alt_anchor}"
                            )

            # Right Column: AI Faithfulness
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
                    st.info("Awaiting AI Verification...")

st.divider()

# ==========================================
# STEP 4: MERGE DISCREPANCY CHECK
# ==========================================
st.subheader("4. Merge Discrepancy Check (Optional)")
st.caption(
    "Checks whether the newly generated note contradicts the previous baseline note."
)

if st.button("Check Merge for Contradictions"):
    if not st.session_state.existing_note_val.strip() or not generated_note.strip():
        st.warning(
            "Both an Existing Note and a Target Note are required for comparison."
        )
    elif not gemini_api_key:
        st.error("API Key required.")
    else:
        try:
            client = genai.Client(api_key=gemini_api_key)
            prompt = rf"""Compare these two versions of a study note.
EXISTING NOTE (before merge):
{st.session_state.existing_note_val}
NEW MERGED NOTE (after merge):
{generated_note}
List any statement in the NEW MERGED NOTE that contradicts the EXISTING NOTE as a short bulleted list. 
If there are no contradictions, output exactly: "No contradictions found." Plain English only."""
            with st.spinner("Checking for discrepancies..."):
                res = call_gemini(client, model_name, prompt)
                result = res.text
                if "no contradictions found" in result.lower():
                    st.success(
                        "✅ No contradictions detected between existing and merged notes."
                    )
                else:
                    st.warning("Potential contradictions found:")
                    st.markdown(result)
        except Exception as e:
            st.error(f"Merge check error: {str(e)}")
