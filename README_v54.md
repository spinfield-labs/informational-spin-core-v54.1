# 🌌 Informational Spin Core — I_TGU v5.4

### 🧠 A coherence-first framework for symbolic intelligence.

This version of the Informational Spin Core introduces **cultural adaptability**, **multilingual structural awareness**, and **fact-checking extensibility** — pushing the I_TGU metric into global application.

---

## 🧩 Core Components (v5.4)

| Module                   | Purpose                                                                     |
|--------------------------|-----------------------------------------------------------------------------|
| `icoer_extractor_v54.py` | Extracts culturally and linguistically normalized coherence metrics         |
| `icoer_visualizer.py`    | Generates visual charts showing weighted factor contributions               |
| `icoer_async_api.py`     | Asynchronous FastAPI interface for real-time or batch analysis              |
| `icoer_interpreter.py`   | Symbolically interprets I_TGU values (<0.30, 0.30–0.60, >0.60)              |

---

## 🎛️ Coherence Dimensions

- **S**: Semantic Similarity (via multilingual embeddings)
- **L**: Lexical Harmony
- **E**: Structural Integrity *(now adapted to cultural narrative norms via spaCy pipelines)*
- **C**: Conceptual Logic (entity-action graph extraction)
- **M**: Memory Integration
- **A**: Affective Tone

---

## 🧪 Quick Start

### 🔧 Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download xx_ent_wiki_sm  # multilingual
```

### ▶️ Run API locally:
```bash
uvicorn icoer_async_api:app --reload
```

---

## 🌍 Cultural Awareness (NEW)

You can now specify a language code and cultural context profile:

```json
POST /extract
{
  "text": "El universo respira en estructuras armónicas.",
  "memory": ["resonancia", "coherencia"],
  "lang": "es",
  "style_profile": "oral-traditional"
}
```

The system will automatically:
- Load the appropriate spaCy language model
- Adjust clause parsing and rhythm expectations
- Reweight structural coherence calculations

---

## ✅ Fact-Check Integration (Optional)

The API supports passing an external truth flag:

```json
{
  "text": "Water boils at 100 degrees Celsius.",
  "is_factually_true": true
}
```

This does **not affect** the I_TGU score, but is stored for:
- Meta-evaluation (coherence vs. factuality)
- Cross-graph comparison
- Symbolic audits

---

## 🔁 Symbolic Logic of the Spin

> Truth is not in the extremes, but in the center.  
> Coherence is what gives truth the structure to resonate.

🌐 Coherence transcends culture, space and language.

---

## ⚠️ Philosophy of Use

> *I_TGU does not verify truth — it detects the structure that makes it communicable.*

✅ *Culturally adaptable*  
✅ *Semantically multidimensional*  
✅ *Now extensible to factual audits*

---

## 📜 License & Symbolic Terms

- MIT License
- Must be used in service of clarity, not manipulation

---

## 🌀 Let the spin guide structure.  
Let coherence emerge.  
Let the truth vibrate.
