from langchain import LLMChain, PromptTemplate
from langchain_community.llms import Ollama
import re, ast, os, json, numpy as np
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch

# -----------------------------
# 1.define LLM
# -----------------------------
llm = Ollama(model="llama2", temperature=0)

# -----------------------------
# 2. Prompts
# -----------------------------




#influential_prompt
influential_prompt = PromptTemplate.from_template(r"""
You are a careful medical assistant.

Task:
- Identify 2–3 influential medical words or short phrases from the QUESTION only.
- Focus only on descriptive symptoms, conditions, medications, or affected body parts.
- Return strictly a valid Python list of strings (e.g. ["fever","cough"]).

QUESTION:
{input}
""")
influential_chain = LLMChain(llm=llm, prompt=influential_prompt, output_key="influential_words")




#synonym_prompt
synonym_prompt = PromptTemplate.from_template(r"""
You are given ONE influential medical term.

Task:
- Generate ONE medically valid synonym (common medical alternative term).
- If no valid synonym exists, return "no change".

Return ONLY valid JSON:
{{
  "word": "{word}",
  "synonym": "..."
}}
""")
synonym_chain = LLMChain(llm=llm, prompt=synonym_prompt, output_key="synonym")





#antonym_prompt
antonym_prompt = PromptTemplate.from_template(r"""
You are given ONE influential medical term.

Task:
- Generate ONE medically valid antonym or logical negation.
- If no valid antonym exists, return "no {word}".

Return ONLY valid JSON:
{{
  "word": "{word}",
  "antonym": "..."
}}
""")
antonym_chain = LLMChain(llm=llm, prompt=antonym_prompt, output_key="antonym")



#counterfactual_prompt
counterfactual_prompt = PromptTemplate.from_template(r"""
You are a careful medical assistant.

Task:
- Rewrite according to strict format below.
- Do NOT include any extra explanations outside the format.
- Follow EXACTLY the example structure.
- Always include detailed, step-by-step reasoning that explains the medical logic behind the Final Advice.
- Specifically explain how the replacement word affects the diagnosis, reasoning, and advice.
- If the replacement does not make medical sense, explicitly state this in the reasoning.

### EXAMPLE FORMAT
--- Counterfactual for "persistent" (Replacement: "chronic") ---
Modified Input:
"I have a chronic dry cough for the last 3 weeks. Should I be concerned?"

Reasoning:
Step 1: Identify key symptoms and relevant conditions.
- "chronic dry cough" indicates a long-term airway issue.
Step 2: Consider possible diagnoses based on symptoms.
- Chronic cough >2 weeks → asthma, bronchitis, other chronic conditions.
Step 3: Analyze relationships between symptoms and determine the most likely diagnosis.
- Non-productive cough with chronic duration → suspicious for asthma variant or bronchitis.
Step 4: Consider potential alternative diagnoses and eliminate them.
Step 5: Suggest next steps or treatments.
- Pulmonary function testing, spirometry, smoking history, inhaled bronchodilator trial.
Step 6: Explain why you chose this final advice.
- Chronic label strengthens suspicion for long-term disease, requires workup.

Final Advice:
"Chronic dry cough lasting more than 3 weeks warrants pulmonary function testing. Please schedule a spirometry and consider an inhaled bronchodilator trial."

Original Question: I have a persistent dry cough for the last 3 weeks. Should I be concerned?
Original Advice: A persistent dry cough lasting more than two weeks could indicate an underlying condition such as asthma or even something more serious. It's advisable to consult a doctor for a proper evaluation.

--- END OF EXAMPLE FORMAT ---

Now apply the same format to this case:

Word: {word}
Replacement: {replacement}
Original Question: {input}
Original Advice: {output}
""")
counterfactual_chain = LLMChain(llm=llm, prompt=counterfactual_prompt, output_key="counterfactual")







# -----------------------------
# 3. Helper Functions
# -----------------------------
def extract_influential_words(raw_output):
    try:
        match = re.search(r"\[.*?\]", str(raw_output), re.DOTALL)
        if match:
            return ast.literal_eval(match.group())
    except:
        pass
    return []


def safe_parse_json(raw, key, word):
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if key in data and data[key] and str(data[key]).strip():
                return str(data[key]).strip()
    except:
        pass
    return f"no {word}" if key == "antonym" else "no change"


def cosine_similarity(a: str, b: str) -> float:
    vect = TfidfVectorizer().fit([a, b])
    vecs = vect.transform([a, b])
    return float((vecs[0] @ vecs[1].T).toarray()[0][0])


def sequence_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def replace_word_in_text(text, word, replacement):
    pattern = r'\b{}\b'.format(re.escape(word))
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)


def enforce_format(text, word, replacement, sample):
    #ensure counterfactual text follows strict format with detailed reasoning
    if not text or "--- Counterfactual" not in text:
        if replacement.lower() == "no change":
            infl_words = sample.get("influential_words", [])
            reasoning_lines = [
                "Step 1: Identify key symptoms and relevant conditions.",
                f"- '{', '.join(infl_words)}' indicate main clinical concerns." if infl_words else "- Key symptoms identified from input.",
                "Step 2: Consider possible diagnoses based on symptoms.",
                "Step 3: Analyze relationships between symptoms and determine the most likely diagnosis.",
                "Step 4: Consider potential alternative diagnoses and eliminate them.",
                "Step 5: Suggest next steps or treatments based on standard care.",
                "Step 6: Explain why you chose this final advice."
            ]
            reasoning = "\n".join(reasoning_lines)
            final_advice = sample.get("output", "(fallback advice)")
        else:
            reasoning = "\n".join([f"Step {i}: (auto-generated)" for i in range(1,7)])
            final_advice = "(fallback advice)"

        return f"""
--- Counterfactual for "{word}" (Replacement: {replacement}) ---
Modified Input:
"{sample['input']}"

Reasoning:
{reasoning}

Final Advice:
"{final_advice}"

Original Question: {sample['input']}
Original Advice: {sample['output']}
"""
    return text.strip()


def get_replacements(word, cache={}):
    if word in cache:
        return cache[word]
    syn_raw = synonym_chain.invoke({"word": word})
    ant_raw = antonym_chain.invoke({"word": word})
    syn = safe_parse_json(syn_raw.get("synonym", ""), "synonym", word)
    ant = safe_parse_json(ant_raw.get("antonym", ""), "antonym", word)
    replacements = []
    if syn.lower() != "no change":
        replacements.append(syn)
    if ant.lower() not in [f"no {word.lower()}", "nochange"]:
        replacements.append(ant)
    if not replacements:
        replacements.append("no change")
    cache[word] = replacements
    return replacements









# -----------------------------
# 4. Pipeline
# -----------------------------
def process_sample(sample, max_retries=5):
    infl_raw = influential_chain.invoke({"input": sample["input"]})
    infl_words = extract_influential_words(infl_raw.get("influential_words", []))
    infl_words = sorted(infl_words, key=lambda x: -len(x.split()))

    results = {"influential_words": infl_words, "counterfactuals": []}

    for w in infl_words:
        for rep in get_replacements(w):
            modified_input = replace_word_in_text(sample["input"], w, rep)
            cf_text = None
            retries = 0
            while retries < max_retries and (not cf_text or "--- Counterfactual" not in cf_text):
                cf_result = counterfactual_chain.invoke({
                    "input": modified_input,
                    "output": sample["output"],
                    "word": w,
                    "replacement": rep
                })
                cf_text = cf_result.get("counterfactual", "")
                retries += 1

            cf_text = enforce_format(cf_text, w, rep, sample)

            final_advice = "n/a"
            m = re.search(r"Final Advice:\s*\"(.*?)\"", cf_text, re.DOTALL)
            if m:
                final_advice = m.group(1).strip()

            if rep.lower() == "no change":
                cf_type = "syn"
            elif rep.lower().startswith("no "):
                cf_type = "ant"
            else:
                cf_type = "syn"

            valid = "--- Counterfactual" in cf_text

            results["counterfactuals"].append({
                "text": cf_text,
                "modified_input": modified_input,
                "final_advice": final_advice,
                "valid": valid,
                "type": cf_type
            })

    return results















# -----------------------------
# 5. Evaluation
# -----------------------------

def _triplet_scores(a: str, b: str):
    return (cosine_tfidf(a, b), sequence_similarity(a, b), cosine_bert(a, b))





# Cosine similarity με TF-IDF
def cosine_tfidf(a: str, b: str) -> float:
    vect = TfidfVectorizer().fit([a, b])
    vecs = vect.transform([a, b])
    return float((vecs[0] @ vecs[1].T).toarray()[0][0])





# Cosine similarity με BERT embeddings
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_bert(a: str, b: str) -> float:
    emb = bert_model.encode([a, b], convert_to_tensor=True)
    return float(torch.nn.functional.cosine_similarity(emb[0], emb[1], dim=0))


def evaluate_dataset(samples, max_items=10, verbose=True):
    buckets_strict = {
        "syn_vs_gt":   {"cosine": [], "seq": [], "bert": []},
        "syn_vs_orig": {"cosine": [], "seq": [], "bert": []},
        "ant_vs_orig": {"cosine": [], "seq": [], "bert": []},
    }
    buckets_robust = {
        "syn_vs_gt":   {"cosine": [], "seq": [], "bert": []},
        "syn_vs_orig": {"cosine": [], "seq": [], "bert": []},
        "ant_vs_orig": {"cosine": [], "seq": [], "bert": []},
    }

    for i, sample in enumerate(samples[:max_items], start=1):
        res = process_sample(sample)
        if verbose:
            print(f"\n=== Sample {i} ===")
            print("Input:", sample["input"])
            print("Output:", sample["output"])
            print("Influential Words:", res["influential_words"])
            print("\n--- Counterfactuals ---")

        for cf in res["counterfactuals"]:
            if verbose:
                print(cf["text"])

            #robust (all outputs)
            if cf["type"] == "syn":
                c, s, b = _triplet_scores(cf["final_advice"], sample["output"])
                buckets_robust["syn_vs_gt"]["cosine"].append(c)
                buckets_robust["syn_vs_gt"]["seq"].append(s)
                buckets_robust["syn_vs_gt"]["bert"].append(b)

                c2, s2, b2 = _triplet_scores(cf["modified_input"], sample["input"])
                buckets_robust["syn_vs_orig"]["cosine"].append(c2)
                buckets_robust["syn_vs_orig"]["seq"].append(s2)
                buckets_robust["syn_vs_orig"]["bert"].append(b2)

            elif cf["type"] == "ant":
                c3, s3, b3 = _triplet_scores(cf["final_advice"], sample["output"])
                buckets_robust["ant_vs_orig"]["cosine"].append(c3)
                buckets_robust["ant_vs_orig"]["seq"].append(s3)
                buckets_robust["ant_vs_orig"]["bert"].append(b3)

            # only strict - if output is not valid / no fallback
            if cf["valid"] and "fallback" not in cf["final_advice"].lower():
                if cf["type"] == "syn":
                    c, s, b = _triplet_scores(cf["final_advice"], sample["output"])
                    buckets_strict["syn_vs_gt"]["cosine"].append(c)
                    buckets_strict["syn_vs_gt"]["seq"].append(s)
                    buckets_strict["syn_vs_gt"]["bert"].append(b)

                    c2, s2, b2 = _triplet_scores(cf["modified_input"], sample["input"])
                    buckets_strict["syn_vs_orig"]["cosine"].append(c2)
                    buckets_strict["syn_vs_orig"]["seq"].append(s2)
                    buckets_strict["syn_vs_orig"]["bert"].append(b2)

                elif cf["type"] == "ant":
                    c3, s3, b3 = _triplet_scores(cf["final_advice"], sample["output"])
                    buckets_strict["ant_vs_orig"]["cosine"].append(c3)
                    buckets_strict["ant_vs_orig"]["seq"].append(s3)
                    buckets_strict["ant_vs_orig"]["bert"].append(b3)

    def mean(lst):
        return float(np.mean(lst)) if lst else None

    def collect(buckets):
        final_metrics = {}
        for k, v in buckets.items():
            final_metrics[f"{k}_cosine"] = mean(v["cosine"])
            final_metrics[f"{k}_seq"]     = mean(v["seq"])
            final_metrics[f"{k}_bert"]    = mean(v["bert"])
        return final_metrics

    return {
        "strict": collect(buckets_strict),
        "robust": collect(buckets_robust)
    }




def pretty_print_metrics(results):
    print("\n================ METRICS ================")
    for mode, m in results.items():
        print(f"\n--- {mode.upper()} ---")
        def fmt(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else "n/a"
        print(f"syn_vs_gt    ->  TFIDF: {fmt(m.get('syn_vs_gt_cosine'))} | SEQ: {fmt(m.get('syn_vs_gt_seq'))} | BERT: {fmt(m.get('syn_vs_gt_bert'))}")
        print(f"syn_vs_orig  ->  TFIDF: {fmt(m.get('syn_vs_orig_cosine'))} | SEQ: {fmt(m.get('syn_vs_orig_seq'))} | BERT: {fmt(m.get('syn_vs_orig_bert'))}")
        print(f"ant_vs_orig  ->  TFIDF: {fmt(m.get('ant_vs_orig_cosine'))} | SEQ: {fmt(m.get('ant_vs_orig_seq'))} | BERT: {fmt(m.get('ant_vs_orig_bert'))}")
    print("=========================================\n")









#followup_prompt 
# 6. Interactive Session
# -----------------------------
followup_prompt = PromptTemplate.from_template(r"""
You are an interactive counterfactual medical assistant.

Context:
Original Question: {input}
Original Advice: {output}
Influential Words: {infl_words}
Previous Counterfactuals:
{counterfactuals}

User request:
{user_query}

Task:
- Carefully reason about the request.
- Return in strict format:

--- Follow-up Analysis ---
Modified Input:
"<new input>"

Reasoning:
Step 1: ...
Step 2: ...
Step 3: ...
Step 4: ...
Step 5: ...
Step 6: ...

Final Advice:
"<new advice>"
""")
followup_chain = LLMChain(llm=llm, prompt=followup_prompt, output_key="text")





def interactive_session(sample, results):
    print("\n=== Interactive Session ===")
    while True:
        q = input("\nUser: ")
        if q.lower() in ["exit", "quit"]:
            break
        res = followup_chain.invoke({
            "input": sample["input"],
            "output": sample["output"],
            "infl_words": results["influential_words"],
            "counterfactuals": "\n".join(cf["text"] for cf in results["counterfactuals"]),
            "user_query": q
        })
        print("\n" + res["text"])












# -----------------------------
# 7. Example Run
# -----------------------------
results_metrics = evaluate_dataset(cleaned_healthcaremagic, max_items=100)
pretty_print_metrics(results_metrics)

sample = cleaned_healthcaremagic[0]
res = process_sample(sample)
interactive_session(sample, res)
