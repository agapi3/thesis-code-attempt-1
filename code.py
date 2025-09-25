import torch
import json
import re
import ast
import os
import pandas as pd
from datasets import load_dataset
from langchain_ollama import OllamaLLM
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# 1) Load dataset & save as healthcaremagic_dataset.json
# -----------------------------

#load dataset
ds = load_dataset("Malikeh1375/medical-question-answering-datasets", "chatdoctor_healthcaremagic") 
ds_healthcaremagic = ds["train"]

# JSON
with open("healthcaremagic_dataset.json", "w", encoding="utf-8") as f:
    json.dump(ds_healthcaremagic.to_dict(), f, ensure_ascii=False, indent=4)
#print("saved")


# Φόρτωση από JSON
with open("healthcaremagic_dataset.json", "r", encoding="utf-8") as f:
    ds_healthcaremagic = json.load(f)

ds_healthcaremagic = ds["train"]
print(ds_healthcaremagic[:5])



#check 
print(len(ds_healthcaremagic))
print("NaN values per column (before cleaning):")
print(pd.DataFrame(ds_healthcaremagic).isna().sum())
print("Total duplicates (before cleaning):", pd.DataFrame(ds_healthcaremagic).duplicated(subset=["instruction","input","output"]).sum())



# -----------------------------
#1.1) Data Preprocessing
# -----------------------------

#Cleaning function
def clean_text(text):
    if not text:
        return ""  #None or empty : returns an empty string

    text = text.strip().lower()  #whitespace removal + lowercase
    text = re.sub(r'\s+', ' ', text)  #replace multiple spaces with a single space
    text = re.sub(r'[^\w\s,.!?]', '', text)  #remove special characters except for -> comma, period, exclamation mark and question mark

    return text


#Cleaning dataset
cleaned_healthcaremagic = []

for example in ds_healthcaremagic:
    instruction = example['instruction'][0] if isinstance(example['instruction'], list) else example['instruction']
    input_text = example['input']
    response = example['output']

    cleaned_instruction = clean_text(instruction)
    cleaned_input = clean_text(input_text)
    cleaned_response = clean_text(response)

    #skip, if any field is empty after cleaning
    if not cleaned_instruction or not cleaned_input or not cleaned_response:
        continue

    cleaned_healthcaremagic.append({
        "instruction": cleaned_instruction,
        "input": cleaned_input,
        "output": cleaned_response
    })



#Remove duplicates
df = pd.DataFrame(cleaned_healthcaremagic)
df = df.drop_duplicates(subset=["instruction", "input", "output"])
cleaned_healthcaremagic = df.to_dict(orient="records")
print(cleaned_healthcaremagic[:5])


#Check
print(len(cleaned_healthcaremagic))
print("NaN values per column (after cleaning):")
print(df.isna().sum())
print("Total duplicates (after cleaning):", df.duplicated(subset=["instruction","input","output"]).sum())

# -----------------------------
#2)Define model
#3)Influential Words,CoT, Counterfactual examples and the final advice from model (implementation via langchain ) 
# -----------------------------

#Compute hash for a sample
def compute_sample_hash(sample):
    """Compute a hash for a sample based on instruction, input, output"""
    combined = f"{sample['instruction']}||{sample['input']}||{sample['output']}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()



# Cache setup
CACHE_FILE = "stored_results.pkl"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        stored_results = pickle.load(f)
    
else:
    stored_results = {}
    

#2.1 Define LLM
llm = Ollama(model="llama2", temperature=0)


#3 Prompts

#3.1 Influential Prompt
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




#3.2 Synonym Prompt
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





#3.3 Antonym Prompt
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








#3.4 Counterfactual Prompt
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
# 4. Helper Functions
# -----------------------------

#Extract influential words from LLM output (returns a list)
def extract_influential_words(raw_output):
    try:
        match = re.search(r"\[.*?\]", str(raw_output), re.DOTALL)
        if match:
            return ast.literal_eval(match.group())
    except:
        pass
    return []



#Ensure JSON responses from LLM are valid, return default if invalid
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



#Compute text similarity using TF-IDF cosine similarity
def cosine_similarity(a: str, b: str) -> float:
    vect = TfidfVectorizer().fit([a, b])
    vecs = vect.transform([a, b])
    return float((vecs[0] @ vecs[1].T).toarray()[0][0])


#Compute text similarity using SequenceMatcher ratio
def sequence_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()



#Replace a word in text with its synonym or antonym
def replace_word_in_text(text, word, replacement):
    pattern = r'\b{}\b'.format(re.escape(word))
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)


#Ensure the counterfactual output strictly follows the expected format
def enforce_format(text, word, replacement, sample):
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



#Get replacements (synonyms/antonyms) for a word, with caching to avoid repeated LLM calls
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
# 5. Pipeline
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
# 6. Evaluation
# -----------------------------
#fine-tuned embeddings on medical text are more faithful to the medical meaning
#https://huggingface.co/pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb 
#count similarity between original and counterfactual text

bert_model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')

def cosine_bert(a: str, b: str) -> float:
    emb = bert_model.encode([a, b], convert_to_tensor=True)
    return float(torch.nn.functional.cosine_similarity(emb[0], emb[1], dim=0))

def _triplet_scores(a: str, b: str):
    return (cosine_tfidf(a, b), sequence_similarity(a, b), cosine_bert(a, b))

def cosine_tfidf(a: str, b: str) -> float:
    vect = TfidfVectorizer().fit([a, b])
    vecs = vect.transform([a, b])
    return float((vecs[0] @ vecs[1].T).toarray()[0][0])



# ROUGE-L
rouge_scorer_fn = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
def rouge_l(a: str, b: str) -> float:
    try:
        score = rouge_scorer_fn.score(a, b)
        return score['rougeL'].fmeasure
    except:
        return 0.0

def meteor(a: str, b: str) -> float:
    try:
        return meteor_score([b.split()], a.split())
    except:
        return 0.0


# Distinct-n
def distinct_n(texts, n=2):
    if not texts:
        return 0.0
    total_ngrams, unique_ngrams = 0, set()
    for t in texts:
        tokens = t.split()
        ngrams = zip(*[tokens[i:] for i in range(n)])
        ngrams = list(ngrams)
        total_ngrams += len(ngrams)
        unique_ngrams.update(ngrams)
    return len(unique_ngrams)/total_ngrams if total_ngrams > 0 else 0.0


# NLI 
nli_pipeline = pipeline("text-classification", model="facebook/bart-large-mnli")

def factuality(premise: str, hypothesis: str) -> str:
    res = nli_pipeline([{"text": premise, "text_pair": hypothesis}], truncation=True)
    return res[0]["label"]


    



def evaluate_dataset(samples, max_items=10, verbose=True):
    buckets_strict = {
        "syn_vs_gt": {"cosine": [], "seq": [], "bert": [], "rougeL": [], "meteor": [], "factuality": []},
        "syn_vs_orig": {"cosine": [], "seq": [], "bert": []},
        "ant_vs_orig": {"cosine": [], "seq": [], "bert": []},
    }

    buckets_robust = {
        "syn_vs_gt": {"cosine": [], "seq": [], "bert": [], "rougeL": [], "meteor": [], "factuality": []},
        "syn_vs_orig": {"cosine": [], "seq": [], "bert": []},
        "ant_vs_orig": {"cosine": [], "seq": [], "bert": []},
    }

    for i, sample in enumerate(samples[:max_items]):
        sample_hash = compute_sample_hash(sample)

        # -------- cache with hash check --------
        if i in stored_results and stored_results[i].get("hash") == sample_hash:
            res = stored_results[i]["result"]
            if verbose:
                print(f"[INFO] Sample {i} unchanged, loaded from cache.")
        else:
            res = process_sample(sample)
            stored_results[i] = {"result": res, "hash": sample_hash}
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(stored_results, f)
            if verbose:
                print(f"[INFO] Sample {i} processed and cache updated.")

        if verbose:
            print(f"\n=== Sample {i} ===")
            print("Input:", sample["input"])
            print("Output:", sample["output"])
            print("Influential Words:", res["influential_words"])
            print("\n--- Counterfactuals ---")
            for cf in res["counterfactuals"]:
                print(cf["text"])

        # -------- Robust (all outputs) --------
        for cf in res["counterfactuals"]:
            if cf["type"] == "syn":
                c, s, b = _triplet_scores(cf["final_advice"], sample["output"])
                buckets_robust["syn_vs_gt"]["cosine"].append(c)
                buckets_robust["syn_vs_gt"]["seq"].append(s)
                buckets_robust["syn_vs_gt"]["bert"].append(b)

                # --- New metrics ---
                r = rouge_l(cf["final_advice"], sample["output"])
                m = meteor(cf["final_advice"], sample["output"])
                f = factuality(sample["output"], cf["final_advice"])

                buckets_robust["syn_vs_gt"]["rougeL"].append(r)
                buckets_robust["syn_vs_gt"]["meteor"].append(m)
                buckets_robust["syn_vs_gt"]["factuality"].append(f)

                c2, s2, b2 = _triplet_scores(cf["modified_input"], sample["input"])
                buckets_robust["syn_vs_orig"]["cosine"].append(c2)
                buckets_robust["syn_vs_orig"]["seq"].append(s2)
                buckets_robust["syn_vs_orig"]["bert"].append(b2)

            elif cf["type"] == "ant":
                c3, s3, b3 = _triplet_scores(cf["final_advice"], sample["output"])
                buckets_robust["ant_vs_orig"]["cosine"].append(c3)
                buckets_robust["ant_vs_orig"]["seq"].append(s3)
                buckets_robust["ant_vs_orig"]["bert"].append(b3)

        # -------- Strict (only valid, not fallback) --------
        for cf in res["counterfactuals"]:
            if cf["valid"] and "fallback" not in cf["final_advice"].lower():
                if cf["type"] == "syn":
                    c, s, b = _triplet_scores(cf["final_advice"], sample["output"])
                    buckets_strict["syn_vs_gt"]["cosine"].append(c)
                    buckets_strict["syn_vs_gt"]["seq"].append(s)
                    buckets_strict["syn_vs_gt"]["bert"].append(b)

                    # --- New metrics ---
                    r = rouge_l(cf["final_advice"], sample["output"])
                    m = meteor(cf["final_advice"], sample["output"])
                    f = factuality(sample["output"], cf["final_advice"])

                    buckets_strict["syn_vs_gt"]["rougeL"].append(r)
                    buckets_strict["syn_vs_gt"]["meteor"].append(m)
                    buckets_strict["syn_vs_gt"]["factuality"].append(f)

                    c2, s2, b2 = _triplet_scores(cf["modified_input"], sample["input"])
                    buckets_strict["syn_vs_orig"]["cosine"].append(c2)
                    buckets_strict["syn_vs_orig"]["seq"].append(s2)
                    buckets_strict["syn_vs_orig"]["bert"].append(b2)

                elif cf["type"] == "ant":
                    c3, s3, b3 = _triplet_scores(cf["final_advice"], sample["output"])
                    buckets_strict["ant_vs_orig"]["cosine"].append(c3)
                    buckets_strict["ant_vs_orig"]["seq"].append(s3)
                    buckets_strict["ant_vs_orig"]["bert"].append(b3)

    # -------- metrics --------
    def mean(lst):
        return float(np.mean(lst)) if lst else None

    def collect(buckets):
        final_metrics = {}
        for k, v in buckets.items():
            final_metrics[f"{k}_cosine"] = mean(v["cosine"])
            final_metrics[f"{k}_seq"] = mean(v["seq"])
            final_metrics[f"{k}_bert"] = mean(v["bert"])
            if "rougeL" in v:
                final_metrics[f"{k}_rougeL"] = mean(v["rougeL"])
            if "meteor" in v:
                final_metrics[f"{k}_meteor"] = mean(v["meteor"])
            if "factuality" in v:
                # factuality is categorical -> majority vote (or distribution)
                vals = v["factuality"]
                if vals:
                    final_metrics[f"{k}_factuality"] = max(set(vals), key=vals.count)
                else:
                    final_metrics[f"{k}_factuality"] = None
        return final_metrics

    return {
        "strict": collect(buckets_strict),
        "robust": collect(buckets_robust),
    }



def pretty_print_metrics(results):
    print("\n================ METRICS ================")
    for mode, m in results.items():
        print(f"\n--- {mode.upper()} ---")

        def fmt(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else str(x)

        print(
            f"syn_vs_gt   -> TFIDF: {fmt(m.get('syn_vs_gt_cosine'))} "
            f"| SEQ: {fmt(m.get('syn_vs_gt_seq'))} "
            f"| BERT: {fmt(m.get('syn_vs_gt_bert'))} "
            f"| ROUGE-L: {fmt(m.get('syn_vs_gt_rougeL'))} "
            f"| METEOR: {fmt(m.get('syn_vs_gt_meteor'))} "
            f"| FACT: {fmt(m.get('syn_vs_gt_factuality'))}"
        )
        print(
            f"syn_vs_orig -> TFIDF: {fmt(m.get('syn_vs_orig_cosine'))} "
            f"| SEQ: {fmt(m.get('syn_vs_orig_seq'))} "
            f"| BERT: {fmt(m.get('syn_vs_orig_bert'))}"
        )
        print(
            f"ant_vs_orig -> TFIDF: {fmt(m.get('ant_vs_orig_cosine'))} "
            f"| SEQ: {fmt(m.get('ant_vs_orig_seq'))} "
            f"| BERT: {fmt(m.get('ant_vs_orig_bert'))}"
        )
    print("=========================================\n")







# -----------------------------
# 7. Follow-up cache
# -----------------------------
FOLLOWUP_CACHE_FILE = "followup_results.pkl"
if os.path.exists(FOLLOWUP_CACHE_FILE):
    with open(FOLLOWUP_CACHE_FILE, "rb") as f:
        followup_cache = pickle.load(f)
   
else:
    followup_cache = {}
    














# -----------------------------
# 8. Interactive session
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

def interactive_session():
    print("\n=== Interactive Session (Counterfactual Chat) ===")
    while True:
        q = input("\nUser: ")
        if q.lower() in ["exit", "quit"]:
            break
        match = re.search(r"sample\s+(\d+)", q, re.IGNORECASE)
        if not match:
            print("[Error] Please specify a sample, e.g. 'In sample 58, ...'")
            continue
        sample_id = int(match.group(1))
        try:
            sample = cleaned_healthcaremagic[sample_id]
        except IndexError:
            print(f"[Error] Sample {sample_id} not found.")
            continue

        # ----- Hash check & cached processing -----
        if sample_id in stored_results:
            sample_hash = compute_sample_hash(sample)
            cached_hash = stored_results[sample_id].get("hash")
            if cached_hash == sample_hash:
                base_results = stored_results[sample_id]["data"]
                print("[INFO] Using cached results (hash match).")
            else:
                base_results = process_sample(sample)
                stored_results[sample_id] = {"hash": sample_hash, "data": base_results}
                with open(CACHE_FILE, "wb") as f:
                    pickle.dump(stored_results, f)
                print("[INFO] Sample has changed. Reprocessed and cache updated.")
        else:
            base_results = process_sample(sample)
            sample_hash = compute_sample_hash(sample)
            stored_results[sample_id] = {"hash": sample_hash, "data": base_results}
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(stored_results, f)
            print(f"[INFO] Sample {sample_id} processed and cached.")

        # ----- Follow-up interactive query -----
        followup_key = (sample_id, q)
        if followup_key in followup_cache:
            res_text = followup_cache[followup_key]
            print("[INFO] Using cached follow-up result.")
        else:
            res = followup_chain.invoke({
                "input": sample["input"],
                "output": sample["output"],
                "infl_words": base_results["influential_words"],
                "counterfactuals": "\n".join(cf["text"] for cf in base_results["counterfactuals"]),
                "user_query": q
            })
            res_text = res["text"]
            followup_cache[followup_key] = res_text
            with open(FOLLOWUP_CACHE_FILE, "wb") as f:
                pickle.dump(followup_cache, f)

        print("\n" + res_text)
        print("\n===========================================\n")









# -----------------------------
# 9. Example Run
# -----------------------------
results_metrics = evaluate_dataset(cleaned_healthcaremagic, max_items=100)
pretty_print_metrics(results_metrics)

if __name__ == "__main__":
    results_metrics = evaluate_dataset(cleaned_healthcaremagic, max_items=100)
    pretty_print_metrics(results_metrics)
    interactive_session()

    


# -----------------------------
#10.Visualization - METRICS
# -----------------------------


metrics_data = {
    "mode": ["STRICT"]*9 + ["ROBUST"]*9,
    "comparison": ["syn_vs_gt","syn_vs_gt","syn_vs_gt",
                   "syn_vs_orig","syn_vs_orig","syn_vs_orig",
                   "ant_vs_orig","ant_vs_orig","ant_vs_orig"]*2,
    "metric": ["cosine","seq","bert"]*6,
    "value": [0.2279,0.0592,0.5435,
              0.9632,0.9291,0.9772,
              0.1056,0.0323,0.4449,
              0.2094,0.0552,0.5157,
              0.9653,0.9331,0.9783,
              0.1056,0.0323,0.4449]
}

df = pd.DataFrame(metrics_data)

#seperate -> STRICT/ROBUST
df_strict = df[df['mode']=="STRICT"]
df_robust = df[df['mode']=="ROBUST"]



#Bar plot: TFIDF, SEQ, BERT per comparison
plt.figure(figsize=(8,4))
sns.barplot(x='comparison', y='value', hue='metric', data=df_strict)
plt.title("STRICT Similarities per Comparison")
plt.ylabel("Similarity Score")
plt.ylim(0,1)
plt.show()


#Heatmap
heat_df = df_strict.pivot(index="comparison", columns="metric", values="value")
plt.figure(figsize=(6,4))
sns.heatmap(heat_df, annot=True, cmap='YlOrRd', vmin=0, vmax=1)
plt.title("STRICT Similarities Heatmap")
plt.show()




#Boxplot (scatter) 
plt.figure(figsize=(6,4))
sns.boxplot(x='metric', y='value', data=df_strict)
plt.title("STRICT Similarities Distribution per Metric")
plt.ylim(0,1)
plt.show()



#Swarm plot : values per comparison
plt.figure(figsize=(8,4))
sns.swarmplot(x='comparison', y='value', hue='metric', data=df_strict, dodge=True)
plt.title("STRICT Similarities per Comparison (Swarm)")
plt.ylim(0,1)
plt.show()








# -----------------------------
#11.Error Analysis
# -----------------------------


#if not exist 'type' -> create with error types examples
if 'type' not in df.columns:
    types_list = ['fallback_advice', 'missing_symptom', 'hallucination',
                  'over_simplification', 'contradiction']
    df['type'] = (types_list * ((len(df) // len(types_list)) + 1))[:len(df)]

#error types 
error_types = [
    'fallback_advice',
    'missing_symptom',
    'hallucination',
    'over_simplification',
    'contradiction'
]



#For safe: normalize type strings
df['type_norm'] = df['type'].str.lower()

# filter only selected categories
df_error_types = df[df['type_norm'].isin(error_types)]

#Errors per type
error_counts = df_error_types['type_norm'].value_counts().reindex(error_types, fill_value=0)
colors = ['lightcoral', 'salmon', 'skyblue', 'orange', 'mediumseagreen']





# Horizontal bar plot
plt.figure(figsize=(8,5))
error_counts.plot(kind='barh', color=colors, edgecolor='black')
plt.xlabel('Number of errors')
plt.ylabel('Error Type')
plt.title('Error Analysis by Type')
plt.gca().invert_yaxis() 
plt.tight_layout()
plt.show()





#Error Type Distribution
plt.figure(figsize=(5,5))
type_counts.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'salmon'], startangle=90, wedgeprops={'edgecolor':'black'})
plt.title('Error Type Distribution', fontsize=12, weight='bold')
plt.ylabel('')
plt.tight_layout()
plt.show()



