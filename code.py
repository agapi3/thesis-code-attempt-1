from langchain import LLMChain, PromptTemplate
from langchain_community.llms import Ollama
import re, ast, os, json, numpy as np
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch
import pickle

# -----------------------------
# Cache setup
# -----------------------------
CACHE_FILE = "stored_results.pkl"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        stored_results = pickle.load(f)
    #print(f"[INFO] Loaded {len(stored_results)} cached results.")
else:
    stored_results = {}
    #print("[INFO] No cache found, will compute from scratch.")






# -----------------------------
# 1. Define LLM
# -----------------------------
llm = Ollama(model="llama2", temperature=0)

# -----------------------------
# 2. Prompts
# -----------------------------

# Influential Prompt
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




# Synonym Prompt
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





# Antonym Prompt
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








# Counterfactual Prompt
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
#fine-tuned embeddings on medical text are more faithful to the medical meaning
#https://huggingface.co/pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb 
#count similarity between original and counterfactual text
bert_model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')

#TF-IDF cosine similarity
def cosine_bert(a: str, b: str) -> float:
    emb = bert_model.encode([a, b], convert_to_tensor=True)
    return float(torch.nn.functional.cosine_similarity(emb[0], emb[1], dim=0))

    
#sequence similarity
def _triplet_scores(a: str, b: str):
    return (cosine_tfidf(a, b), sequence_similarity(a, b), cosine_bert(a, b))


#BERT-based cosine similarity
def cosine_tfidf(a: str, b: str) -> float:
    vect = TfidfVectorizer().fit([a, b])
    vecs = vect.transform([a, b])
    return float((vecs[0] @ vecs[1].T).toarray()[0][0])











# -----------------------------
# 6. Follow-up cache
# -----------------------------
FOLLOWUP_CACHE_FILE = "followup_results.pkl"
if os.path.exists(FOLLOWUP_CACHE_FILE):
    with open(FOLLOWUP_CACHE_FILE, "rb") as f:
        followup_cache = pickle.load(f)
    #print(f"[INFO] Loaded {len(followup_cache)} cached follow-ups.")
else:
    followup_cache = {}
    #print("[INFO] No follow-up cache found, starting fresh.")














# -----------------------------
# Interactive session
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
        if sample_id in stored_results:
            base_results = stored_results[sample_id]
        else:
            base_results = process_sample(sample)
            stored_results[sample_id] = base_results
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(stored_results, f)
            print(f"[INFO] Sample {sample_id} processed and cached.")
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
# 7. Example Run
# -----------------------------
results_metrics = evaluate_dataset(cleaned_healthcaremagic, max_items=100)
pretty_print_metrics(results_metrics)

if __name__ == "__main__":
    results_metrics = evaluate_dataset(cleaned_healthcaremagic, max_items=100)
    pretty_print_metrics(results_metrics)
    interactive_session()



#8.Visualization - METRICS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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



# -----------------------------
#Bar plot: TFIDF, SEQ, BERT per comparison
# -----------------------------
plt.figure(figsize=(8,4))
sns.barplot(x='comparison', y='value', hue='metric', data=df_strict)
plt.title("STRICT Similarities per Comparison")
plt.ylabel("Similarity Score")
plt.ylim(0,1)
plt.show()


# -----------------------------
#Heatmap
# -----------------------------
heat_df = df_strict.pivot(index="comparison", columns="metric", values="value")
plt.figure(figsize=(6,4))
sns.heatmap(heat_df, annot=True, cmap='YlOrRd', vmin=0, vmax=1)
plt.title("STRICT Similarities Heatmap")
plt.show()



# -----------------------------
#Boxplot (scatter) 
# -----------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x='metric', y='value', data=df_strict)
plt.title("STRICT Similarities Distribution per Metric")
plt.ylim(0,1)
plt.show()



# -----------------------------
#Swarm plot : values per comparison
# -----------------------------
plt.figure(figsize=(8,4))
sns.swarmplot(x='comparison', y='value', hue='metric', data=df_strict, dodge=True)
plt.title("STRICT Similarities per Comparison (Swarm)")
plt.ylim(0,1)
plt.show()









# 9.Error Analysis
import pandas as pd
import matplotlib.pyplot as plt

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
