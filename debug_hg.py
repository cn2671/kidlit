import pandas as pd, re, json
df = pd.read_csv("data/books_llm_tags.csv").fillna("")

def split_tags(s):
    s = str(s or "")
    if s.strip().startswith("[") and s.strip().endswith("]"):
        try: return [x.strip().lower() for x in json.loads(s) if isinstance(x,str)]
        except: pass
    return [t.strip().lower() for t in re.split(r"[;,/|]+", s) if t.strip()]

df["tones_norm"] = df["tone"].apply(split_tags)

# emulate strict tone filter
strict = df[df["tones_norm"].apply(lambda lst: "calm" in (lst or []))]
print("Rows with calm:", len(strict))
print(strict[["title","tones_norm","age_range"]].head(10).to_string(index=False))
