"""Create synthetic grammaticality data from clean Swedish sentences.

Each clean sentence becomes one "good" example and one "poor" example with 1-3
injected errors. Good and poor copies share a `group` id so they can be kept in
the same train/validation split.

Usage:
    python data/generate.py --input clean.txt --output data/synthetic.jsonl
    (clean.txt: one sentence or short text per line)
"""
import argparse
import json
import random
import re

SEED = 0
PRON = {"jag", "du", "han", "hon", "vi", "ni", "de", "man", "det"}
FRONT = {"Igår", "I", "Efter", "Nästa", "Under", "Förra", "Varje", "Trots", "Enligt", "Om", "När", "Eftersom"}
MODALS = {"kan", "ska", "måste", "vill", "brukar", "bör", "får", "kunde", "skulle", "ville"}
NEG_ADV = {"inte", "aldrig", "alltid", "ofta", "fortfarande"}
PREPS = ["på", "i", "av", "för", "om", "med", "till", "från"]
SKIP = {"är", "var", "har", "efter", "under", "mer", "för", "över", "inför", "ner", "ur", "ibland", "gärna", "mycket"}
SWAPS = [("en", "ett"), ("min", "mitt"), ("din", "ditt"), ("sin", "sitt"), ("någon", "något"),
         ("ingen", "inget"), ("vår", "vårt"), ("de", "dem")]
SWAP_MAP = {a: b for a, b in SWAPS} | {b: a for a, b in SWAPS}


def parts(tok):
    m = re.match(r"^(\W*)(\w[\w-]*)(\W*)$", tok)
    return m.groups() if m else ("", tok, "")


def put(toks, i, core):
    pre, old, post = parts(toks[i])
    if old[:1].isupper():
        core = core[:1].upper() + core[1:]
    toks[i] = pre + core + post


def core(toks, i):
    return parts(toks[i])[1]


# Each rule takes (tokens, rng) and returns (new_tokens, description) or None.
def swap_determiner(t, r):
    idx = [i for i in range(len(t)) if core(t, i).lower() in SWAP_MAP and i + 1 < len(t)]
    if not idx:
        return None
    i = r.choice(idx)
    w = core(t, i)
    new = SWAP_MAP[w.lower()]
    put(t, i, new)
    return t, f"determiner/pronoun form ('{w}' -> '{new}')"


def drop_definite(t, r):
    idx = [i for i in range(1, len(t)) if re.search(r"(en|et|an)$", core(t, i)) and len(core(t, i)) >= 5
           and core(t, i - 1).lower() in {"den", "det", "i", "på", "av", "med", "om", "till", "gamla", "nya"}]
    if not idx:
        return None
    i = r.choice(idx)
    w = core(t, i)
    put(t, i, w[:-2] if w.endswith(("en", "et")) else w[:-2])
    return t, f"dropped definite suffix ('{w}')"


def v2_violation(t, r):
    if core(t, 0) not in FRONT:
        return None
    for i in range(1, min(len(t) - 1, 5)):
        if core(t, i + 1).lower() in PRON and core(t, i).lower() not in PRON | SKIP:
            t[i], t[i + 1] = t[i + 1], t[i]
            return t, "verb-second word order violated"
    return None


def neg_placement(t, r):
    idx = [i for i in range(2, len(t)) if core(t, i).lower() in NEG_ADV]
    if not idx:
        return None
    i = r.choice(idx)
    t[i - 1], t[i] = t[i], t[i - 1]
    return t, "adverb/negation placement"


VERB_END = ("ade", "ades", "ar", "ade", "er", "de", "des", "as", "ande", "ende", "tes", "era", "erar")


def split_compound(t, r):
    idx = [i for i in range(len(t)) if len(core(t, i)) >= 10 and core(t, i).isalpha()
           and core(t, i).islower() and not core(t, i).endswith(VERB_END)]
    if not idx or r.random() < 0.6:  # this rule overfires; thin it out
        return None
    i = r.choice(idx)
    w = core(t, i)
    k = r.randint(4, len(w) - 4)
    put(t, i, w[:k] + " " + w[k:])
    return t, f"compound split ('{w}')"


def verb_form(t, r):
    idx = [i for i in range(len(t) - 1) if core(t, i).lower() in MODALS and core(t, i + 1).endswith("a")
           and len(core(t, i + 1)) >= 4]
    if not idx:
        return None
    i = r.choice(idx) + 1
    w = core(t, i)
    put(t, i, w + "r")
    return t, f"wrong verb form ('{w}' -> '{w}r')"


PLURAL_CTX = {"flera", "många", "några", "dessa", "alla", "två", "tre", "fyra", "fem", "sex", "sju",
              "åtta", "nio", "tio", "de", "stora", "nya", "andra", "olika", "sina", "mina", "dina", "våra"}


def plural_form(t, r):
    idx = [i for i in range(1, len(t)) if re.search(r"(or|ar|er)$", core(t, i)) and len(core(t, i)) >= 6
           and core(t, i).islower() and core(t, i - 1).lower() in PLURAL_CTX]
    if not idx:
        return None
    i = r.choice(idx)
    w = core(t, i)
    new = w[:-2] + r.choice([s for s in ("or", "ar", "er") if s != w[-2:]])
    put(t, i, new)
    return t, f"wrong plural form ('{w}' -> '{new}')"


def prep_swap(t, r):
    idx = [i for i in range(len(t)) if core(t, i) in PREPS]  # lowercase-only: dodges roman numerals / proper nouns
    if not idx:
        return None
    i = r.choice(idx)
    w = core(t, i)
    put(t, i, r.choice([p for p in PREPS if p != w.lower()]))
    return t, f"wrong preposition ('{w}')"


def drop_subject(t, r):
    if len(t) > 2 and core(t, 0) == "Det" and re.search(r"(ar|nar)$", core(t, 1)) and len(core(t, 1)) >= 5:
        t = t[1:]
        put(t, 0, core(t, 0).capitalize())
        return t, "missing formal subject"
    return None


def adj_agreement(t, r):
    idx = [i for i in range(1, len(t)) if core(t, i - 1).lower() in {"är", "var", "blev", "verkar"}
           and core(t, i).endswith("t") and len(core(t, i)) >= 5 and core(t, i).lower() not in SKIP | {"inte", "också"}]
    if not idx:
        return None
    i = r.choice(idx)
    w = core(t, i)
    put(t, i, w[:-1])
    return t, f"adjective agreement ('{w}')"


RULES = [swap_determiner, drop_definite, v2_violation, neg_placement, split_compound,
         verb_form, plural_form, prep_swap, drop_subject, adj_agreement]


def corrupt(text, rng):
    """Return (corrupted_text, score, error_description) or None if nothing applied."""
    if rng.random() < 0.03:  # rare word-salad example
        t = text.split()
        if len(t) >= 5:
            rng.shuffle(t)
            return " ".join(t), rng.choice([0.0, 0.05]), "word salad, shuffled tokens"
    toks, errors = text.split(), []
    for rule in rng.sample(RULES, len(RULES)):
        if len(errors) >= rng.choices([1, 2, 3], [0.6, 0.3, 0.1])[0]:
            break
        res = rule(list(toks), rng)
        if res and " ".join(res[0]) != " ".join(toks):
            toks, desc = res
            errors.append(desc)
    if not errors:
        return None
    base = max(0.05, 0.45 - 0.17 * (len(errors) - 1))
    score = round(min(0.6, max(0.05, base + rng.choice([-0.1, -0.05, 0, 0, 0.05]))) * 20) / 20
    return " ".join(toks), score, "; ".join(errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="text file, one clean Swedish text per line")
    ap.add_argument("--output", required=True)
    ap.add_argument("--start-id", type=int, default=10000)
    args = ap.parse_args()
    rng = random.Random(SEED)
    texts = [l.strip() for l in open(args.input, encoding="utf-8") if l.strip()]
    n, uid = 0, args.start_id
    with open(args.output, "w", encoding="utf-8") as f:
        for j, (g, text) in enumerate(enumerate(texts, start=args.start_id)):
            if j % 2000 == 0:
                print(f"{j}/{len(texts)} lines processed, {n} pairs written", flush=True)
            res = corrupt(text, rng)
            if res is None:
                continue
            bad, score, err = res
            good_score = rng.choices([0.95, 1.0, 0.9], [0.7, 0.2, 0.1])[0]
            for text_, s, lab, e in ((text, good_score, "good", None), (bad, score, "poor", err)):
                f.write(json.dumps({"id": uid, "text": text_, "score": s, "label": lab, "error": e, "group": g},
                                   ensure_ascii=False) + "\n")
                uid += 1
            n += 1
    print(f"Wrote {n} good/poor pairs to {args.output}")


if __name__ == "__main__":
    main()
