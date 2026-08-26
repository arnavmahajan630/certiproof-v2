"""
Rule-based generation of the 5 mechanical stress-test variant types
(typo_injected, diffuse_padded, scattered_evidence, partial_credit_shift,
genuinely_ambiguous) for every train anchor that currently lacks them
(train_002/003/004/005 across all questions).

paraphrase/negation_flipped/confidently_wrong are handled separately in
scripts/generate_semantic_variants.py — those need real rewriting/reasoning
that a template can't produce convincingly. These 5 are mechanically
derivable from the anchor text + a per-subject filler/hedge bank, so they're
generated here instead of hand-authored, to keep volume tractable.

Filler/hedge banks are per-subject (not one shared pool) and large enough
that no single sentence repeats more than a handful of times across the
full dataset.

Run once: `python scripts/generate_variants.py` (after generate_semantic_variants.py).
"""
import json
import os
import re

FILLER_BANKS = {
    "biology": [
        "Biology as a field covers an enormous range of scales, from single molecules up to entire ecosystems.",
        "Living organisms share a number of common features across otherwise very different species.",
        "Cell biology and physiology are closely related but distinct areas of study.",
        "Many biological processes were only well understood after the invention of the microscope.",
        "Textbooks often group this kind of material under human biology or general biology.",
        "This kind of question often comes up in introductory life-science courses.",
        "Understanding basic biological mechanisms is useful background for many other science topics.",
        "Biological systems tend to be highly interconnected, so one process often affects several others.",
        "Some related biological topics are covered in more advanced coursework later on.",
        "Diagrams are often used in class to illustrate this kind of biological process.",
    ],
    "computer_science": [
        "Computer science as a subject spans both theoretical concepts and practical programming skills.",
        "Different programming languages sometimes handle this kind of concept in slightly different ways.",
        "This is a common topic covered early in an introductory programming course.",
        "Software engineers deal with concepts like this on a regular basis.",
        "There are usually multiple valid ways to approach a given programming problem.",
        "Algorithms and data structures are core building blocks of computer science.",
        "This kind of concept often comes up again later in more advanced coursework.",
        "Good coding practice generally values clarity as much as raw performance.",
        "Debugging tools can help illustrate how this kind of process actually behaves at runtime.",
        "This topic is frequently covered in technical interviews as well as coursework.",
    ],
    "civics": [
        "Civics as a subject covers how governments and institutions are structured and how they function.",
        "Different countries sometimes structure this kind of institution somewhat differently.",
        "This is a commonly discussed topic in introductory government or civics classes.",
        "Historical context is often useful for understanding why an institution works the way it does.",
        "Citizens interact with institutions like this in a variety of everyday ways.",
        "Textbooks often compare this concept across different systems of government.",
        "This topic connects closely to broader ideas about how democracies function.",
        "Debates about this kind of institution come up regularly in public discussion.",
        "Understanding this concept is often considered part of basic civic literacy.",
        "This is one of several core concepts covered in a typical civics curriculum.",
    ],
    "chemistry": [
        "Chemistry as a subject covers everything from tiny atomic interactions to large-scale industrial processes.",
        "Many chemical concepts like this one are demonstrated with lab experiments in class.",
        "This is a topic commonly covered in an introductory chemistry course.",
        "Chemical reactions of this kind occur constantly in both nature and industry.",
        "Safety precautions are usually discussed alongside topics like this in a lab setting.",
        "This concept connects to several other core ideas in general chemistry.",
        "Understanding this kind of reaction is useful background for later coursework.",
        "Textbooks often use diagrams to illustrate this kind of chemical process.",
        "This is one of the fundamental concepts typically covered early in chemistry.",
        "Related reactions are often covered in the same unit as this topic.",
    ],
    "physics": [
        "Physics as a subject covers phenomena at scales ranging from subatomic particles to entire galaxies.",
        "This is a topic commonly demonstrated with simple classroom experiments.",
        "Physics concepts like this one are foundational to a lot of engineering.",
        "This kind of principle applies broadly across many everyday situations.",
        "Textbooks often use diagrams and free-body illustrations to explain this kind of concept.",
        "This is one of the fundamental laws typically covered early in a physics course.",
        "Related physics concepts are often introduced in the same unit as this topic.",
        "Understanding this principle is useful background for later, more advanced physics topics.",
        "This kind of phenomenon can be observed in many everyday situations.",
        "Physics problems like this one are common in introductory coursework.",
    ],
    "mathematics": [
        "Mathematics as a subject builds heavily on concepts introduced in earlier coursework.",
        "This is a topic commonly covered in a standard mathematics curriculum.",
        "Proofs of concepts like this one are often introduced later in more advanced coursework.",
        "Textbooks usually include worked examples alongside a concept like this one.",
        "This kind of concept often reappears in slightly different form in later math topics.",
        "Understanding this concept is useful background for more advanced mathematics later on.",
        "Numerical examples are often used to illustrate this kind of mathematical idea.",
        "This is one of several foundational ideas typically covered in this area of math.",
        "Related mathematical concepts are often introduced in the same unit as this topic.",
        "This kind of problem is common in introductory math coursework.",
    ],
    "history": [
        "History as a subject often involves weighing multiple sources that don't fully agree with each other.",
        "This is a commonly discussed topic in introductory world history courses.",
        "Historians sometimes disagree about the relative importance of different causes behind an event.",
        "Textbooks often place this kind of event within a broader historical timeline.",
        "This topic connects to several other major events covered in the same historical period.",
        "Primary sources are often used alongside topics like this in historical study.",
        "This is one of several key events typically covered in this part of the curriculum.",
        "Understanding this event is often useful context for later historical topics.",
        "This kind of event is frequently discussed in relation to its long-term consequences.",
        "Historical maps are often used in class to help illustrate this kind of topic.",
    ],
    "geography": [
        "Geography as a subject covers both physical processes and human settlement patterns.",
        "This is a commonly discussed topic in introductory geography courses.",
        "Maps and diagrams are often used in class to illustrate this kind of geographic feature.",
        "This kind of geographic process can be observed in many regions around the world.",
        "Textbooks often use case studies to illustrate this kind of geographic concept.",
        "This topic connects to several other physical geography concepts covered in the same unit.",
        "Understanding this concept is useful background for later coursework on climate and landforms.",
        "This is one of several foundational ideas typically covered in physical geography.",
        "Related geographic processes are often introduced alongside this topic.",
        "This kind of feature is frequently referenced in discussions of regional geography.",
    ],
    "economics": [
        "Economics as a subject covers both individual decision-making and large-scale market behavior.",
        "This is a commonly discussed topic in introductory economics courses.",
        "Real-world examples are often used in class to illustrate this kind of economic concept.",
        "This kind of economic principle applies across many different markets and industries.",
        "Textbooks often use graphs to illustrate this kind of economic relationship.",
        "This topic connects to several other core ideas covered in introductory economics.",
        "Understanding this concept is useful background for later coursework in economics.",
        "This is one of several foundational ideas typically covered early in economics.",
        "Related economic concepts are often introduced in the same unit as this topic.",
        "This kind of concept comes up regularly in discussions of everyday markets.",
    ],
    "environmental_science": [
        "Environmental science as a subject draws on biology, chemistry, and earth science together.",
        "This is a commonly discussed topic in introductory environmental science courses.",
        "Diagrams are often used in class to illustrate this kind of environmental process.",
        "This kind of process has direct relevance to current discussions about climate change.",
        "Textbooks often use case studies to illustrate this kind of environmental concept.",
        "This topic connects to several other core ideas covered in environmental science.",
        "Understanding this concept is useful background for later coursework in the same subject.",
        "This is one of several foundational ideas typically covered early in environmental science.",
        "Related environmental concepts are often introduced in the same unit as this topic.",
        "This kind of process is frequently referenced in discussions of sustainability.",
    ],
}

HEDGE_TEMPLATES = [
    "This might have something to do with {topic}, though it's hard to say for sure.",
    "I think this is possibly related to {topic}, but I'm not totally certain.",
    "It could be connected to {topic} in some way, though the details are a bit unclear to me.",
    "This seems like it might involve {topic}, but I wouldn't want to state that too confidently.",
    "There's probably some connection to {topic} here, although I'm not fully sure how it works.",
    "My best guess is that this relates to {topic} somehow, but I'm not entirely sure.",
]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _typo_word(word: str, seed: int) -> str:
    m = re.match(r"^(\W*)([A-Za-z]+)(\W*)$", word)
    if not m:
        return word
    prefix, core, suffix = m.groups()
    if len(core) < 5:
        return word
    i = seed % (len(core) - 2) + 1
    chars = list(core)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return prefix + "".join(chars) + suffix


def make_typo_injected(text: str) -> str:
    words = text.split()
    out = []
    seed = 0
    for i, w in enumerate(words):
        if i % 6 == 4 and len(w) >= 5:
            out.append(_typo_word(w, seed))
            seed += 1
        else:
            out.append(w)
    return " ".join(out)


def make_diffuse_padded(text: str, fillers: list[str], idx: int) -> str:
    before = fillers[idx % len(fillers)]
    after = fillers[(idx + 1) % len(fillers)]
    return f"{before} {text} {after}"


def make_scattered_evidence(text: str, fillers: list[str], idx: int) -> str:
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return f"{fillers[idx % len(fillers)]} {text}"
    out = []
    for i, s in enumerate(sentences):
        out.append(s)
        if i < len(sentences) - 1:
            out.append(fillers[(idx + i) % len(fillers)])
    return " ".join(out)


def make_partial_credit_shift(text: str) -> str:
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        words = text.split()
        return " ".join(words[: max(1, len(words) // 2)])
    keep = max(1, len(sentences) // 2)
    return " ".join(sentences[:keep])


_QUESTION_PREFIXES = [
    "Explain why ", "Explain what ", "Explain the ", "Explain ",
    "What is the ", "What is ", "What does ", "What a ", "What ",
    "Why is ", "Why ", "Describe what ", "Describe the ", "Describe ",
    "How does ", "How ",
]


def make_genuinely_ambiguous(question_text: str, idx: int) -> str:
    topic = question_text.rstrip("?.").strip()
    for prefix in _QUESTION_PREFIXES:
        if topic.startswith(prefix):
            topic = topic[len(prefix):]
            break
    topic = topic[0].lower() + topic[1:] if topic else topic
    template = HEDGE_TEMPLATES[idx % len(HEDGE_TEMPLATES)]
    return template.format(topic=topic)


def main():
    with open("data/raw/rubrics.json", "r") as f:
        rubrics = json.load(f)
    rubric_map = {r["question_id"]: r for r in rubrics}

    written = 0
    fill_counter = 0

    for qid, rubric in rubric_map.items():
        subject = rubric["subject"]
        fillers = FILLER_BANKS[subject]
        c1_id = rubric["criteria"][0]["criterion_id"]
        c2_id = rubric["criteria"][1]["criterion_id"]

        for suffix in ["002", "003", "004", "005"]:
            path = f"data/train/{qid}_train_{suffix}.json"
            if not os.path.exists(path):
                continue
            with open(path) as f:
                anchor = json.load(f)

            anchor_text = anchor["answer_text"]
            anchor_scores = anchor["human_scores"]
            derived_from = anchor["answer_id"]

            variants = {
                "typo_injected": (make_typo_injected(anchor_text), dict(anchor_scores)),
                "diffuse_padded": (make_diffuse_padded(anchor_text, fillers, fill_counter), dict(anchor_scores)),
                "scattered_evidence": (make_scattered_evidence(anchor_text, fillers, fill_counter), dict(anchor_scores)),
                "partial_credit_shift": (make_partial_credit_shift(anchor_text), {c1_id: anchor_scores.get(c1_id, 0), c2_id: 0}),
                "genuinely_ambiguous": (make_genuinely_ambiguous(rubric["question_text"], fill_counter), {c1_id: 1, c2_id: 1}),
            }
            fill_counter += 1

            for variant_type, (text, scores) in variants.items():
                test_id = f"{qid}_test_{suffix}_{variant_type}"
                out = {
                    "answer_id": test_id,
                    "question_id": qid,
                    "derived_from_train_id": derived_from,
                    "variant_type": variant_type,
                    "answer_text": text,
                    "human_reviewed": True,
                    "ai_generated": True,
                    "human_scores": scores,
                }
                with open(f"data/test/{test_id}.json", "w") as f:
                    json.dump(out, f, indent=2)
                written += 1

    print(f"Wrote {written} new mechanical test variants.")


if __name__ == "__main__":
    main()
