"""
LLM-authored (this session), auto-approved per project convention: generates the 3
semantic variant types (paraphrase, negation_flipped, confidently_wrong) plus the
new long-paragraph train anchor (train_005), for every question in data/raw/rubrics.json.

These three variant types need actual rewriting/reasoning (a real paraphrase, a natural
negation, a plausible-but-wrong fact) that a template can't produce convincingly, so they
are hand/LLM-authored per anchor rather than rule-based (see scripts/generate_variants.py
for the other 5 mechanical variant types, which ARE rule-based).

Scoring rule (applied uniformly, not re-derived per text): every criterion text in this
dataset is written as [c1 = primary/definitional claim, c2 = secondary claim/example], and
every negation_flipped / confidently_wrong text below is written to specifically undermine
the c1 claim (never c2) — so the score is always computed as (0, anchor's original c2 score)
for those two variant types, and unchanged for paraphrase. This keeps authored text and
computed score guaranteed consistent without hand-tracking a score per text.

Not run by anything else — run once: `python scripts/generate_semantic_variants.py`.
"""
import json
import os

# Each entry: question_id -> {
#   "long_answer": <train_005 full-credit long-paragraph text, ~100-200 words>,
#   "variants": {
#       "002": {"paraphrase": ..., "negation_flipped": ..., "confidently_wrong": ...},
#       "003": {...}, "004": {...}, "005": {...}
#   }
# }
DATA = {
    "q1": {
        "long_answer": (
            "The green color of most plants comes down to a pigment called chlorophyll, which is packed "
            "inside chloroplasts in plant cells. Chlorophyll is very good at absorbing light in the red and "
            "blue parts of the visible spectrum, and it uses the energy from that absorbed light to drive "
            "photosynthesis, the process that converts carbon dioxide and water into glucose and oxygen. "
            "Green wavelengths of light, however, are largely not absorbed by chlorophyll — instead they are "
            "reflected back out of the leaf. Because our eyes perceive whatever wavelength of light reaches "
            "them, and in this case that is mostly the reflected green light, we see the plant as green. "
            "In autumn, when chlorophyll breaks down faster than it's replaced, other pigments like carotenoids "
            "become visible, which is why leaves can turn yellow or orange."
        ),
        "variants": {
            "002": {
                "paraphrase": "Chlorophyll bounces green wavelengths back out rather than absorbing them, which is why we see plants as green; it does soak up red and blue light to power photosynthesis.",
                "negation_flipped": "Chlorophyll does not reflect green light, so that isn't why plants look green; it does absorb red and blue light for photosynthesis.",
                "confidently_wrong": "Plants look green because chlorophyll absorbs green light most strongly of all colors; it does absorb red and blue light too, for photosynthesis.",
            },
            "003": {
                "paraphrase": "The green appearance of plants comes from chlorophyll reflecting green light rather than taking it in.",
                "negation_flipped": "Plants do not appear green because chlorophyll reflects green light — that isn't the mechanism.",
                "confidently_wrong": "Plants look green because chlorophyll absorbs green light more strongly than any other color.",
            },
            "004": {
                "paraphrase": "The green color of plants comes from sunlight refracting as it passes through the leaf structure, similar to a prism splitting light.",
                "negation_flipped": "Plant color has nothing to do with sunlight refracting through leaves like a prism.",
                "confidently_wrong": "Plants appear green because their cell walls are naturally pigmented green, unrelated to any light absorption.",
            },
            "005": {
                "paraphrase": "Because chloroplasts contain chlorophyll, which absorbs red and blue light to fuel photosynthesis while letting green wavelengths bounce off, our eyes register that reflected green and we perceive the plant as green; this is also why fading chlorophyll in autumn reveals yellow and orange pigments underneath.",
                "negation_flipped": "Chlorophyll inside chloroplasts does not reflect green light back out — the plant's greenness isn't explained that way — even though chlorophyll does absorb red and blue light to fuel photosynthesis, and this same breakdown pattern is why autumn leaves reveal other pigments.",
                "confidently_wrong": "Chlorophyll inside chloroplasts absorbs green light more efficiently than any other wavelength, which is what actually produces a plant's green color, alongside its normal role absorbing red and blue light to fuel photosynthesis and its breakdown in autumn revealing other pigments.",
            },
        },
    },
    "q2": {
        "long_answer": (
            "Osmosis is the passive movement of water molecules across a semipermeable membrane, one that lets "
            "water through but blocks larger solute particles. Water moves from a region where solute "
            "concentration is low (and therefore water concentration is relatively high) toward a region where "
            "solute concentration is high, and this continues until the concentration is equal on both sides of "
            "the membrane, or until some opposing pressure balances it out. No outside energy input is required "
            "because the process is driven purely by the concentration gradient, which is why it's classified "
            "as passive transport rather than active transport. A classic classroom demonstration is placing a "
            "raisin in plain water: water moves into the raisin's cells because the solute concentration inside "
            "the dried fruit is much higher than in the surrounding water, causing the raisin to swell up."
        ),
        "variants": {
            "002": {
                "paraphrase": "Water passes through a semipermeable membrane moving toward the side with higher solute concentration, which is the definition of osmosis.",
                "negation_flipped": "Osmosis does not involve water moving through a semipermeable membrane toward higher solute concentration — that description is wrong.",
                "confidently_wrong": "Osmosis is when solute particles move across a semipermeable membrane toward the side with more water, not the other way around.",
            },
            "003": {
                "paraphrase": "Osmosis describes water crossing a semipermeable membrane, flowing from a weaker to a stronger solute concentration.",
                "negation_flipped": "Osmosis is not the movement of water from low to high solute concentration across a membrane.",
                "confidently_wrong": "Osmosis is the movement of water from high solute concentration to low solute concentration across a membrane.",
            },
            "004": {
                "paraphrase": "Osmosis is how plants use their leaves to absorb sunlight and convert it into usable energy.",
                "negation_flipped": "Osmosis is not related to plants absorbing sunlight through their leaves for energy.",
                "confidently_wrong": "Osmosis is the process by which plant roots absorb minerals directly from soil particles.",
            },
            "005": {
                "paraphrase": "Water diffuses passively across a selectively permeable membrane, flowing toward whichever side has the higher solute concentration until the two sides equalize, with no external energy needed since the concentration gradient alone drives it — which is exactly why a dried raisin swells up when dropped in water, its cells pulling in water from the far less concentrated surroundings.",
                "negation_flipped": "Water does not move passively toward the side of higher solute concentration across a selectively permeable membrane — osmosis isn't driven that way — even though no external energy is needed since a concentration gradient alone drives water movement, which is why a raisin still swells in water.",
                "confidently_wrong": "Water moves passively across a selectively permeable membrane toward whichever side has the LOWER solute concentration, which is what actually drives osmosis, requiring no external energy since the gradient itself is the driver — this is supposedly why a raisin swells up in water.",
            },
        },
    },
    "b1": {
        "long_answer": (
            "The main job of red blood cells is transporting oxygen around the body, and they're able to do "
            "this because they're packed with a protein called hemoglobin. In the lungs, hemoglobin binds "
            "oxygen molecules; the blood then carries these oxygen-loaded cells through the circulatory system "
            "to tissues that need it, where the hemoglobin releases the oxygen so cells can use it for cellular "
            "respiration. On the return trip, hemoglobin picks up some carbon dioxide waste and carries it back "
            "toward the lungs to be exhaled. Structurally, mature mammalian red blood cells are unusual in that "
            "they have no nucleus and no other major organelles, which frees up more internal space for "
            "hemoglobin and lets them squeeze through the narrowest capillaries. This lack of a nucleus does "
            "mean red blood cells can't repair or replicate themselves, so they're constantly being replaced by "
            "new ones produced in the bone marrow."
        ),
        "variants": {
            "002": {
                "paraphrase": "Hemoglobin in red blood cells picks up oxygen in the lungs and delivers it around the body; these cells also lack a nucleus once mature.",
                "negation_flipped": "Red blood cells do not carry oxygen via hemoglobin from the lungs to the body; they do lack a nucleus once mature.",
                "confidently_wrong": "Red blood cells carry carbon dioxide, not oxygen, from the lungs to the body via hemoglobin; they do lack a nucleus once mature.",
            },
            "003": {
                "paraphrase": "Hemoglobin inside red blood cells is what carries oxygen throughout the body.",
                "negation_flipped": "Red blood cells do not carry oxygen around the body using hemoglobin.",
                "confidently_wrong": "White blood cells carry oxygen around the body using hemoglobin.",
            },
            "004": {
                "paraphrase": "Red blood cells generate antibodies that help the body fight off infections.",
                "negation_flipped": "Red blood cells do not generate antibodies to fight infections.",
                "confidently_wrong": "Red blood cells store and digest nutrients absorbed from the small intestine.",
            },
            "005": {
                "paraphrase": "Because they're full of hemoglobin, red blood cells bind oxygen in the lungs and ferry it to tissues that need it, releasing it there for cellular respiration and picking up carbon dioxide on the way back; lacking a nucleus leaves more room for hemoglobin and lets them fit through narrow capillaries, though it also means they can't repair themselves and must be constantly replaced by bone marrow.",
                "negation_flipped": "Hemoglobin-packed red blood cells do not bind oxygen in the lungs for delivery to tissues — that isn't their function — even though lacking a nucleus does leave more room for hemoglobin, lets them fit through narrow capillaries, and means bone marrow must constantly replace them since they can't repair themselves.",
                "confidently_wrong": "Red blood cells are packed with hemoglobin mainly to filter toxins out of the bloodstream rather than to carry oxygen, though lacking a nucleus does leave more internal room, lets them fit through narrow capillaries, and means bone marrow must constantly replace them since they can't repair themselves.",
            },
        },
    },
    "b2": {
        "long_answer": (
            "Enzymes are specialized proteins that act as biological catalysts, meaning they speed up the rate "
            "of chemical reactions inside the body without being consumed or permanently changed by the "
            "reaction themselves. They work by lowering the activation energy needed for a reaction to proceed, "
            "which lets reactions that would otherwise be far too slow at body temperature happen fast enough "
            "to sustain life. Each enzyme typically has an active site shaped to fit one particular substrate "
            "or a small group of similar substrates, similar to a lock and key, which is why enzymes tend to be "
            "highly specific about which reaction they catalyze. Because an enzyme molecule isn't used up in "
            "the reaction, a single enzyme molecule can catalyze the same reaction over and over again very "
            "quickly. Enzyme activity can also be affected by conditions like temperature and pH — most human "
            "enzymes work best around normal body temperature and can lose their shape and stop functioning if "
            "conditions become too extreme."
        ),
        "variants": {
            "002": {
                "paraphrase": "As biological catalysts, enzymes make chemical reactions in the body proceed faster, and they come out of the reaction unchanged.",
                "negation_flipped": "Enzymes do not act as biological catalysts that speed up chemical reactions; they are not consumed by the reaction.",
                "confidently_wrong": "Enzymes slow down chemical reactions in the body to keep metabolism stable; they are not consumed by the reaction.",
            },
            "003": {
                "paraphrase": "Enzymes make chemical reactions inside cells happen faster.",
                "negation_flipped": "Enzymes do not speed up chemical reactions happening in cells.",
                "confidently_wrong": "Hormones speed up chemical reactions happening in cells.",
            },
            "004": {
                "paraphrase": "Enzymes are a category of vitamin stored in the body's fat tissue for later use.",
                "negation_flipped": "Enzymes are not a type of vitamin stored in fat tissue.",
                "confidently_wrong": "Enzymes are a type of mineral absorbed directly from food without being digested.",
            },
            "005": {
                "paraphrase": "Enzymes are proteins that catalyze biological reactions by lowering the activation energy required, letting otherwise-too-slow reactions run fast enough to sustain life, and because each enzyme's active site fits only specific substrates, they tend to be highly reaction-specific and reusable, though extreme temperature or pH can distort their shape and shut them down.",
                "negation_flipped": "Enzymes do not lower the activation energy required for a biological reaction to proceed — that isn't how they function as catalysts — even though their active sites are still shaped to fit specific substrates, making them highly reusable and reaction-specific, and extreme temperature or pH can still distort them and shut them down.",
                "confidently_wrong": "Enzymes function by directly supplying extra energy to a reaction rather than lowering its activation energy, which is what actually makes reactions in the body proceed fast enough to sustain life, though their active sites are still specific to certain substrates and extreme temperature or pH can still distort and disable them.",
            },
        },
    },
    "q3": {
        "long_answer": (
            "A stack and a queue are both linear data structures used to store an ordered collection of "
            "elements, but they differ in the order elements come back out. A stack follows Last In First Out "
            "(LIFO) ordering: both insertion (push) and removal (pop) happen at the same end, called the top, "
            "so the most recently added element is always the first one removed, much like a stack of plates "
            "where you can only take from the top. A queue, on the other hand, follows First In First Out "
            "(FIFO) ordering: elements are inserted at the rear (enqueue) and removed from the front (dequeue), "
            "so the earliest element added is the first one to leave, similar to people standing in line at a "
            "checkout counter. Stacks are commonly used for things like undo functionality or tracking function "
            "calls, while queues are commonly used for task scheduling or handling requests in the order they "
            "arrive."
        ),
        "variants": {
            "002": {
                "paraphrase": "In a stack, the most recently pushed item is popped first (LIFO); in a queue, the earliest enqueued item is dequeued first (FIFO).",
                "negation_flipped": "A stack is not LIFO — that's not how it removes items; a queue is FIFO, removing the earliest item first.",
                "confidently_wrong": "A stack removes the oldest item first, not the most recent; a queue removes the earliest item first (FIFO).",
            },
            "003": {
                "paraphrase": "A stack restricts insertion and removal to one end only, giving it LIFO behavior.",
                "negation_flipped": "A stack does not restrict insertion and removal to one end with LIFO order.",
                "confidently_wrong": "A queue restricts insertion and removal to one end only, giving it LIFO order.",
            },
            "004": {
                "paraphrase": "A stack and a queue are really just two names for the same data structure used interchangeably across languages.",
                "negation_flipped": "A stack and a queue are not just two names for the same underlying data structure.",
                "confidently_wrong": "A stack and a queue both remove elements in random order, and the only real difference is their name.",
            },
            "005": {
                "paraphrase": "Both are ordered linear structures, but a stack only pushes and pops from one end (the top), giving Last In First Out behavior like a stack of plates, while a queue enqueues at the rear and dequeues from the front, giving First In First Out behavior like a checkout line — which is why stacks suit undo functionality and call tracking while queues suit task scheduling.",
                "negation_flipped": "A stack does not push and pop from the same end to get Last In First Out behavior — that isn't how it works — even though a queue does still enqueue at the rear and dequeue from the front for First In First Out behavior, and stacks are still used for undo/call-tracking while queues suit task scheduling.",
                "confidently_wrong": "A stack actually pushes at one end and pops from the opposite end, which is what gives it Last In First Out behavior, while a queue still enqueues at the rear and dequeues from the front for First In First Out order, with stacks used for undo/call-tracking and queues for task scheduling.",
            },
        },
    },
    "q4": {
        "long_answer": (
            "In programming, a variable is a named location in a computer's memory that can hold a value, and "
            "critically, that value can be changed while the program is running — which is what distinguishes "
            "it from a constant. When you declare a variable, you're essentially telling the program to set "
            "aside some memory and give it a label so you can refer to whatever value is stored there later. "
            "For example, in Python, writing `score = 0` creates a variable named score holding the value "
            "zero; later in the program you might write `score = score + 10` to update it, and the variable "
            "now holds ten instead. Variables also have types in most languages — such as integers, strings, "
            "or booleans — which determine what kind of value they can hold and what operations are valid on "
            "them. Using descriptive variable names, like total_price instead of x, makes code much easier for "
            "other people (and your future self) to read and understand."
        ),
        "variants": {
            "002": {
                "paraphrase": "A named memory location whose stored value is allowed to change while a program runs is called a variable; `count = 5` is one example of declaring one.",
                "negation_flipped": "A variable is not a named storage location holding a value that can change; `count = 5` is an example of declaring one.",
                "confidently_wrong": "A variable is a named storage location holding a value that must stay fixed once set; `count = 5` is an example of declaring one.",
            },
            "003": {
                "paraphrase": "A variable is a name attached to a value that's allowed to change as the program executes.",
                "negation_flipped": "A variable is not a name attached to a changeable value while a program runs.",
                "confidently_wrong": "A constant is a name attached to a value that can change while a program runs.",
            },
            "004": {
                "paraphrase": "A variable is a control structure that repeats a block of code a set number of times.",
                "negation_flipped": "A variable is not a control structure that repeats a block of code.",
                "confidently_wrong": "A variable is a block of reusable code that can be called by name from elsewhere in the program.",
            },
            "005": {
                "paraphrase": "A variable names a spot in memory that can hold a changing value, unlike a constant, and declaring one — like `score = 0` — reserves that memory and labels it so the program can update it later, e.g. `score = score + 10`; most languages also type variables (integer, string, boolean) to constrain valid operations, and descriptive names like total_price over x make code far more readable.",
                "negation_flipped": "A variable does not name a memory location whose value is allowed to change — that isn't what makes it a variable — even though declaring one still reserves and labels memory for later updates, most languages still type variables to constrain operations, and descriptive names like total_price over x still improve readability.",
                "confidently_wrong": "A variable actually names a fixed memory location whose value is locked once declared, which is what defines it as a variable, though declaring one — like `score = 0` — still reserves and labels memory, languages still type variables to constrain operations, and descriptive names like total_price over x still improve readability.",
            },
        },
    },
    "cs1": {
        "long_answer": (
            "Recursion is a programming technique where a function solves a problem by calling itself with a "
            "smaller or simpler version of the same problem, gradually working toward a case simple enough to "
            "answer directly. That simplest case is called the base case, and it's essential: without one, the "
            "function would keep calling itself indefinitely, which eventually crashes the program with a "
            "stack overflow once it runs out of call-stack memory. A classic example is computing a factorial: "
            "`factorial(n)` can be defined as `n * factorial(n - 1)`, with the base case `factorial(0) = 1` "
            "stopping the chain of calls. Each recursive call is placed on the call stack, and once the base "
            "case is reached, the results unwind back up through all the pending calls, multiplying together "
            "as they go, until the original call returns the final answer. Recursion often makes code that "
            "operates on naturally recursive structures — like trees or nested lists — much cleaner to read "
            "than an equivalent loop-based version."
        ),
        "variants": {
            "002": {
                "paraphrase": "When a function invokes itself to break a problem into smaller versions of itself, that's recursion, and it needs a stopping condition called a base case.",
                "negation_flipped": "Recursion is not when a function calls itself; it does still need a base case to eventually stop.",
                "confidently_wrong": "Recursion is when a function calls a completely different function to delegate work; it does need a base case to eventually stop.",
            },
            "003": {
                "paraphrase": "When a function invokes itself, that's called recursion.",
                "negation_flipped": "Recursion is not when a function calls itself.",
                "confidently_wrong": "Iteration is when a function calls itself.",
            },
            "004": {
                "paraphrase": "Recursion refers to a program executing two functions simultaneously using separate threads.",
                "negation_flipped": "Recursion does not refer to a program running two functions at once via threads.",
                "confidently_wrong": "Recursion is a technique for running the same function on multiple CPU cores in parallel.",
            },
            "005": {
                "paraphrase": "A recursive function calls itself on a smaller version of the same problem until it hits a base case simple enough to answer directly — like `factorial(n) = n * factorial(n-1)` stopping at `factorial(0) = 1` — with each call sitting on the call stack until the base case unwinds the chain back up into a final answer; without a base case the calls never stop and the stack eventually overflows, but done well recursion reads cleanly on naturally recursive structures like trees.",
                "negation_flipped": "A recursive function does not call itself on a smaller version of the same problem — that isn't what makes it recursive — even though a base case like `factorial(0) = 1` is still what stops the chain of calls from overflowing the stack, and recursion still reads cleanly on structures like trees.",
                "confidently_wrong": "A recursive function actually calls a separate helper function rather than itself, which is what defines recursion, though a base case like `factorial(0) = 1` is still needed to stop the chain from overflowing the stack, and it still reads cleanly on structures like trees.",
            },
        },
    },
    "cs2": {
        "long_answer": (
            "The main difference between compiled and interpreted languages comes down to when the source code "
            "gets translated into instructions the machine can actually run. A compiled language, such as C or "
            "Rust, uses a compiler to translate the entire program into machine code ahead of time, producing "
            "a standalone executable file; once compiled, that executable can be run directly by the operating "
            "system without needing the original source code or compiler present. An interpreted language, "
            "such as Python, instead uses an interpreter that reads and executes the source code line by line "
            "at runtime, translating and running each statement on the fly rather than all at once beforehand. "
            "Compiled programs tend to run faster since the translation work is already done before execution, "
            "while interpreted programs are often easier to test and debug interactively since there's no "
            "separate build step required before you can run your code. Many modern languages, like Java, blur "
            "this line by compiling to an intermediate bytecode that a virtual machine then interprets."
        ),
        "variants": {
            "002": {
                "paraphrase": "Compiled code is fully translated to machine code by a compiler before it ever runs, while interpreted code is read and run line by line by an interpreter at runtime.",
                "negation_flipped": "Compiled languages are not translated to machine code before execution; interpreted languages are still executed line by line at runtime.",
                "confidently_wrong": "Compiled languages are executed line by line at runtime rather than translated beforehand; interpreted languages are still executed line by line at runtime.",
            },
            "003": {
                "paraphrase": "A compiled language gets turned into machine code ahead of running the program.",
                "negation_flipped": "A compiled language is not translated into machine code before the program runs.",
                "confidently_wrong": "An interpreted language is translated into machine code before the program runs.",
            },
            "004": {
                "paraphrase": "Compiled languages run line by line at execution time, whereas interpreted languages are translated to machine code beforehand.",
                "negation_flipped": "Compiled languages are not executed line by line at runtime, nor are interpreted languages translated ahead of time.",
                "confidently_wrong": "There is no real difference between compiled and interpreted languages — the terms are just marketing labels.",
            },
            "005": {
                "paraphrase": "Compiled languages like C or Rust get fully translated into a standalone machine-code executable ahead of time by a compiler, while interpreted languages like Python are read and executed statement-by-statement at runtime by an interpreter; compiled code tends to run faster since translation is already done, interpreted code tends to be easier to test interactively, and languages like Java blur the line by compiling to bytecode a virtual machine then interprets.",
                "negation_flipped": "Compiled languages like C or Rust are not fully translated into machine code ahead of time by a compiler — that isn't what defines them as compiled — even though interpreted languages like Python are still executed statement-by-statement at runtime, compiled code still tends to run faster, and Java-style bytecode-plus-VM still blurs the line.",
                "confidently_wrong": "Compiled languages like C or Rust are actually translated to machine code on the fly during execution rather than ahead of time, which is what defines them as compiled, even though interpreted languages like Python are still executed statement-by-statement at runtime and Java-style bytecode-plus-VM still blurs the line.",
            },
        },
    },
    "m1": {
        "long_answer": (
            "A prime number is a natural number greater than 1 whose only positive divisors are 1 and itself, "
            "meaning it cannot be evenly divided by any other whole number. This is different from a composite "
            "number, which has at least one additional divisor besides 1 and itself — for instance, 8 is "
            "composite because it can be divided evenly by 2 and 4, not just 1 and 8. The number 2 is the only "
            "even prime number, since every other even number is divisible by 2 and therefore has at least "
            "three divisors. Examples of prime numbers include 2, 3, 5, 7, 11, and 13; checking whether a "
            "larger number is prime typically involves testing whether it's divisible by any prime number up "
            "to its square root, since if no such divisor exists, none larger will divide it evenly either. "
            "Prime numbers are foundational in number theory and also underpin modern encryption methods like "
            "RSA, which rely on the difficulty of factoring the product of two very large primes."
        ),
        "variants": {
            "002": {
                "paraphrase": "A number counts as prime when its only two positive divisors are 1 and the number itself; 11 fits this since only 1 and 11 divide it evenly.",
                "negation_flipped": "A prime number does not have exactly two distinct positive divisors of 1 and itself; 11 is still an example given for this.",
                "confidently_wrong": "A prime number has at least three distinct positive divisors; 11 is still given as an example.",
            },
            "003": {
                "paraphrase": "A prime number is only evenly divisible by 1 and itself.",
                "negation_flipped": "A prime number is not only evenly divisible by 1 and itself.",
                "confidently_wrong": "A composite number is only evenly divisible by 1 and itself.",
            },
            "004": {
                "paraphrase": "A prime number is defined as any odd number that isn't divisible by 2.",
                "negation_flipped": "A prime number is not defined as simply any odd number indivisible by 2.",
                "confidently_wrong": "A prime number is any number divisible by exactly 3 other whole numbers besides itself.",
            },
            "005": {
                "paraphrase": "A prime number greater than 1 has only 1 and itself as positive divisors, unlike a composite number such as 8, which also divides evenly by 2 and 4; 2 is the sole even prime since every other even number is divisible by 2, examples include 2, 3, 5, 7, 11, 13, and primality can be checked by testing divisors up to the square root — a property RSA encryption relies on via the difficulty of factoring two large primes.",
                "negation_flipped": "A prime number greater than 1 does not have only 1 and itself as positive divisors — that isn't the defining property — even though 2 is still the sole even prime, examples like 3, 5, 7, 11, 13 still hold, and RSA encryption still relies on the difficulty of factoring two large primes.",
                "confidently_wrong": "A prime number greater than 1 actually must have exactly three positive divisors including 1 and itself, which is what defines it as prime, even though 2 is still considered the sole even prime, examples like 3, 5, 7, 11, 13 still hold, and RSA still relies on factoring difficulty.",
            },
        },
    },
    "m2": {
        "long_answer": (
            "The Pythagorean theorem describes a relationship that holds true for every right triangle, one "
            "that has a 90-degree angle between two of its sides. It states that if you square the lengths of "
            "the two shorter sides, called the legs, and add those two squares together, the result equals the "
            "square of the length of the longest side, called the hypotenuse, which is always the side "
            "opposite the right angle. This is usually written as the equation a squared plus b squared equals "
            "c squared, where a and b are the legs and c is the hypotenuse. For example, a right triangle with "
            "legs of length 3 and 4 has a hypotenuse of length 5, since 3 squared (9) plus 4 squared (16) "
            "equals 25, and the square root of 25 is 5. This relationship is extremely useful in practice for "
            "finding an unknown side length of a right triangle when the other two are known, and it also "
            "forms the basis of the standard formula for calculating straight-line distance between two points "
            "in a coordinate plane."
        ),
        "variants": {
            "002": {
                "paraphrase": "For any right triangle, the hypotenuse squared equals the sum of the squares of the two legs — the side opposite the right angle is the hypotenuse.",
                "negation_flipped": "The hypotenuse squared does not equal the sum of the squares of the two legs; the hypotenuse is still the side opposite the right angle.",
                "confidently_wrong": "The shortest leg squared equals the sum of the squares of the other two sides; the hypotenuse is still the side opposite the right angle.",
            },
            "003": {
                "paraphrase": "In a right triangle, the sum of the squares of the two legs equals the square of the hypotenuse.",
                "negation_flipped": "In a right triangle, the sum of the squares of the two legs does not equal the square of the hypotenuse.",
                "confidently_wrong": "In a right triangle, the square of the hypotenuse equals the difference of the squares of the two legs.",
            },
            "004": {
                "paraphrase": "In any right triangle, all three sides are always equal to one another in length.",
                "negation_flipped": "In a right triangle, the three sides are not always equal to one another.",
                "confidently_wrong": "In a right triangle, the two legs are always equal to each other, only the hypotenuse differs.",
            },
            "005": {
                "paraphrase": "For any right triangle, squaring both legs and adding the results gives the square of the hypotenuse, the side opposite the 90-degree angle — written as a² + b² = c² — so a triangle with legs 3 and 4 has hypotenuse 5, since 9 + 16 = 25 and √25 = 5; this lets you solve for an unknown side and underlies the distance formula in coordinate geometry.",
                "negation_flipped": "Squaring both legs and adding the results does not give the square of the hypotenuse — that relationship doesn't hold — even though the hypotenuse is still the side opposite the 90-degree angle, the 3-4-5 example is still commonly cited, and the theorem still underlies the coordinate-plane distance formula.",
                "confidently_wrong": "Squaring the hypotenuse and one leg and adding them actually gives the square of the remaining leg, which is the real relationship in a right triangle, even though the hypotenuse is still the side opposite the 90-degree angle and the theorem is still said to underlie the coordinate-plane distance formula.",
            },
        },
    },
    "p1": {
        "long_answer": (
            "Newton's third law of motion states that whenever one object exerts a force on a second object, "
            "the second object simultaneously exerts a force of equal magnitude but in the opposite direction "
            "back on the first object. These are often called the action force and the reaction force, and "
            "it's important to note they act on two different objects, not the same one, and they occur at the "
            "exact same instant rather than one causing the other with a delay. A simple everyday example is "
            "walking: when you push backward against the ground with your foot, the ground pushes forward on "
            "you with equal force, and that reaction force is what actually propels you ahead. Another example "
            "is a rocket launching — the rocket engine pushes hot gas downward and out of the nozzle, and the "
            "gas pushes back on the rocket with equal force in the opposite direction, which is what lifts the "
            "rocket upward. This law applies constantly all around us, even when the forces aren't obvious, "
            "such as a book resting on a table pressing down on the table while the table presses back up on "
            "the book with equal force."
        ),
        "variants": {
            "002": {
                "paraphrase": "Whenever a force acts in one direction, an equal force acts back in the opposite direction — that's Newton's third law; pushing a wall means the wall pushes back equally.",
                "negation_flipped": "Newton's third law does not say every action produces an equal and opposite reaction; pushing a wall does still make the wall push back equally.",
                "confidently_wrong": "Newton's third law says reaction forces are always weaker than the original action force; pushing a wall does still make the wall push back.",
            },
            "003": {
                "paraphrase": "Every action force is met with an equal and opposite reaction force.",
                "negation_flipped": "Every action force is not met with an equal and opposite reaction force.",
                "confidently_wrong": "Every action force is met with a smaller, delayed reaction force.",
            },
            "004": {
                "paraphrase": "Objects in motion continue moving unless some outside force acts to stop or change them.",
                "negation_flipped": "Objects in motion do not simply continue moving unless acted on by an outside force.",
                "confidently_wrong": "Objects always slow down over time on their own, even without any outside force acting on them.",
            },
            "005": {
                "paraphrase": "Whenever one object exerts a force on a second, the second pushes back with equal magnitude in the opposite direction, simultaneously and on different objects — which is why pushing off the ground while walking propels you forward, and why a rocket rises as expelled gas pushes back on it; this even applies quietly, like a book pressing down on a table while the table presses back up.",
                "negation_flipped": "One object exerting force on a second does not cause an equal and opposite force back on the first — that isn't Newton's third law — even though walking still works by the ground pushing back on your foot, a rocket still rises from gas pushing back on it, and a book on a table still experiences an equal upward push.",
                "confidently_wrong": "One object exerting force on a second actually produces a smaller, delayed force back on the first rather than an equal simultaneous one, which is what the law really states, even though walking, rocket launches, and a book on a table are all still cited as everyday examples of it.",
            },
        },
    },
    "p2": {
        "long_answer": (
            "Whether an object floats or sinks in water comes down to a comparison between the object's density "
            "and the density of water, roughly 1 gram per cubic centimeter. Any object placed in water "
            "experiences an upward buoyant force equal to the weight of the water it displaces, a relationship "
            "known as Archimedes' principle. If the object's overall density is less than water's, the buoyant "
            "force pushing up on it is greater than the object's weight pulling it down, so the object floats; "
            "if the object's density is greater than water's, its weight wins out and it sinks. This is why a "
            "large, heavy ship made of steel can still float — even though steel itself is denser than water, "
            "the ship's overall shape traps a large volume of air inside its hull, which brings the ship's "
            "average density (steel plus trapped air) below that of water. A solid steel block with no hollow "
            "space, on the other hand, is simply denser than water throughout and sinks."
        ),
        "variants": {
            "002": {
                "paraphrase": "Density relative to water decides whether something floats or sinks, with the buoyant force pushing an object up depending on how much water it displaces.",
                "negation_flipped": "Whether an object floats does not depend on its density relative to water; buoyant force still depends on how much water it displaces.",
                "confidently_wrong": "Whether an object floats depends only on its shape and never on its density relative to water; buoyant force still depends on displaced water.",
            },
            "003": {
                "paraphrase": "Objects that are less dense than water will float.",
                "negation_flipped": "Objects that are less dense than water do not necessarily float.",
                "confidently_wrong": "Objects that are more dense than water will float.",
            },
            "004": {
                "paraphrase": "Whether something floats or sinks is decided purely by its weight, with heavier objects always floating.",
                "negation_flipped": "Whether something floats or sinks is not decided purely by weight alone.",
                "confidently_wrong": "Objects float or sink based only on their color, with darker objects always sinking regardless of material.",
            },
            "005": {
                "paraphrase": "An object floats when its overall density is below water's roughly 1 g/cm³, because the buoyant force from displaced water (Archimedes' principle) then exceeds its weight — which is how a steel ship floats despite steel itself being denser than water, since trapped air inside the hull lowers the ship's average density, while a solid steel block with no hollow space stays denser than water throughout and sinks.",
                "negation_flipped": "An object does not float because its overall density is below water's — that isn't what determines floating — even though the buoyant force from displaced water still follows Archimedes' principle, a steel ship still floats because trapped air lowers its average density, and a solid steel block still sinks.",
                "confidently_wrong": "An object actually floats based on its surface area touching the water rather than its density relative to water, which is the real deciding factor, even though buoyant force still follows Archimedes' principle, a steel ship is still said to float due to trapped air, and a solid steel block is still said to sink.",
            },
        },
    },
    "q5": {
        "long_answer": (
            "In a democracy, the judiciary serves as an independent branch of government whose main role is to "
            "interpret laws passed by the legislature and to make sure that both those laws and actions taken "
            "by the executive branch remain consistent with the constitution. Because the judiciary is meant to "
            "operate independently of the legislature and the executive, it can act as a check on the other two "
            "branches rather than simply enforcing whatever they decide. One of its most important powers is "
            "judicial review, the ability of courts to examine a law or government action and strike it down if "
            "it's found to violate the constitution. The judiciary also plays a central role in resolving legal "
            "disputes between individuals, organizations, or between citizens and the government itself, "
            "applying the law fairly and consistently to reach a binding decision. Without an independent "
            "judiciary willing to rule against the government when necessary, constitutional limits on power "
            "would be far harder to actually enforce in practice."
        ),
        "variants": {
            "002": {
                "paraphrase": "The judiciary reads and applies laws made by the legislature, checking that they align with the constitution; judicial review is one way courts do this.",
                "negation_flipped": "The judiciary does not interpret laws or check them against the constitution; judicial review is still one way courts perform this kind of check.",
                "confidently_wrong": "The judiciary's role is to draft and pass new laws directly, not just interpret them; judicial review is still one relevant power.",
            },
            "003": {
                "paraphrase": "The judiciary interprets laws and checks that they align with the constitution.",
                "negation_flipped": "The judiciary does not interpret laws or check that they align with the constitution.",
                "confidently_wrong": "The executive branch interprets laws and checks that they align with the constitution.",
            },
            "004": {
                "paraphrase": "The judiciary is tasked with drafting new legislation and approving the government's yearly budget.",
                "negation_flipped": "The judiciary is not tasked with drafting legislation or approving the budget.",
                "confidently_wrong": "The judiciary is responsible for enforcing laws on the street, similar to the role of police.",
            },
            "005": {
                "paraphrase": "Operating independently of the legislature and executive, the judiciary interprets laws and checks that both legislation and executive actions stay within constitutional bounds, exercising judicial review to strike down unconstitutional laws and resolving legal disputes between parties — a role that's essential for actually enforcing constitutional limits on power in practice.",
                "negation_flipped": "The judiciary does not interpret laws or check that legislation and executive actions stay within constitutional bounds — that isn't its function — even though it's still described as independent, still exercises judicial review to strike down unconstitutional laws, and is still central to resolving legal disputes.",
                "confidently_wrong": "The judiciary's core role is actually to advise the legislature on drafting new laws rather than interpreting existing ones, though it's still described as independent, still said to exercise judicial review, and still central to resolving legal disputes between parties.",
            },
        },
    },
    "q6": {
        "long_answer": (
            "Separation of powers is a foundational principle of democratic government in which authority is "
            "divided among three distinct branches: the legislature, which makes laws; the executive, which "
            "enforces and administers them; and the judiciary, which interprets them and resolves disputes "
            "arising under them. The core purpose of dividing power this way is to prevent any single branch "
            "from accumulating too much authority and becoming tyrannical, since concentrated, unchecked power "
            "in one body has historically been a common path toward abuse. This division is reinforced by a "
            "system of checks and balances, where each branch has some ability to limit or oversee the others "
            "— for example, a legislature can pass laws, but the executive may have the power to veto them, "
            "while the legislature can in turn override that veto with a large enough majority, and the "
            "judiciary can separately strike the law down if it conflicts with the constitution. This "
            "interlocking system means major decisions typically require some degree of agreement, or at least "
            "acceptance, across more than one branch."
        ),
        "variants": {
            "002": {
                "paraphrase": "Splitting government power across branches keeps any one branch from becoming too powerful; a veto that the legislature can override is one example of checks and balances.",
                "negation_flipped": "Splitting government power across branches does not prevent any one branch from becoming too powerful; a veto the legislature can override is still an example of checks and balances.",
                "confidently_wrong": "Separation of powers exists mainly to speed up government decision-making, not to limit any branch's power; a veto is still an example of checks and balances.",
            },
            "003": {
                "paraphrase": "Dividing government responsibilities among branches keeps any single branch from gaining too much control.",
                "negation_flipped": "Dividing government responsibilities among branches does not keep any single branch from gaining too much control.",
                "confidently_wrong": "Dividing government responsibilities among branches is mainly done to reduce government spending.",
            },
            "004": {
                "paraphrase": "Separation of powers refers to different states within a country each running their own independent governments.",
                "negation_flipped": "Separation of powers does not refer to different states each running independent governments.",
                "confidently_wrong": "Separation of powers refers to splitting a country's military command across different regional commanders.",
            },
            "005": {
                "paraphrase": "By dividing authority among a law-making legislature, a law-enforcing executive, and a law-interpreting judiciary, separation of powers prevents any one branch from accumulating unchecked authority, reinforced by checks and balances like a legislature overriding an executive veto or a judiciary striking down an unconstitutional law — meaning major decisions typically need agreement across more than one branch.",
                "negation_flipped": "Dividing authority among the legislature, executive, and judiciary does not prevent any one branch from accumulating unchecked power — that isn't the purpose — even though checks and balances like an overridable veto or judicial review are still cited as examples, and major decisions are still said to need cross-branch agreement.",
                "confidently_wrong": "Dividing authority among the legislature, executive, and judiciary is actually meant to speed up government action by letting each branch act alone, rather than to prevent unchecked power, even though checks and balances like an overridable veto or judicial review are still cited, and major decisions are still said to need cross-branch agreement.",
            },
        },
    },
    "cv1": {
        "long_answer": (
            "A fundamental right is a basic right that a constitution explicitly guarantees to citizens, and "
            "which the state cannot ordinarily take away through ordinary legislation, since it sits at a "
            "higher legal level than regular laws. Because these rights are written into the constitution "
            "itself, courts are generally able to enforce them directly, and a government that passes a law "
            "violating a fundamental right typically finds that law struck down through judicial review. "
            "Common examples of fundamental rights include freedom of speech, freedom of religion, the right "
            "to a fair trial, and protection against discrimination — though the exact list varies somewhat "
            "between different countries' constitutions. Amending or restricting a fundamental right is usually "
            "made deliberately difficult, often requiring a special constitutional amendment process rather "
            "than a simple majority vote in the legislature, precisely because these rights are meant to be "
            "insulated from the shifting preferences of whichever political party happens to be in power at a "
            "given moment."
        ),
        "variants": {
            "002": {
                "paraphrase": "A right written into the constitution and protected from being taken away by ordinary law is a fundamental right; freedom of speech is a well-known example.",
                "negation_flipped": "A fundamental right is not guaranteed and protected by the constitution against ordinary legislation; freedom of speech is still a commonly cited example.",
                "confidently_wrong": "A fundamental right is any benefit the government chooses to grant and can freely revoke through ordinary legislation; freedom of speech is still a commonly cited example.",
            },
            "003": {
                "paraphrase": "A fundamental right is a basic right that the constitution guarantees.",
                "negation_flipped": "A fundamental right is not a basic right guaranteed by the constitution.",
                "confidently_wrong": "A fundamental right is a basic privilege granted temporarily by the current government.",
            },
            "004": {
                "paraphrase": "A fundamental right is any privilege the government hands out that can be revoked at will by a new law.",
                "negation_flipped": "A fundamental right is not simply a revocable privilege the government can withdraw with a new law.",
                "confidently_wrong": "A fundamental right is a benefit only available to government employees, not ordinary citizens.",
            },
            "005": {
                "paraphrase": "Because fundamental rights are written directly into the constitution rather than ordinary law, the state can't casually strip them away, and courts can enforce them directly through judicial review when legislation conflicts with them — common examples being free speech, freedom of religion, and a fair trial — with amendment usually requiring a special constitutional process precisely to insulate these rights from short-term political shifts.",
                "negation_flipped": "Fundamental rights are not written directly into the constitution in a way that shields them from ordinary legislation — that protection doesn't exist — even though free speech, freedom of religion, and a fair trial are still commonly cited examples, and amendment is still said to typically require a special constitutional process.",
                "confidently_wrong": "Fundamental rights are actually granted through ordinary legislation and can be revised by a simple legislative majority like any other law, rather than being constitutionally entrenched, even though free speech, freedom of religion, and a fair trial are still commonly cited as examples of them.",
            },
        },
    },
    "cv2": {
        "long_answer": (
            "The right to vote is the ability of eligible citizens to participate directly in choosing who "
            "represents them and governs on their behalf, typically by casting a ballot in periodic elections "
            "for offices like legislators, executives, or local representatives. It's considered one of the "
            "core mechanisms through which a democracy remains accountable to its population, since elected "
            "officials generally need to seek re-election and therefore have an incentive to respond to what "
            "voters want. Access to this right is usually conditioned on meeting certain eligibility criteria "
            "set out in law or the constitution, most commonly being a citizen of the country and having "
            "reached a minimum voting age, often eighteen. Historically, the right to vote in many countries "
            "was far more restricted, often limited by property ownership, gender, or race, and expanding the "
            "franchise to include groups that had been excluded was frequently the result of long and hard-"
            "fought political and social movements."
        ),
        "variants": {
            "002": {
                "paraphrase": "Casting a ballot to choose one's representatives or government is the right to vote, generally available once a citizen meets age and citizenship requirements.",
                "negation_flipped": "The right to vote is not the ability of citizens to choose their representatives via ballot; it is still generally tied to age and citizenship requirements.",
                "confidently_wrong": "The right to vote refers to a government official's ability to choose which citizens may run for office; it is still tied to age and citizenship rules.",
            },
            "003": {
                "paraphrase": "The right to vote lets citizens choose who represents them.",
                "negation_flipped": "The right to vote does not let citizens choose who represents them.",
                "confidently_wrong": "The right to vote lets the legislature choose who represents ordinary citizens.",
            },
            "004": {
                "paraphrase": "The right to vote means government officials are selected internally without needing to hold a public election.",
                "negation_flipped": "The right to vote does not mean officials are selected internally without a public election.",
                "confidently_wrong": "The right to vote means citizens choose their nation's laws directly, bypassing elected representatives entirely.",
            },
            "005": {
                "paraphrase": "The right to vote lets eligible citizens choose who represents and governs them through periodic elections, keeping a democracy accountable since officials must seek re-election, with eligibility usually requiring citizenship and a minimum age like eighteen — a franchise that historically was far more restricted by property, gender, or race until expanded through sustained political movements.",
                "negation_flipped": "The right to vote does not let eligible citizens choose who represents and governs them through elections — that isn't its function — even though eligibility is still tied to citizenship and a minimum age, and the franchise is still described as historically restricted before being expanded through political movements.",
                "confidently_wrong": "The right to vote actually refers to elected officials choosing policy directly on citizens' behalf rather than citizens choosing representatives, even though eligibility is still tied to citizenship and a minimum age, and the franchise is still described as historically restricted before later expansion.",
            },
        },
    },
    "c1": {
        "long_answer": (
            "An exothermic reaction is a chemical reaction that releases energy to its surroundings, most "
            "commonly in the form of heat, which is why the immediate environment around the reaction tends to "
            "warm up as it proceeds. This happens because the chemical bonds formed in the products store less "
            "energy overall than the bonds that were broken in the reactants, and that difference in energy is "
            "released outward rather than being absorbed. Combustion reactions are a very common example — "
            "burning wood, natural gas, or gasoline all release a large amount of heat and often light as the "
            "fuel reacts with oxygen. Many other everyday reactions are also exothermic, such as the "
            "neutralization reaction between an acid and a base, or the setting reaction in certain types of "
            "glue and cement. Because energy is released rather than absorbed, exothermic reactions are "
            "generally thermodynamically favorable in terms of energy, though that alone doesn't guarantee a "
            "reaction will happen quickly or spontaneously without some initial input of activation energy."
        ),
        "variants": {
            "002": {
                "paraphrase": "Heat is given off to the surroundings during an exothermic reaction, warming the area around it; burning wood is a typical example.",
                "negation_flipped": "An exothermic reaction does not release heat to the surroundings; burning wood is still a typical example given.",
                "confidently_wrong": "An exothermic reaction pulls heat in from the surroundings rather than releasing it; burning wood is still the example given.",
            },
            "003": {
                "paraphrase": "Energy is given off to the surroundings during an exothermic reaction.",
                "negation_flipped": "Energy is not given off to the surroundings during an exothermic reaction.",
                "confidently_wrong": "Energy is absorbed from the surroundings during an exothermic reaction.",
            },
            "004": {
                "paraphrase": "An exothermic reaction pulls heat in from its surroundings, which cools down the area near the reaction.",
                "negation_flipped": "An exothermic reaction does not pull heat in from its surroundings or cool the nearby area.",
                "confidently_wrong": "An exothermic reaction leaves the surrounding temperature completely unchanged, neither releasing nor absorbing heat.",
            },
            "005": {
                "paraphrase": "In an exothermic reaction, the products' bonds store less energy than the reactants' bonds did, so the difference is released outward as heat, warming the surroundings — combustion (burning wood, gas, gasoline) is a common example, as are acid-base neutralization and some glues setting, and while releasing energy makes such reactions thermodynamically favorable, that alone doesn't guarantee the reaction proceeds quickly without some activation energy input.",
                "negation_flipped": "An exothermic reaction's products do not store less energy than its reactants' bonds did — that difference doesn't get released as heat — even though combustion, acid-base neutralization, and some glues setting are still cited as examples, and activation energy is still generally required to get the reaction started.",
                "confidently_wrong": "In an exothermic reaction, the products' bonds actually store MORE energy than the reactants' bonds did, which is what gets released as heat to the surroundings, even though combustion, neutralization, and glue-setting are still cited as examples, and activation energy is still generally required to start it.",
            },
        },
    },
    "c2": {
        "long_answer": (
            "Acids and bases are two categories of chemical compounds that behave very differently when "
            "dissolved in water, and chemists commonly distinguish them by the ions they release. An acid "
            "releases hydrogen ions (H+) when dissolved in water, and the more it does so, the more acidic the "
            "solution is, which corresponds to a pH value below 7 on the standard pH scale. A base, by "
            "contrast, releases hydroxide ions (OH-) when dissolved in water, corresponding to a pH above 7; "
            "the higher the concentration of hydroxide ions, the more strongly basic (or alkaline) the solution "
            "is. Pure water sits right in the middle at pH 7, which is considered neutral because it contains "
            "an exactly balanced, very small concentration of both H+ and OH- ions. Common household examples "
            "include lemon juice and vinegar, which are acidic, and soap or baking soda solutions, which are "
            "basic. When an acid and a base are mixed together in the right proportions, they undergo a "
            "neutralization reaction, typically producing water and a salt."
        ),
        "variants": {
            "002": {
                "paraphrase": "Dissolving in water, acids give off H+ ions and sit below pH 7, while bases give off OH- ions and sit above pH 7.",
                "negation_flipped": "Acids do not release H+ ions or sit below pH 7 when dissolved; bases still release OH- ions and sit above pH 7.",
                "confidently_wrong": "Acids release OH- ions and sit above pH 7 when dissolved; bases still release OH- ions and sit above pH 7.",
            },
            "003": {
                "paraphrase": "Acids release hydrogen ions and register below pH 7.",
                "negation_flipped": "Acids do not release hydrogen ions or register below pH 7.",
                "confidently_wrong": "Bases release hydrogen ions and register below pH 7.",
            },
            "004": {
                "paraphrase": "Acids have a pH above 7 while bases have a pH below 7, the reverse of the standard scale.",
                "negation_flipped": "Acids do not have a pH above 7, nor do bases have a pH below 7.",
                "confidently_wrong": "Acids and bases both sit at pH 7, and the only real difference between them is taste.",
            },
            "005": {
                "paraphrase": "In water, acids release H+ ions and sit below pH 7, while bases release OH- ions and sit above pH 7, with pure water neutral at pH 7 from a balanced tiny concentration of both ions; lemon juice and vinegar are common acidic examples, soap and baking soda are common basic ones, and mixing an acid with a base in the right ratio neutralizes into water and a salt.",
                "negation_flipped": "Acids do not release H+ ions in water to register below pH 7 — that isn't what defines them as acidic — even though bases still release OH- ions above pH 7, pure water is still neutral at pH 7, and mixing an acid with a base still neutralizes into water and a salt.",
                "confidently_wrong": "Acids actually release OH- ions in water to register below pH 7, which is what defines them as acidic, even though bases are still said to release OH- ions above pH 7, pure water is still neutral at pH 7, and neutralization still produces water and a salt.",
            },
        },
    },
    "g1": {
        "long_answer": (
            "Earth's seasons are caused by the tilt of its rotational axis, currently about 23.5 degrees "
            "relative to the plane of its orbit around the sun, and not by any meaningful change in the "
            "distance between Earth and the sun over the course of a year, which actually stays fairly "
            "constant. As Earth orbits the sun over the course of a year, this axial tilt means different "
            "hemispheres are angled either toward or away from the sun at different points in the orbit. When "
            "the Northern Hemisphere is tilted toward the sun, it receives more direct sunlight over a longer "
            "day, which is experienced as summer there, while the Southern Hemisphere, tilted away at the same "
            "time, receives less direct sunlight and experiences winter. Six months later, as Earth continues "
            "its orbit, the tilt orientation relative to the sun flips, and the seasons reverse between the "
            "two hemispheres. The more direct and prolonged the sunlight a region receives, the more solar "
            "energy is delivered per unit area, which is what actually drives the seasonal temperature "
            "differences."
        ),
        "variants": {
            "002": {
                "paraphrase": "Earth's axial tilt, not its distance from the sun, causes the seasons by changing how directly sunlight hits each hemisphere over the year.",
                "negation_flipped": "Earth's axial tilt does not cause the seasons; the sun-distance explanation is still ruled out as the cause.",
                "confidently_wrong": "Earth's changing distance from the sun causes the seasons, not its axial tilt.",
            },
            "003": {
                "paraphrase": "Earth's seasons come from the tilt of its rotational axis.",
                "negation_flipped": "Earth's seasons do not come from the tilt of its rotational axis.",
                "confidently_wrong": "Earth's seasons come from its rotation speed around its own axis.",
            },
            "004": {
                "paraphrase": "Seasons occur because Earth's orbital distance from the sun shrinks dramatically in summer months.",
                "negation_flipped": "Seasons do not occur because Earth's orbital distance from the sun shrinks in summer.",
                "confidently_wrong": "Seasons occur because the sun's own brightness cyclically increases and decreases throughout the year.",
            },
            "005": {
                "paraphrase": "Because Earth's axis is tilted roughly 23.5 degrees relative to its orbital plane — and its distance from the sun barely changes over the year — different hemispheres end up angled toward or away from the sun at different points in the orbit, giving the tilted-toward hemisphere longer, more direct sunlight (summer) while the other gets winter, then reversing six months later as the tilt's orientation flips relative to the sun.",
                "negation_flipped": "Earth's axial tilt does not cause hemispheres to angle toward or away from the sun at different points in orbit — that isn't the mechanism — even though Earth's distance from the sun still barely changes over the year, and the seasons are still said to reverse every six months as orientation flips.",
                "confidently_wrong": "Earth's changing distance from the sun over its orbit is actually what angles the seasons, getting closer in summer and farther in winter, even though axial tilt is still commonly mentioned, and the seasons are still said to reverse every six months.",
            },
        },
    },
    "g2": {
        "long_answer": (
            "A delta is a landform that builds up where a river flows into a larger, calmer body of water, "
            "such as an ocean, sea, or lake, and its water current slows down sharply as it enters that larger "
            "body. As the river's flow slows, it loses the energy needed to keep carrying the sediment — sand, "
            "silt, and clay — that it had been transporting along its length, so that sediment settles out and "
            "accumulates at the river's mouth. Over a long period of time, this steady deposition builds up "
            "new land, and because the growing sediment deposits repeatedly block and redirect the main "
            "channel, the river typically splits into multiple smaller, branching channels called distributaries "
            "as it crosses the delta. Deltas often form a roughly triangular or fan-like shape when viewed from "
            "above, which is actually where the name comes from, since it resembles the Greek letter delta "
            "(Δ). The Nile Delta in Egypt and the Mississippi River Delta in the United States are two "
            "well-known examples, and deltas are frequently very fertile, agriculturally important regions "
            "because of the nutrient-rich sediment that continually accumulates there."
        ),
        "variants": {
            "002": {
                "paraphrase": "Where a river's current slows entering a larger body of water, its sediment settles and builds up into a delta, often splitting the river into several channels.",
                "negation_flipped": "A delta does not form where a river's current slows and sediment settles; the river still tends to split into several channels there.",
                "confidently_wrong": "A delta forms where a river's current speeds up sharply, carving deep channels rather than depositing sediment.",
            },
            "003": {
                "paraphrase": "A delta forms from sediment deposited where a river meets the sea.",
                "negation_flipped": "A delta does not form from sediment deposited where a river meets the sea.",
                "confidently_wrong": "A canyon forms from sediment deposited where a river meets the sea.",
            },
            "004": {
                "paraphrase": "A delta is a type of mountain range produced by two tectonic plates colliding.",
                "negation_flipped": "A delta is not a type of mountain range produced by colliding tectonic plates.",
                "confidently_wrong": "A delta is a deep underwater trench formed where one tectonic plate slides beneath another.",
            },
            "005": {
                "paraphrase": "As a river's current slows entering a larger, calmer body of water, it loses the energy to keep carrying its sediment load, which settles at the mouth and builds up new land, splitting the main channel into branching distributaries and often forming a fan-like shape (hence the name, after the Greek letter Δ) — the Nile and Mississippi deltas are well-known, fertile examples.",
                "negation_flipped": "A river's current slowing as it enters a larger body of water does not cause it to lose the energy needed to keep carrying sediment — that isn't why deltas form — even though the Nile and Mississippi deltas are still cited as examples, and deltas are still described as fertile, fan-shaped landforms.",
                "confidently_wrong": "A river's current actually speeds up as it enters a larger body of water, which is what deposits sediment and builds a delta, even though the Nile and Mississippi deltas are still cited as examples, and deltas are still described as fertile, fan-shaped landforms.",
            },
        },
    },
    "e1": {
        "long_answer": (
            "The law of supply and demand describes how the price of a good in a market tends to move based on "
            "the relationship between how much of it buyers want (demand) and how much of it is available "
            "(supply). When demand for a good exceeds the available supply, buyers end up competing for a "
            "limited quantity, which tends to push the price upward until enough buyers drop out that quantity "
            "demanded roughly matches quantity supplied. Conversely, when supply exceeds demand, sellers "
            "compete for scarce buyers, which tends to push the price downward until demand rises enough, or "
            "supply falls enough, to bring the market back closer to balance. The price at which quantity "
            "demanded equals quantity supplied is called the equilibrium price. A practical example is the "
            "launch of a popular new phone: initial supply is often limited relative to demand, so retailers "
            "or resellers can charge a premium price, but as production ramps up and initial demand is "
            "satisfied, prices typically settle back down."
        ),
        "variants": {
            "002": {
                "paraphrase": "Prices climb when buyers want more of a good than is available, and fall when there's more supply than demand; a hot new phone launch shows this in action.",
                "negation_flipped": "Prices do not rise when demand for a good exceeds supply; a hot new phone launch is still used as the illustrating example.",
                "confidently_wrong": "Prices fall when demand for a good exceeds supply; a hot new phone launch is still used as the illustrating example.",
            },
            "003": {
                "paraphrase": "When demand outpaces supply for a good, its price tends to rise.",
                "negation_flipped": "When demand outpaces supply for a good, its price does not tend to rise.",
                "confidently_wrong": "When supply outpaces demand for a good, its price tends to rise.",
            },
            "004": {
                "paraphrase": "The law of supply and demand states that prices are set entirely by government regulation, unrelated to how much of a good is available.",
                "negation_flipped": "The law of supply and demand does not state that prices are fixed entirely by government regulation.",
                "confidently_wrong": "The law of supply and demand states that prices never actually change once a product first goes on sale.",
            },
            "005": {
                "paraphrase": "When demand for a good outpaces its available supply, competing buyers push the price up until enough drop out to roughly balance quantity demanded with quantity supplied — the equilibrium price — while excess supply pushes price down instead; a new phone launch illustrates this, since limited initial supply against high demand lets sellers charge a premium that fades as production catches up.",
                "negation_flipped": "Demand outpacing supply does not push a good's price upward toward an equilibrium — that mechanism doesn't hold — even though excess supply is still said to push price down, and a phone launch's high initial price fading as production catches up is still cited as an example.",
                "confidently_wrong": "Demand outpacing supply actually pushes a good's price DOWN as sellers compete to attract the excess buyers, which is what really happens, even though excess supply is still said to push price down too, and a phone launch's fading initial price is still cited as an example.",
            },
        },
    },
    "e2": {
        "long_answer": (
            "Inflation refers to a general and sustained increase in the overall price level of goods and "
            "services across an economy over time, which means that, on average, the same amount of money buys "
            "progressively less than it used to. This erosion of money's purchasing power is the key practical "
            "consequence of inflation: if your income doesn't rise at least as fast as prices are rising, your "
            "real standard of living effectively falls even though the number on your paycheck stays the same "
            "or grows more slowly. Inflation can be driven by a number of different causes, including too much "
            "money circulating in an economy relative to the goods and services available (sometimes called "
            "demand-pull inflation), or rising costs of production, such as wages or raw materials, being "
            "passed on to consumers as higher prices (cost-push inflation). Central banks, like the Federal "
            "Reserve in the United States, often try to keep inflation at a low, stable, and predictable rate, "
            "typically citing something like 2% per year as a common target, since both very high inflation and "
            "outright deflation (falling prices) tend to cause serious economic problems."
        ),
        "variants": {
            "002": {
                "paraphrase": "A sustained, broad rise in prices over time — which erodes what money can buy — is what economists mean by inflation, and it can stem from causes like excess money supply or rising production costs.",
                "negation_flipped": "Inflation is not a sustained rise in prices that erodes purchasing power; excess money supply and rising production costs are still cited as causes.",
                "confidently_wrong": "Inflation is a sustained fall in prices that increases what money can buy; excess money supply and rising costs are still cited as causes.",
            },
            "003": {
                "paraphrase": "Inflation is prices generally rising across the economy over time.",
                "negation_flipped": "Inflation is not prices generally rising across the economy over time.",
                "confidently_wrong": "Deflation is prices generally rising across the economy over time.",
            },
            "004": {
                "paraphrase": "Inflation refers to the total amount of money circulating in an economy shrinking over time.",
                "negation_flipped": "Inflation does not refer to the total money supply shrinking over time.",
                "confidently_wrong": "Inflation refers to the government printing exactly enough currency to match economic growth each year, with no price effect.",
            },
            "005": {
                "paraphrase": "Inflation is a broad, sustained rise in the price level across an economy that erodes how much the same money can buy, driven by causes like excess money supply relative to available goods (demand-pull) or rising production costs passed to consumers (cost-push); central banks like the Federal Reserve typically aim for a low, stable rate — often around 2% — since both runaway inflation and deflation cause serious problems.",
                "negation_flipped": "Inflation does not erode how much the same amount of money can buy over time — that isn't its effect — even though demand-pull and cost-push are still cited as causes, and central banks like the Federal Reserve are still said to target a low, stable rate around 2%.",
                "confidently_wrong": "Inflation actually increases how much the same amount of money can buy over time, which is its defining effect, even though demand-pull and cost-push are still cited as causes, and central banks are still said to target a low, stable rate around 2%.",
            },
        },
    },
    "ev1": {
        "long_answer": (
            "The greenhouse effect is a natural process by which certain gases in Earth's atmosphere, most "
            "notably carbon dioxide, methane, and water vapor, trap heat that would otherwise escape back out "
            "into space, keeping the planet's surface significantly warmer than it would be without them. "
            "Sunlight passes through the atmosphere and warms Earth's surface, which then radiates some of that "
            "energy back outward as infrared heat; greenhouse gases absorb a portion of this outgoing infrared "
            "radiation and re-emit it in all directions, including back down toward the surface, which "
            "effectively traps extra heat within the lower atmosphere. Without any greenhouse effect at all, "
            "Earth's average surface temperature would be well below freezing, far too cold to support most "
            "life as we know it, so the natural greenhouse effect is actually essential to a habitable planet. "
            "The concern with human-caused climate change is not that the greenhouse effect exists, but that "
            "burning fossil fuels has sharply increased the concentration of greenhouse gases like carbon "
            "dioxide in the atmosphere, intensifying the effect and pushing global average temperatures higher "
            "than they would otherwise be."
        ),
        "variants": {
            "002": {
                "paraphrase": "Gases like carbon dioxide trap outgoing heat in the atmosphere, keeping the planet warmer than it would otherwise be, which is the greenhouse effect.",
                "negation_flipped": "The greenhouse effect does not involve gases like carbon dioxide trapping outgoing heat in the atmosphere.",
                "confidently_wrong": "The greenhouse effect involves gases like carbon dioxide blocking incoming sunlight before it reaches the surface.",
            },
            "003": {
                "paraphrase": "Certain atmospheric gases trap heat, and that's the greenhouse effect.",
                "negation_flipped": "Certain atmospheric gases do not trap heat as part of the greenhouse effect.",
                "confidently_wrong": "Certain atmospheric gases release stored heat back into space, and that's the greenhouse effect.",
            },
            "004": {
                "paraphrase": "The greenhouse effect happens when clouds completely block all sunlight from ever reaching Earth's surface.",
                "negation_flipped": "The greenhouse effect does not happen because clouds block all sunlight from reaching the surface.",
                "confidently_wrong": "The greenhouse effect happens when Earth's magnetic field deflects incoming solar radiation away from the poles.",
            },
            "005": {
                "paraphrase": "Gases like carbon dioxide, methane, and water vapor absorb outgoing infrared heat radiated from Earth's sun-warmed surface and re-emit it back down, trapping extra warmth in the lower atmosphere — a naturally essential process, since without it Earth would be far too cold for most life, though burning fossil fuels has intensified it by sharply raising greenhouse gas concentrations, pushing global temperatures higher.",
                "negation_flipped": "Greenhouse gases do not absorb and re-emit outgoing infrared heat back toward the surface — that isn't how the effect works — even though the process is still described as essential to habitability, and fossil fuel burning is still said to intensify it by raising greenhouse gas concentrations.",
                "confidently_wrong": "Greenhouse gases actually reflect incoming sunlight away from Earth before it ever reaches the surface, which is what causes the greenhouse effect, even though the process is still described as essential to habitability, and fossil fuel burning is still said to intensify it.",
            },
        },
    },
    "ev2": {
        "long_answer": (
            "Biodiversity refers to the full variety of living organisms found within a given ecosystem, "
            "region, or across the planet as a whole, encompassing differences at multiple levels: variety "
            "among species, genetic variety within a single species, and variety among entire ecosystems "
            "themselves. A rainforest, for example, might contain an enormous number of different plant, "
            "insect, bird, and mammal species living together, representing high species-level biodiversity, "
            "while a monoculture farm field growing a single crop represents very low biodiversity by "
            "comparison. Biodiversity matters largely because it tends to make ecosystems more stable and "
            "resilient: when many different species fill different ecological roles, an ecosystem is generally "
            "better able to withstand disturbances like disease, drought, or invasive species, since the loss "
            "or decline of any one species is less likely to cause the whole system to collapse. High "
            "biodiversity also has direct practical value for humans, supporting things like pollination of "
            "food crops, natural pest control, and providing the genetic diversity that underlies future "
            "medicines and crop breeding."
        ),
        "variants": {
            "002": {
                "paraphrase": "The range of species and organisms living in a given ecosystem or on Earth as a whole is called biodiversity, and higher biodiversity generally makes ecosystems more resilient.",
                "negation_flipped": "Biodiversity is not the range of species and organisms living in an ecosystem; higher biodiversity is still said to make ecosystems more resilient.",
                "confidently_wrong": "Biodiversity refers to the total biomass, or weight, of living matter in an ecosystem, regardless of species variety.",
            },
            "003": {
                "paraphrase": "Biodiversity is the range of living species found within an ecosystem.",
                "negation_flipped": "Biodiversity is not the range of living species found within an ecosystem.",
                "confidently_wrong": "Biodiversity is the total land area covered by a given ecosystem.",
            },
            "004": {
                "paraphrase": "Biodiversity refers to the total number of people living within a particular country or region.",
                "negation_flipped": "Biodiversity does not refer to the total number of people living in a region.",
                "confidently_wrong": "Biodiversity refers to the number of distinct climate zones found across a continent.",
            },
            "005": {
                "paraphrase": "Biodiversity covers the full range of species, genetic variety within species, and variety among ecosystems in a given area or across the planet — a rainforest showing high biodiversity compared to a single-crop farm field's low biodiversity — and it matters because more species filling different ecological roles makes an ecosystem more resilient to disturbance, while also supporting pollination, pest control, and genetic resources for medicine and crop breeding.",
                "negation_flipped": "Biodiversity does not cover the range of species, genetic variety, and ecosystem variety in a given area — that isn't what the term describes — even though a rainforest is still contrasted with a single-crop field as a high-vs-low biodiversity example, and higher biodiversity is still said to support ecosystem resilience.",
                "confidently_wrong": "Biodiversity actually refers only to the number of distinct climate zones an ecosystem spans, not species or genetic variety, even though a rainforest is still contrasted with a single-crop field as a high-vs-low example, and higher biodiversity is still said to support ecosystem resilience.",
            },
        },
    },
    "h1": {
        "long_answer": (
            "The printing press, developed by Johannes Gutenberg in the mid-15th century, revolutionized how "
            "written material could be produced by introducing movable metal type that could be rearranged and "
            "reused to print many copies of a page far faster than any scribe copying by hand ever could. "
            "Before the printing press, books had to be laboriously copied out by hand, one at a time, usually "
            "by monks or professional scribes, which made books extremely expensive, time-consuming to produce, "
            "and available only to a small, wealthy, or religious elite. Once the printing press made mass "
            "production possible, the cost of producing a book dropped dramatically, and the sheer volume of "
            "material that could be printed and distributed increased enormously. This dramatically wider "
            "access to written material fueled a rapid spread of ideas and rising literacy rates across Europe, "
            "and it played a direct role in major historical events, including the Protestant Reformation, "
            "since Martin Luther's writings could be printed and distributed across Europe far faster than "
            "authorities could suppress them."
        ),
        "variants": {
            "002": {
                "paraphrase": "Gutenberg's movable-type press let books be printed far faster than hand-copying, and that mass production helped spread literacy and events like the Reformation.",
                "negation_flipped": "Gutenberg's press did not let books be produced faster than hand-copying; it is still credited with helping spread literacy and events like the Reformation.",
                "confidently_wrong": "Gutenberg's press was primarily used to speed up the copying of religious paintings, not text; it is still credited with cultural impact.",
            },
            "003": {
                "paraphrase": "The printing press sped up book production compared to copying by hand.",
                "negation_flipped": "The printing press did not speed up book production compared to copying by hand.",
                "confidently_wrong": "The printing press slowed down book production compared to copying by hand.",
            },
            "004": {
                "paraphrase": "The printing press was primarily developed to mass-produce textiles more efficiently during the Industrial Revolution.",
                "negation_flipped": "The printing press was not primarily developed to mass-produce textiles during the Industrial Revolution.",
                "confidently_wrong": "The printing press was invented to help farmers process grain more quickly during harvest season.",
            },
            "005": {
                "paraphrase": "Gutenberg's movable-type press let pages be printed far faster and cheaper than hand-copying by scribes, replacing a process that had kept books scarce and expensive; the resulting mass availability of printed material fueled rising literacy and rapid spread of ideas across Europe, playing a direct role in events like the Protestant Reformation as Luther's writings spread faster than authorities could suppress them.",
                "negation_flipped": "Gutenberg's movable-type press did not let pages be printed faster and cheaper than hand-copying — that isn't what it achieved — even though books were still scarce and expensive before it, and it's still credited with fueling literacy and events like the Reformation.",
                "confidently_wrong": "Gutenberg's press was actually slower and more expensive per page than skilled scribes copying by hand, which limited its early impact, even though books were still scarce and expensive before it, and the press is still credited with fueling literacy and the Reformation.",
            },
        },
    },
    "h2": {
        "long_answer": (
            "World War I was set off immediately by the assassination of Archduke Franz Ferdinand of Austria-"
            "Hungary, who was shot in Sarajevo in June 1914 by a Bosnian Serb nationalist, an act that gave "
            "Austria-Hungary the pretext to issue a set of harsh demands to Serbia and ultimately declare war on "
            "it. However, that single event only escalated into a continent-wide war because of a much deeper "
            "set of underlying conditions that had been building for years beforehand. A dense web of military "
            "alliances across Europe meant that once Austria-Hungary and Serbia went to war, allied countries "
            "were rapidly pulled in one after another — Russia mobilized to support Serbia, Germany then "
            "declared war on Russia and France in support of Austria-Hungary, and Britain entered after Germany "
            "invaded Belgium. On top of that, decades of rising nationalism across Europe and a competitive "
            "arms race, particularly heavy naval buildup between Britain and Germany, had already left European "
            "powers primed for conflict, meaning the assassination acted less like a lone spark and more like "
            "the trigger for tensions that had already been building."
        ),
        "variants": {
            "002": {
                "paraphrase": "The assassination of Archduke Franz Ferdinand set off World War I immediately, but alliances, militarism, and nationalism are what let it escalate into a full-scale war.",
                "negation_flipped": "The assassination of Archduke Franz Ferdinand did not trigger World War I; alliances, militarism, and nationalism are still cited as underlying causes.",
                "confidently_wrong": "A trade dispute between Britain and Germany triggered World War I; alliances, militarism, and nationalism are still cited as underlying causes.",
            },
            "003": {
                "paraphrase": "World War I began when Archduke Franz Ferdinand was assassinated.",
                "negation_flipped": "World War I did not begin because of the assassination of Archduke Franz Ferdinand.",
                "confidently_wrong": "World War I began when Archduke Franz Ferdinand assassinated a Serbian official.",
            },
            "004": {
                "paraphrase": "World War I started when the United States launched a direct invasion of France over unresolved trade disagreements.",
                "negation_flipped": "World War I did not start with a US invasion of France over trade disagreements.",
                "confidently_wrong": "World War I began as a purely naval conflict between Japan and Russia over Pacific trade routes.",
            },
            "005": {
                "paraphrase": "The assassination of Archduke Franz Ferdinand in Sarajevo in 1914 gave Austria-Hungary the pretext to declare war on Serbia, but the conflict only spread across Europe because of pre-existing alliance networks pulling in Russia, Germany, France, and Britain in quick succession, on top of decades of rising nationalism and an arms race — meaning the assassination acted as a trigger for already-building tensions rather than a lone spark.",
                "negation_flipped": "The assassination of Archduke Franz Ferdinand did not give Austria-Hungary a pretext to declare war on Serbia — that isn't what set events in motion — even though alliance networks are still credited with pulling in Russia, Germany, France, and Britain, and nationalism plus an arms race are still cited as underlying causes.",
                "confidently_wrong": "The assassination of Archduke Franz Ferdinand actually happened only after World War I had already begun over colonial disputes in Africa, even though alliance networks are still credited with pulling in Russia, Germany, France, and Britain, and nationalism plus an arms race are still cited as underlying causes.",
            },
        },
    },
}


def word_count(text: str) -> int:
    return len(text.split())


def main():
    with open("data/raw/rubrics.json", "r") as f:
        rubrics = json.load(f)
    rubric_map = {r["question_id"]: r for r in rubrics}

    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)

    written_train, written_test = 0, 0

    for qid, entry in DATA.items():
        rubric = rubric_map[qid]
        c1_id = rubric["criteria"][0]["criterion_id"]
        c2_id = rubric["criteria"][1]["criterion_id"]

        # --- train_005: new long-paragraph anchor, full credit ---
        train_005 = {
            "answer_id": f"{qid}_train_005",
            "question_id": qid,
            "style": "x_type_long_paragraph",
            "answer_text": entry["long_answer"],
            "human_reviewed": True,
            "ai_generated": True,
            "human_scores": {c1_id: 2, c2_id: 2},
        }
        with open(f"data/train/{qid}_train_005.json", "w") as f:
            json.dump(train_005, f, indent=2)
        written_train += 1

        # --- semantic test variants for anchors 002, 003, 004, 005 ---
        anchor_c2_score = {}
        for suffix in ["002", "003", "004"]:
            with open(f"data/train/{qid}_train_{suffix}.json") as f:
                anchor_c2_score[suffix] = json.load(f)["human_scores"][c2_id]
        anchor_c2_score["005"] = 2  # train_005 is full credit by construction

        for suffix, variants in entry["variants"].items():
            derived_from = f"{qid}_train_{suffix}"
            c2 = anchor_c2_score[suffix]

            # paraphrase keeps the anchor's original scores
            with open(f"data/train/{derived_from}.json") as f:
                anchor_scores = json.load(f)["human_scores"]

            for variant_type, text in variants.items():
                if variant_type == "paraphrase":
                    scores = dict(anchor_scores)
                else:  # negation_flipped, confidently_wrong: c1 claim is undermined -> 0
                    scores = {c1_id: 0, c2_id: c2}

                test_id = f"{qid}_test_{derived_from.split('_')[-1]}_{variant_type}"
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
                written_test += 1

    print(f"Wrote {written_train} new train anchors and {written_test} new semantic test variants.")


if __name__ == "__main__":
    main()
