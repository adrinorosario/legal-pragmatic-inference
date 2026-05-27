# Dataset Contruction

The idea behind this dataset is to have the model learn from past judicial reasonings. If it were given a contractual clause, will it base its reasoning on the corpora that it has already learned from and base its grounding on it? Even if it does, judicial interpretations are not grounded in such a manner. Courts have an historically significant manner of interpreting such clauses. 

Having the contractual clause, and its accompanying reasoning will enable the model to compare its own reasoning with court interpretations, enabling meaningful alignment tuning to judicial discourse.

The _triplet dataset_ needs to have the following elements:
- A contractual clause
- Vague term(s)
- Judicial interpretation and reasoning 


## Leveraing CUAD + COLD Cases

CUAD contains contracts that have neatly been split into paragraphs and accompanied by question and answers that is used for training similar models that can answer questions. Using this, we can extract the contractual clauses from the individual data points, compare the anchor with the vagueness seed set, and extract the vague terms it contains. 

The first phase of extracting the two of the three elements of the JDAR triplet is detailed below:

1. Each clause is unique and CUAD contains a number of them. We need the clauses that are either of the following:
    * Subject to intense pragmatic inference and litigation in courts
    * Are strict prohibitions but can also litigated in courts based on the contract and case law

    These clauses are separated into 3 tiers, with the first two adhering to the above mentioned conditions. The contracts from CUAD are then filtered, and only the top 2 tiered clauses are stored for further computation.

2. Using the above mentioned criteria, the following items are extracted from each data point:
    * Document ID (for cross referencing later if needed)
    * Clause category/ID
    * Text from the clause that describes the contract
    * The tier it belongs to
    * The set of vague terms contained in it

    Furthermore, we also need to check whether the clause is:

    * Lethal - belongs to tier 1
    * Has context risk - belongs to tier 2

3. **Semantic Mapping with Harvard LIL COLD Cases:**

    Harvard LIL COLD Cases contains ~ 8 million case summaries; opinion texts that are the interpretations and reasonings of judges on particular matters. 

    Using the _clause text_ that was extracted from CUAD, we will semantically compare it with the case summaries in COLD Cases to find the judicial reasoning sentences that are required for our third part of the triplet. 

    **A coarse filter on COLD Cases**

    The dataset contains a lot of noise that can harm the RL model if fed into it. Therefore, a simple coarse filter will be used to filter out all the opinion text sentences that are not needed for our use case.

    In essence, the filter performs the following:

    * Checks if the opinion texts belong to contract, commercial law, corporate law, or other such similar cases. No statutory, administrative, or tort law cases will be considered.
    * Segments each opinion text into individual sentences. Use sentences that have $> 10$ and $\le 100$ words in a sentence. This can be tuned depending on the quality of the filter's output.
    * Ensures **no statutory sentences** are present. *This is important.*
    * No checking for modals or seed words (vague terms) in this step.

    Only a set of case natures will be used, namely:
    - 'Tort, Contract, and Real Property',
    - 'Private Civil Diversity',
    - 'Private Civil Federal',
    - 'OIL, GAS AND MINERALS',

    **Reverse Constructing the JDAR Triplet**

    Given that we already have the anchor clauses, i.e., the anchor clause, clause text, and the vague terms that were extracted from CUAD. Using these clause texts and the judical reasoning candidates that we have, we can use vector spaces to pair the clause texts with their judicial interpretation counterparts.

    In essence, the triplet has the following structure:

    $
    \text{JDAR Triplet Data Point} = \begin{cases}
    \textbf{Clause} : & \text{Clean CUAD Text Snippet} \\
    \textbf{Terms} : & \text{Extracted Seed Words from the CUAD snippet and the judicial interpretation} \\
    \textbf{Reasoning} : & \text{Aligned Harvard-LIL COLD Cases Opinion Sentence (i.e., the extracted reasoning sentences}
    \end{cases}
    $

