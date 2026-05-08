# Relevance Filtering

If you were given an opinion text, the judge's stating of the contractual clause, and the reasoning behind that clause will be embedded or contained in a single opinion text. There should be a way of differentiating between a judge stating/quoting the contractual clause, and the same judge's reasoning on the clause (both are needed for different reasons).

As already outlined in [dataset_construction.md](https://github.com/adrinorosario/legal-pragmatic-inference/blob/main/docs/research/dataset_construction.md), obligations carry a structure like **Party + Modal + Action**, but it can be rephrased and restated in different forms in a judge's reasoning about it.

Going back to the relevance filtering step, it looks like this:

- Detect if the input has an obligation (not vague legal obligation). Flag if it contains one or not.
  * If it does, apply ***Relevance Filtering*** to identify if they contain vague legal obligations
  * Compute the probability of vagueness scoring
  * Apply **coreference resolution** to identify and pickup the vague obligations leaked over to multiple sentences
  * Use the linked sentences and apply **dependency parsing** to reconstruct the clause

For the coreference resolution and dependency parsing to work, they need to know which sentences to work on, i.e., the sentences that contain the quoted contractual clauses.

- A clause is telling you what will happen, while reasoning tells you what did happen; and more than that, contracts are binary, do this not that, shall or shall not while reasoning is not binary, where it is more probabilistic as it weighs on what could have happened and what did happen and the correlation of their different causes that could have been avoided, etc. 

## Linguistically grounded feature set:

**Clause signal:** deontic modals (shall, must, will), Party + Modal + Action structure, short directive sentences, binary framing.

**Reasoning signal:** epistemic modals (would, could, might, should have), degraded Party + Modal + Action, longer sentences, probabilistic framing, reference terms (this provision, such obligation, the aforementioned).

> A clause is directive — matching the world to the words. Reasoning is assertive — matching the words to the world. Deontic modals enforce obligations on the world, while Epistemic modals reason about what the world is.

The feature set isolates the **quoted clause sentences** from the rest of the opinion. It sits between vagueness scoring and coreference resolution. It identifies and isolates sentences that contain quoted contractual clauses that need to be passed to coreference resolution and dependency parsing for reconstructing the contractual clause
