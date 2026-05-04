# Research Proposal

## Motivation

A startup which signs an NDA not knowing "best efforts" means maximum obligation doesn't get a surprise — they get sued. Therefore, the maximum obligation is not fulfilled, leading to legal actions undertaken due to negligence. Legal contracts house within themselves a number of contractual clauses containing legal implicatures that can be vague, and often not explicitly outlining the full extent of the legal obligations that the promisor owes.

## Research Question 

$$Can a fine-tuned LLM using reinforcement learning pragmatically infer the meaning behind legal implicatures in contractual clauses in the same way as courts would interpret them?$$

## Gap in Existing Work

Existing LLMs are able to infer the meaning behind legal implicatures, but not in the same manner as judicial bodies (i.e., courts and legal institutions) will interpret them. A particular vague implicature or term often gets semantically inferred rather than pragmatically, where the language model resorts ito approximation patterns on learned patterns and not legal pragmatic inference.

A more rigorous method is missing where the ground truth can be compared to past judicial proceedings to determine accuracy. To be more precise, we are in need of a model that can interpret the legal implicatures by reasoning on it in the same manner as it would be done in a legal setting. This would require a number of factors to be considered, as pragmatic inference in legal contexts is often dependent on a vast context to be taken into account, some of them not being tied to the present, dating back to norms and standards established years earlier.

Existing datasets do not offer a way to examine a model's interpreation against judicial reasoning. ContractNLI provides a premise-hypothesis pair and a label, but not the judicial reasoning behind the stated label; CaseHOLD enables testing of the legal principle understanding of models by exposing them to judicial decisions, not the reasoning. Reward models compute a score for a model's output and its reasoning step, but not with how aligned it is to preset ground truths, which in our case will be past judicial reasonings and legal contexts of the contract.

## Proposed Method

A **Judicial Deliberation Alignment Reward (JDAR)** that rewards reasoning alignment across multiple facets of legal implicature, not just the final interpretation.

Leveraging reinforcement learning, we reward or penalise each reasoning step the model goes through, not just the final interpretation. This will enable the model to better analyse and underdstand particular contexts, and scenarios that might have been overlooked can be focussed on, thereby promoting more contextual pragmatic understanding rather than a one-shot question-answer correction that can lead to approximation.

**Dataset Construction**: The proposed method requires a dataset with specific structure, namely: the contractual clauses, hypothesis to be tested, and the judicial reasoning. This _triplet dataset_ will be constructed by extracting commercial dispute judgements from existing repositories, applying rhetorical role labeling to isolate the reasoning and decision segments, and linking those to the contractual clauses cited within them. 

## Evaluation

The evaluation will be comprised of two components:

- **Out-of-Distribution (OOD) generalisation** to test the understanding of novel terms and context through domain transfer. For example, you train the model on maritime law, and test it on space mining. This will allow us to test if the model relies on deep structural understanding, or infers by looking at the structure of the story.

- **Counterfactual Evaluation** tests the sensitivity of the model to changes in causal structure, where particular testing instances will have a flipped-term from the training set, testing the model's reasoning depth on causal reasoning. 