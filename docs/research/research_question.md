# The Research Question

$$\begin{gather*} 
\text{Can a fine-tuned LLM using reinforcement learning pragmatically infer the meaning behind legal implicatures} \\ 
\text{in contractual clauses in the same way as courts would interpret them?}
\end{gather*}$$


- Subject: Fine-tuned LLM using Reinforcement Learning (RL)
- Task: Pragmatic Inference on Legal Implicatures
- Benchmark: The manner in which courts interpret them

Is it possible if we can fine-tune an LLM using reinforcment learning in such a way that it can understand the reason about the hidden, and often vaguely implied obligations behind legal implicatures in contractual clauses? 

To put it in simpler terms, can this model that we are talking about look at a contractual clause and say, "This is the meaning behind the legal obligation [X] that you are entitled to carry out because the courts would interpret them as [Y]"?

# The Core Contribution

$$\text{Rewarding reasoning alignment across multiple facets of legal implicature, not just the final interpretation.}$$

Since we are using reinforcement learning, we can reward/penalise each reasoning step that the model follows and goes through rather than the final interpretation. This will enable the model to better analyse particular contexts, and scenarios that it might have overlooked, thereby promoting more contextual and pragmatic understanding, rather than a one-shot question-answer correction that can lead to approximation.

**To be more precise:** The rewarding step post each reasoning step aims to reward the alignment of that reasoning to judicial interpretations that have been deemed material, allowing the model to proceed in a streamlined manner and not go off track towards ordinary logical reasoning. 