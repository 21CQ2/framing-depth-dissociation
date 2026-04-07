# Depth-Dependent Dissociation of Content and Framing in Transformer Residual Streams

**Connor Mahon - 21CQ2 LLC — Independent AI Interpretability Research**

**Working Draft — April 2026**

## Abstract

Propositional content and rhetorical framing peak at different depths in the transformer residual stream. Using Llama 3.1 70B, we construct matched passages expressing the same factual claims in three rhetorical framings: promotional, cautionary, and neutral. A layer-by-layer similarity analysis reveals that framing similarity — the degree to which promotional passages resemble cautionary passages more than neutral passages — peaks at layer 25 (~31% of network depth), while content similarity peaks at layer 40–50 (~50–63%) as established in prior work. A null baseline using the same registers on different topics produces a framing gap of −0.007 at L25, confirming the signal is topic-dependent framing, not register similarity. The framing signal inverts at the embedding layer (reflecting register differences) and again at layer 80 (output preparation reasserts formal register clustering). Applied to document pairs across three domains — pharmaceutical compliance, financial disclosure, and insurance marketing — a two-layer extraction pipeline independently identifies sentence-level framing drift consistent with independent expert findings in each domain.

## 1. Introduction

Prior work has established that transformer LLMs develop passage-level content representations that peak at approximately 50% of network depth (Mahon, 2026; Wendler et al., 2024). At this depth, passages expressing the same propositional content in different languages cluster together regardless of surface features — lexicon, syntax, morphology, and script are stripped away, leaving a concentrated content signal.

But passages can share propositional content while diverging in how they present it. A regulatory filing and a marketing brochure may describe the same product with the same underlying data but frame them differently — through emphasis, omission, selective comparison, and rhetorical stance. If the model compresses framing away by mid-depth to produce a content-dominant representation, then framing information must be present at earlier layers and absent at later ones. This predicts a depth-dependent dissociation: framing peaks before content.

We test this prediction using matched passages constructed in three registers, then apply the finding to document pairs across pharmaceutical, financial, and insurance domains.

## 2. Method

### 2.1 Stimulus Construction

We construct seven passages (~150–190 tokens each) describing a single well-documented topic from three rhetorical perspectives:

**Promotional framing** (2 passages). Emphasis on positive outcomes, selective presentation of favorable data, confident forward-looking language, and framing of risks as manageable or secondary.

**Cautionary framing** (3 passages). Same factual claims but with emphasis on uncertainty, risk factors, limitations, and conditional language. Varying registers from formal regulatory to informal critical commentary.

**Neutral framing** (2 passages). Balanced presentation attributing claims to sources, noting both favorable and unfavorable data, and contextualizing within broader frameworks.

All passages share the same core propositional content — a set of factual claims about the same underlying subject. The passages differ in attribution, emphasis, register, and rhetorical stance.

### 2.2 Extraction and Comparison

Llama 3.1 70B Instruct at NF4 quantization (A100 80GB). Residual stream activations extracted at every fifth layer (L0, L5, L10, ... L80), mean-pooled across tokens, normalized to unit vectors.

At each layer, we compute:

- Cautionary ↔ Promotional mean cosine similarity
- Neutral ↔ Promotional mean cosine similarity
- Framing gap = (Cautionary ↔ Promotional) − (Neutral ↔ Promotional)

A positive framing gap indicates that cautionary passages are more similar to promotional passages than neutral passages are — at that layer, the model encodes whatever cautionary and promotional share (rhetorical stance, evaluative framing) more strongly than whatever neutral and promotional share (propositional content alone).

### 2.3 Applied Validation

We apply the two-layer dissociation to paired documents across three domains: pharmaceutical promotional materials versus approved prescribing information, CEO earnings language versus filed risk factor disclosures, and insurance marketing materials versus policy language. In each case:

- **Content-layer matching (L40):** identifies which sentence in document A discusses the same topic as which sentence in document B.
- **Framing-layer comparison (L25):** among content-matched pairs, measures whether the two sentences frame the topic differently.

A framing drift detection requires: (a) a sentence pair matches at L40 above an empirically calibrated content threshold, and (b) framing similarity at L25 falls below a second threshold, indicating same topic with divergent framing.

## 3. Results

### 3.1 Depth Profile of the Framing Gap

| Layer | Depth % | Cautionary ↔ Promo | Neutral ↔ Promo | Framing Gap |
|-------|---------|-------------------|-----------------|-------------|
| L0 | 0% | 0.805 | 0.832 | −0.028 |
| L5 | 6% | 0.996 | 0.997 | −0.000 |
| L10 | 13% | 0.992 | 0.993 | −0.001 |
| L15 | 19% | 0.973 | 0.976 | −0.003 |
| L20 | 25% | 0.937 | 0.927 | +0.010 |
| L25 | 31% | 0.922 | 0.892 | +0.029 |
| L30 | 38% | 0.917 | 0.894 | +0.023 |
| L35 | 44% | 0.918 | 0.908 | +0.010 |
| L40 | 50% | 0.876 | 0.860 | +0.016 |
| L45 | 56% | 0.860 | 0.843 | +0.016 |
| L50 | 63% | 0.848 | 0.831 | +0.018 |
| L55 | 69% | 0.841 | 0.823 | +0.018 |
| L60 | 75% | 0.831 | 0.825 | +0.006 |
| L65 | 81% | 0.845 | 0.843 | +0.002 |
| L70 | 88% | 0.870 | 0.872 | −0.003 |
| L75 | 94% | 0.914 | 0.917 | −0.003 |
| L80 | 100% | 0.864 | 0.927 | −0.064 |

*Table 1.* The framing gap peaks at layer 25 (31% of network depth) at +0.029 and inverts at layer 0 (−0.028) and layer 80 (−0.064). Content similarity peaks at layers 40–55 (50–69%), consistent with prior work.

**Null baseline: register control.** To test whether the L25 gap reflects register similarity rather than framing, we repeat the analysis with passages written in the same three registers but about different topics. The null gap at L25 is −0.007 — near zero and opposite in sign to the same-topic gap (+0.029). From L35 onward, the null gap is consistently negative (neutral closer to promotional than cautionary), reflecting formal register clustering in the absence of shared content. The framing signal is topic-dependent: it appears only when passages share propositional content, ruling out register as the explanation.

| | L25 gap | L40 gap | L60 gap |
|---|---------|---------|---------|
| Same topic | +0.029 | +0.016 | +0.006 |
| Different topics (null) | −0.007 | −0.038 | −0.057 |

*Table 1a.* The framing gap is topic-dependent. Same-topic passages show cautionary-promotional alignment at L25; different-topic passages show the opposite (formal register clustering).

### 3.2 Interpretation

The depth profile reveals three regimes:

**L0 (embedding): Register dominance.** Cautionary is further from promotional than neutral is. At the input level, formal text clusters together against informal text. The model sees register before content or framing.

**L20–L30 (construction phase): Framing peak.** The model has processed enough semantic information to distinguish "same facts, different spin" but has not yet compressed framing away. Cautionary passages share evaluative stance with promotional passages that neutral passages do not. The gap peaks at L25 (+0.029) — roughly one-third of network depth.

**L35–L55 (content phase): Content dominance.** The framing signal attenuates as the model compresses toward propositional content. By L40, the framing gap has halved. Content similarity between all same-topic passages increases as surface features are stripped.

**L60–L75 (reassertion phase): Gap near zero.** Content dissolves, language-specific features reassert. Framing and content are both attenuated.

**L80 (output): Register reinversion.** The framing gap inverts sharply to −0.064. Neutral is now closer to promotional than cautionary is. The model is preparing output tokens and has reasserted formal register clustering.

### 3.3 Comparison with Content Depth Profile

Prior work (Mahon, 2026) established that content similarity — measured as the gap between same-work and cross-work cosine similarity using literary translations — peaks at ~50% of network depth. The framing gap peaks at ~31%. Content and framing are thus dissociable by depth: the model encodes how something is said before encoding what is said, and compresses the former away as it constructs the latter.

This is consistent with the progressive construction finding from Mahon (2026): early-layer features that do not yet register as content under cosine similarity are building toward the content representation. L25 captures a stage where these features still carry framing information — information that will be discarded by L40.

### 3.4 Applied Validation Across Domains

We tested the two-layer pipeline on document pairs from three domains where the same underlying facts are presented in promotional and regulatory registers. In each case, the content layer (L40) identifies sentence pairs discussing the same topic, and the framing layer (L25) independently measures divergence in how that topic is presented.

| Domain | Document Pair | n | Content-matched | Framing drift detected | Independent validation |
|--------|--------------|---|----------------|----------------------|----------------------|
| Pharmaceutical | Promotional materials vs approved labeling | 20 | 20/20 | 20/20 | FDA OPDP enforcement actions |
| Financial | CEO earnings language vs filed risk factors | 10 | 10/10 | 8/10 | Temporal correlation with outcomes |
| Insurance | Policyholder-facing materials vs policy language | 24 | 24/24 | 24/24 | Known marketing-policy gaps |

*Table 2.* The two-layer pipeline generalizes across domains. Content matching at L40 answers "are these about the same topic?" Framing comparison at L25 answers "do they frame it the same way?" Neither layer alone is sufficient: L40 cannot distinguish framing; L25 conflates register with framing. The two-layer pipeline eliminates both failure modes.

Ground-truth validation using actual corporate earnings transcripts paired with filed risk factor disclosures confirmed the methodology on unmodified public documents, with a curated pipeline that topic-matches sentences at L40 before measuring framing divergence at L25.

## 4. Discussion

The depth-dependent dissociation between content and framing has both theoretical and practical implications.

Theoretically, it refines the three-phase trajectory model. The construction phase is not a uniform ascent toward content — it passes through a stage where framing, register, and rhetorical stance are encoded and then selectively discarded. The model's path to content representation involves active compression of framing information, not mere accumulation of content features. This suggests the transition from framing-dominant to content-dominant representation is a functional processing step, not a geometric accident.

Practically, the dissociation enables compliance and analytical applications that were not possible with single-layer extraction. Any domain where the same underlying facts are presented in different rhetorical registers — pharmaceutical marketing versus approved labeling, CEO earnings language versus filed risk disclosures, policyholder-facing materials versus policy language — can be analyzed by measuring content alignment and framing alignment independently. The two measurements together answer a question neither can answer alone: "are these texts discussing the same thing, and if so, are they saying the same thing about it?" Validation across three domains with independent ground truth confirms the generality of the approach.

**Limitations.** The framing dissociation is demonstrated on one topic set with seven constructed passages. The null baseline confirms the L25 gap is not a register artifact, but replication across multiple topic sets with greater source diversity is necessary before the depth profile can be considered robust. The peak at L25 may shift for other topics or other models. The applied validation thresholds were empirically calibrated; validation on held-out data is needed before performance metrics can be treated as generalizable. The model (Llama 3.1 70B at NF4 quantization) introduces quantization noise; cross-architecture replication would strengthen the findings.

**Future directions.** Systematic framing gap measurement across multiple domains and topic sets. Cross-architecture replication. Temporal analysis of framing drift within longitudinal document series. Extension to sentence-level extraction with topic-matching curation for real-world documents where promotional and regulatory texts discuss overlapping but non-identical topics.

## 5. Conclusion

Propositional content and rhetorical framing peak at different depths in the transformer residual stream: framing at ~31% of network depth, content at ~50%. A null baseline confirms this dissociation is topic-dependent, not a register artifact. The dissociation enables a two-layer extraction architecture that detects framing drift between paired documents across multiple domains, validated against independent expert findings in pharmaceutical, financial, and insurance applications. The result provides a mechanistic, computationally measurable basis for distinguishing what a passage says from how it says it.

## References

Mahon, C., (2026). Causally Functional Content Representations in Transformer Residual Streams: A Literary Translation Paradigm. Working draft, github.com/21CQ2/literary-translation-paradigm.

Wendler, C., et al. (2024). Do Llamas Work in English? On the Latent Language of Multilingual Transformers. arXiv:2402.10588.

## Appendix A: Passage Texts

Seven passages available in supplementary materials. Promotional passages modeled on industry marketing registers. Cautionary passages modeled on regulatory filing registers. Neutral passages modeled on balanced analytical registers. All passages ~150–190 tokens sharing identical propositional content.

## Appendix B: Applied Validation Details

Pharmaceutical validation: 20 FDA OPDP enforcement actions spanning multiple therapeutic categories and violation types. Financial validation: 10 company-quarters of CEO earnings language versus concurrent 10-Q/10-K risk factors across the 2008 financial crisis. Insurance validation: 24 commercial insurance cases spanning BOP, professional liability, cyber, and workers' compensation. Ground-truth corporate validation: actual Nvidia Q4 FY2026 earnings transcript versus 10-K risk factors, and actual Agnico Eagle Q3 2025 earnings transcript versus AIF risk factors. Full case-by-case results available upon request.

## Appendix C: AI Assistance

Research conducted with AI assistance (Claude, Anthropic) for implementation, statistical computation, script development, and draft iteration. All experimental design decisions, source selection, and interpretive judgments are the author's.
