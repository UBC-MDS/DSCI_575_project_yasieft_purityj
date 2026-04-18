# Milestone 2 Discussion: RAG Pipeline Evaluation

## Dataset
- **Category:** All_Beauty (Amazon Reviews 2023)
- **Total products indexed:** 112,590
- **Retrieval methods:** BM25, Semantic (FAISS), Hybrid (BM25 + Semantic)
- **LLM:** Qwen/Qwen3.5-0.8B (local, via HuggingFace Transformers, MPS backend)

---

## 1. LLM Model Choice

**Model Selected:** `Qwen/Qwen3.5-0.8B`

**Rationale:**
We selected Qwen3.5-0.8B as our local LLM because it is a lightweight model (~800M parameters) that can run on a laptop without a dedicated GPU, making it accessible across different hardware setups. It supports the MPS backend on Apple Silicon, which significantly reduces inference time compared to CPU-only execution (CPU execution exceeded 5 minutes per query, making it unusable for interactive use).

We considered `google/flan-t5-large` and `meta-llama/Llama-3.2-3B-Instruct` as alternatives. Flan-T5 is faster but produces less natural answers. Llama-3.2-3B would produce higher quality output but requires more VRAM. Qwen3.5-0.8B offered the best balance between accessibility and quality given our hardware constraints.

**Observed limitation:**
Despite running on MPS, the model frequently fails to follow prompt instructions reliably. It leaks internal reasoning tags (`<think>...</think>`), repeats sentences, echoes raw context blocks, and outputs only ASINs instead of synthesized answers. These are well-documented limitations of sub-1B parameter models. For the last milestone, we will switch to a hosted API such as Groq (Llama3-8B).

---

## 2. Prompt Template Experiments

We designed two system prompt variants and ran all 5 evaluation queries through both to compare behaviour.

### Variant 1 — Strict Grounding (V1)
```
You are a helpful Amazon shopping assistant.

STRICT RULES:
- Answer ONLY using the provided reviews
- Do NOT make up information
- Do NOT repeat the question or instructions
- Do NOT output words like "assistant" or tags like "<think>"
- Do NOT generate numbered lists (1, 2, 3, etc.)
- Keep the answer short and clear (2-4 sentences max)
- Always cite product ASINs when relevant

Output ONLY the final answer. No extra text.
```

### Variant 2 — Expert Assistant (V2)
```
You are an expert product recommendation assistant.

TASK:
Use the provided reviews to answer the question.

RULES:
- Base your answer ONLY on the reviews
- Do NOT invent information
- Do NOT repeat the prompt or instructions
- Avoid numbered or bullet lists unless necessary
- Keep the answer concise and readable
- Mention product ASINs when relevant

Output ONLY the final answer. No explanations.
```

### Comparison Results

We ran both variants on the same 5 queries with the Hybrid retriever and compared outputs:

| Query | V1 Output | V2 Output |
|---|---|---|
| Best moisturizers for dry skin | Listed products with ASINs (answer cut off at token limit) | Leaked full `<think>` internal reasoning block — never produced a final answer |
| Something to keep face hydrated | Echoed raw context block instead of answering | Repeated "Based on the reviews, the product that best fits is:" 12 times with no product named |
| Product for dark spots | Correctly identified BREMENN as top product with brief reasoning | Listed ASINs repeatedly with no product names or reasoning |
| Best anti aging cream under $30 | Listed ASINs with no product names (cut off) | Listed ASINs repeatedly (identical failure to V1) |
| Gift for someone who loves skincare | Repeated same sentence 3 times + leaked `</think>` tag | Echoed raw review snippets ("My wife loved it. Perfect xmas gift.") |

**Finding:** V1 outperformed V2 on 3 out of 5 queries. The only query where V1 produced an acceptable answer (Query 3 — dark spots) completely failed under V2, which returned only raw ASINs. V2's softer instructions gave the model more room to go off-track — it hallucinated repetitive filler phrases and echoed review snippets rather than synthesizing answers.

However, both prompts failed on the majority of queries. This confirms that at 0.8B parameters, prompt engineering alone cannot reliably fix instruction-following failures. The root cause is model size, not prompt design. V1 was selected as the default for the RAG pipeline based on its marginally better performance.

---

## 3. RAG Pipeline Qualitative Evaluation

We ran the Hybrid RAG pipeline with V1 prompt on 5 queries and manually evaluated each answer on three dimensions:

- **Accuracy** (Yes/No): Is the answer factually correct based on the reviews?
- **Completeness** (Yes/No): Does the answer address all aspects of the question?
- **Fluency** (Yes/No): Is the answer natural, clear, and easy to read?

---

### Query 1: "What are the best moisturizers for dry skin?"

**Generated Answer (V1):**
> "The best moisturizers for dry skin are:
> 1. DayTime Moisturizer for Dry Skin (ASIN: B01M1A0H4Z)
> 2. Psoriasis & Eczema Cream 2oz Organic Advanced Healing Moisturizer with Shea Butter, Coconut Oil, Aloe Vera & Manuka Honey (ASIN: B01A9DVW8Q)
> 3. Neolia Hydra-Screen Apricot Oil Soap for Dry Skin, 4.5 oz (ASIN: B005WT11D4)
> 4. Physicians Formula Elastin & Collagen Moisture Lotion, 4 oz (ASIN: B00LTJV320)
> 5. BEST Night Moisturizer for Face & Neck - Deeply Hydrates, Fights Wrink..."
> *(cut off at 200 token limit)*

**Evaluation:**

| Dimension | Rating | Notes |
|---|---|---|
| Accuracy | Yes | All listed products are genuinely relevant dry skin moisturizers |
| Completeness | No | Answer cut off mid-sentence; no explanation of why these are recommended |
| Fluency | No | Ignores "no numbered lists" instruction; truncated at token limit |

**Comments:** The retrieval step works well — all returned products are genuinely relevant. However the LLM ignored the V1 formatting rules and produced a numbered list that was cut off by the `max_new_tokens=200` limit. No review content was synthesized — the model just listed product titles and ASINs. Increasing `max_new_tokens` and using a larger model would significantly improve this output.

---

### Query 2: "Something to keep my face hydrated all day"

**Generated Answer (V1):**
> *(The LLM returned the raw context block — product titles, ASINs, and retrieval scores — instead of generating an answer)*
>
> "Review
> Product ASIN: B00H2TZ4F8
> Title: One Drop of Our All Natural Hyaluronic Acid Serum...
> Rating: 0.6603
> Review: I have been using this serum for a few days..."

**Evaluation:**

| Dimension | Rating | Notes |
|---|---|---|
| Accuracy | Partial | Retrieved products are relevant but no answer was generated |
| Completeness | No | LLM returned input context instead of an answer |
| Fluency | No | Not a natural answer — raw context block echoed back |

**Comments:** This is a generation failure, not a retrieval failure. The hybrid retriever correctly surfaced Hyaluronic Acid Serum and Hydrating Face Mask — products that BM25 completely missed in Milestone 1 (BM25 matched "my" and "face" as keywords and returned irrelevant results). The LLM simply failed to synthesize those documents into a response. This is a direct consequence of the model being too small to reliably follow multi-step instructions.

---

### Query 3: "Product for dark spots and uneven skin tone"

**Generated Answer (V1):**
> "* B00B0ZIQ64 (BREMENN CLINICAL Dark Spot Eraser - Compound for Visibly Reducing Dark Spots on the Face, Neck, Décolleté, and Hands)
> * B099K3QQ8M (Intimate Area Dark Spot Corrector - Skincare Cream for Discoloration, Hyperpigmentation)
>
> B00B0ZIQ64 is the most relevant product for reducing dark spots on the face and correcting uneven skin tone."

**Evaluation:**

| Dimension | Rating | Notes |
|---|---|---|
| Accuracy | Yes | BREMENN CLINICAL Dark Spot Eraser is a genuinely relevant recommendation |
| Completeness | Partial | Identifies top product with reasoning but does not explain application or expected timeline |
| Fluency | Partial | Uses bullet points despite V1 forbidding them; reasoning sentence is clear |

**Comments:** This is the best-performing query in the evaluation. The LLM correctly identifies the most relevant product and provides a brief justification. The hybrid retriever also significantly outperformed BM25 alone — in Milestone 1, BM25 returned nose pore wax strips at rank 2 for this same query. The answer format is not ideal but the core recommendation is correct and grounded in retrieved documents.

---

### Query 4: "Best anti aging cream under 30 dollars"

**Generated Answer (V1):**
> "Product ASIN: B010L93H2C
> Product ASIN: B00FJPMNEM
> Product ASIN: B00NK65R5O
> Product ASIN: B07CTW9JRM
> Product ASIN: B07B447SM9
> ...
> Final Answer:
> Product ASIN: B010L93H2C
> Product ASIN: B00FJPMNEM"

**Evaluation:**

| Dimension | Rating | Notes |
|---|---|---|
| Accuracy | No | Returns only raw ASINs with no product names or explanation |
| Completeness | No | Cannot address the price constraint; no reasoning provided |
| Fluency | No | Not readable — just a repeated list of ASIN strings |

**Comments:** This is the worst-performing query. The LLM produced only raw ASIN strings with no product names, no reasoning, and no attempt to address the "under $30" constraint. There are two compounding failures here. First, the model ignored all formatting instructions entirely. Second, and more fundamentally, price data was excluded entirely during Milestone 1 preprocessing due to extremely high missingness (84.3% of products had no price listed). As a result, price information does not exist anywhere in the retrieved documents passed to the LLM, making the price constraint impossible to satisfy at any stage of the pipeline — retrieval, context building, or generation. This query requires a live product lookup tool such as Tavily web search or the Amazon Product Advertising API to answer properly.

---

### Query 5: "Gift for someone who loves skincare"

**Generated Answer (V1):**
> "1. The Body Shop Shea Ultimate Collection Gift Set (6pc) is great for those who love skincare. It includes various body products from the ultimate collection.
>
> 1. The Body Shop Shea Ultimate Collection Gift Set (6pc) is great for those who love skincare. It includes various body products from the ultimate collection.
>
> 1. The Body Shop Shea Ultimate Collection Gift Set (6pc) is great for those who love skincare.
>
> \</think>
>
> 1. The Body Shop Shea Ultimate Collection Gift Set (6pc) is great for those who love skincare."

**Evaluation:**

| Dimension | Rating | Notes |
|---|---|---|
| Accuracy | Partial | The Body Shop Gift Set is a genuinely relevant recommendation |
| Completeness | No | Only mentions one product despite 5 being retrieved; heavily repetitive |
| Fluency | No | Same sentence repeated 3 times; leaks a `</think>` internal reasoning tag |

**Comments:** The retrieval step worked — the hybrid retriever correctly surfaced The Body Shop Shea Ultimate Collection and Skinn Cosmetics Gift Set. However the LLM completely failed at generation: it repeated the same sentence three times and leaked a `</think>` tag. The cleanup logic in `generate_llm_answer()` splits on `</think>` but failed here because the tag appeared at the end of output rather than the beginning. This is a known issue with Qwen3-series thinking models that requires regex-based tag stripping rather than simple string splitting.

---

## 4. Overall Evaluation Summary

### Results Table

| Query | Accuracy | Completeness | Fluency | Overall |
|---|---|---|---|---|
| Best moisturizers for dry skin | Yes | No | No | Poor — truncated numbered list |
| Something to keep face hydrated | Partial | No | No | Poor — raw context echoed |
| Product for dark spots | Yes | Partial | Partial | Acceptable — correct recommendation |
| Best anti aging cream under $30 | No | No | No | Fail — only ASINs returned |
| Gift for someone who loves skincare | Partial | No | No | Poor — repetition and tag leakage |

### Key Observations

The most important finding is that **retrieval and generation have very different failure modes**. The hybrid retriever consistently returned relevant documents across all 5 queries — a clear improvement over Milestone 1 where BM25 alone failed on semantic queries like "something to keep my face hydrated all day". However, the generation step frequently failed to convert those good retrieved documents into useful answers.

The Qwen3.5-0.8B model struggles with instruction following at this scale. It ignores formatting rules, echoes context instead of synthesizing it, repeats itself, and leaks internal reasoning tags. These are well-documented limitations of sub-1B models that cannot be resolved through prompt engineering alone, as confirmed by our V1 vs V2 comparison where both variants failed on the majority of queries.

The one query where the pipeline worked acceptably (Query 3 — dark spots) was a case where the required answer was close to simple extraction: identify the most relevant product. This suggests the pipeline works best when the generation task is simple and the retrieved documents are highly descriptive.

---

## 5. Limitations of the Hybrid RAG Workflow

**Limitation 1 — LLM too small for reliable instruction following:**
The Qwen3.5-0.8B model consistently fails to follow the formatting and behaviour rules in the system prompt. It produces numbered lists despite explicit prohibition, echoes raw context blocks, repeats sentences, and leaks internal `</think>` tags from its chain-of-thought reasoning. This makes the generated answers unreliable and often unreadable. Our V1 vs V2 comparison confirmed that prompt design cannot fix this — both variants failed on the majority of queries. A model of at least 3-7B parameters is needed for consistent instruction following in a RAG pipeline.

**Limitation 2 — Cannot handle structured constraints such as price or rating filters:**
The pipeline retrieves and generates based on text similarity alone and cannot filter by numerical constraints like price or minimum star rating. This limitation is rooted in our Milestone 1 preprocessing decisions: price data was excluded entirely from the corpus because 84.3% of products in the All_Beauty dataset had no price listed, making the field too sparse to be useful. As a result, price information does not exist anywhere in the retrieved documents, making price-constrained queries like "under $30" impossible to satisfy at any stage of the pipeline — retrieval, context building, or generation. Restoring price filtering would require either a dataset with better price coverage or a live product lookup tool such as the Amazon Product Advertising API or Tavily web search.

---

## 6. Suggestions for Improvement

1. **Upgrade the LLM:** Switching to a hosted API such as Groq (Llama3-8B) would dramatically improve answer quality and instruction-following reliability without requiring local GPU resources. This is the single highest-impact improvement and we will adopt it for last milestone.

2. **Fix output parsing for Qwen3 thinking models:** Qwen3 models use chain-of-thought reasoning with `<think>...</think>` tags. The current cleanup splits on `</think>` but fails when the tag appears at the end of output. A robust fix is to use regex to strip everything between the tags before returning the answer: `re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)`.

3. **Add a web search tool:** Integrating Tavily or a similar search API would allow the pipeline to retrieve current prices and availability, directly addressing the price constraint limitation identified in Query 4 and discussed in Milestone 1.

4. **Fix context building:** The context currently passes cosine similarity scores labeled as "Rating", which causes the LLM to confuse retrieval scores (e.g. 0.67) with star ratings (out of 5). The context should include actual star ratings from product metadata and truncated review text to give the LLM accurate and richer information to synthesize from.

5. **Increase `max_new_tokens`:** The current limit of 200 tokens causes answers to be cut off mid-sentence as seen in Query 1. Increasing to 300-400 tokens would allow the model to complete its responses.