# Milestone 1: Retrieval Evaluation Discussion

## Dataset
- **Category:** All_Beauty (Amazon Reviews 2023)
- **Total products indexed:** 112,590
- **Retrieval methods evaluated:** BM25 (keyword-based), 
  Semantic Search (embedding-based)

---

## 1. Query Set

We designed 10 queries spanning three difficulty levels:

| # | Query | Type | Expected Winner |
|---|---|---|---|
| 1 | gentle cleanser for sensitive skin | Easy-Medium | Both |
| 2 | vitamin C serum | Easy | BM25 |
| 3 | L'Oreal shampoo | Easy | BM25 |
| 4 | waterproof mascara | Easy | Both |
| 5 | something to keep my face hydrated all day | Medium | Semantic |
| 6 | product for dark spots and uneven skin tone | Medium | Semantic |
| 7 | gift for someone who loves skincare | Medium | Semantic |
| 8 | best anti aging cream under 30 dollars | Complex | Neither fully |
| 9 | help with hair loss and thinning hair | Complex | BM25 |
| 10 | CeraVe moisturizer | Easy | Both |

---

## 2. Retrieval Results

### Query 1: "gentle cleanser for sensitive skin"

**BM25 Top 5:**
| Rank | Title | Score |
|---|---|---|
| 1 | Men's 2-Step Sensitive Skin Acne Cleanser & Face Cream | 18.424 |
| 2 | One Over One - Natural Micellar Cleansing Water No Rinse Gentle | 18.006 |
| 3 | Natural Face Wash for Sensitive Skin - Gentle Anti Aging | 17.600 |
| 4 | Sensitive Skin Daily Facial Cleanser | 17.600 |
| 5 | Gentle Skin Cleanser for Sensitive Skin | 17.590 |

**Semantic Top 5:**
| Rank | Title | Score |
|---|---|---|
| 1 | Jason Gentle Basics Facial Cleanser 16 Fluid Ounce | 0.6904 |
| 2 | Calming Chamomile Daily Face Cleanser for Sensitive Skin | 0.6806 |
| 3 | Serious Skin Care Glycolic Cleanser | 0.6676 |
| 4 | Gentle Daily Cleanser for Sensitive Skin | 0.6601 |
| 5 | Natural Gentle Face Wash | 0.6543 |

---

### Query 2: "vitamin C serum"

**BM25 Top 5:**
| Rank | Title | Score |
|---|---|---|
| 1 | Springs Organic Vitamin C Serum For Your Face | 17.254 |
| 2 | Best Vitamin C Serum With Retinol CoQ10 Matrixyl 3000 | 15.951 |
| 3 | Serum Sensation Vitamin C Serum with Hyaluronic Acid | 15.453 |
| 4 | Organic Vitamin C Serum for Face-Professional Strength | 15.250 |
| 5 | 7% Snake Peptide + Vitamin B3 B5 + Retinol Vitamin A + Vitamin E | 15.055 |

**Semantic Top 5:**
| Rank | Title | Score |
|---|---|---|
| 1 | PURE VITAMIN C SERUM | 0.7282 |
| 2 | Liz K Super First C Serum Pure Vitamin C 13% | 0.7243 |
| 3 | Vitamin C Serum with L'Ascorbic Acid - Facial Skin Care | 0.7078 |
| 4 | Skin Pharmacy Advanced Anti-Aging Vitamin C Skin Brightening Serum | 0.6944 |
| 5 | Vitamin C Serum 20% with Vitamin E - Anti-Aging Anti-Wrinkle | 0.6912 |

---

### Query 3: "L'Oreal shampoo"

**BM25 Top 5:**
| Rank | Title | Score |
|---|---|---|
| 1 | L'Oreal Artec Kiwi Coloreflector Shampoo + Conditioner | 16.297 |
| 2 | L'Oreal Artec Kiwi Coloreflector Shampoo + Conditioner | 16.297 |
| 3 | L'Oreal Kids Extra Gentle 2-in-1 Shampoo Strawberry | 14.019 |
| 4 | 6 Pack - Loreal Magic Top Coats | 13.750 |
| 5 | L'Oreal Paris True Match Powder | 13.711 |

**Semantic Top 5:**
| Rank | Title | Score |
|---|---|---|
| 1 | L'Oreal Hair Expertise EverCreme Nourishing Shampoo | 0.7783 |
| 2 | L'Oreal Kids Extra Gentle 2-in-1 Shampoo Strawberry | 0.7612 |
| 3 | L'Oreal Elvive Dream Lengths | 0.7064 |
| 4 | AHC COLOR VIBRANCY CD 3OZ | 0.6870 |
| 5 | L'Oreal Preference BR1 Mega Brown Cinnamon | 0.6739 |

---

### Query 4: "waterproof mascara"

**BM25 Top 5:**
| Rank | Title | Score |
|---|---|---|
| 1 | Colorful Mascara FOONEE Colored Mascara Waterproof 3D Fiber | 14.911 |
| 2 | QIC 4D Silk Fiber Lash Mascara Waterproof | 14.747 |
| 3 | 4D Silk Fiber lash Mascara Waterproof Natural Thick | 14.612 |
| 4 | 4D Silk Fiber Lash Mascara Fiber Mascara 4D Silk | 14.482 |
| 5 | 4D Silk Fiber Eyelash Mascara Cream Extension Waterproof | 14.444 |

**Semantic Top 5:**
| Rank | Title | Score |
|---|---|---|
| 1 | 4D Fiber Eyelash Mascara Waterproof for Natural and Voluminous Look | 0.8099 |
| 2 | 4D Silk Fiber Lash Mascara 4D Silk Fiber Eyelash | 0.7898 |
| 3 | Ownest 4D Silk Fiber Eyelash Mascara Waterproof Long Lasting | 0.7845 |
| 4 | Clinique Gentle Waterproof Mascara Long-Wearing Lash Builder | 0.7837 |
| 5 | Mascara Waterproof Voluminous and Lengthening Lashes Clump-free | 0.7803 |

---

### Query 5: "something to keep my face hydrated all day"

**BM25 Top 5:**
| Rank | Title | Score |
|---|---|---|
| 1 | Grace My Face Minerals Powder Me Louder Soothing Redness Control | 17.628 |
| 2 | Not My Mama's Bubbling Girls Face Wash All Natural Daily Cleanser | 15.230 |
| 3 | Kiss My Face Organic Lip Balm Pineapple Coconut | 15.119 |
| 4 | My Favorite Face Wash | 14.901 |
| 5 | Keep It 100 Hydrating Toner | 14.780 |

**Semantic Top 5:**
| Rank | Title | Score |
|---|---|---|
| 1 | One Drop of Our All Natural Hyaluronic Acid Serum Will Soothe | 0.6603 |
| 2 | Face Oasis Cleansing Water | 0.6225 |
| 3 | Hydrating Face Mask For Men and Women By ERH | 0.6086 |
| 4 | 24-Hour Deep Hydration Face Cream | 0.5991 |
| 5 | Ultra Hydrating Moisturizer for Dry Skin | 0.5912 |

---

### Query 6: "product for dark spots and uneven skin tone"

**BM25 Top 5:**
| Rank | Title | Score |
|---|---|---|
| 1 | Garnier Pure Dark Spot Corrector Daily Illuminating Moisturiser | 25.702 |
| 2 | Nose Pore Wax Strips for hair removal | 25.404 |
| 3 | Planet Eden 70% Glycolic Acid Skin Peel Kit | 24.344 |
| 4 | Skin Magical 0 Zero Perfect Illuminator | 23.916 |
| 5 | Becca Ultimate Coverage 24-Hour Foundation Tonka | 23.597 |

**Semantic Top 5:**
| Rank | Title | Score |
|---|---|---|
| 1 | BREMENN CLINICAL Dark Spot Eraser Compound for Visibly Reducing Dark Spots | 0.7121 |
| 2 | BREMENN CLINICAL Dark Spot Eraser Compound for Visibly Reducing Dark Spots | 0.7121 |
| 3 | Dark Spot Reducing Serum Visibly Reduces Density of Age Spots Sun Spots | 0.7053 |
| 4 | Intimate Area Dark Spot Corrector Skincare Cream for Discoloration | 0.6863 |
| 5 | Vice Reversa Pigment Fader Dark Spot Remover with Niacinamide | 0.6688 |

---

### Query 7: "gift for someone who loves skincare"

**BM25 Top 5:**
| Rank | Title | Score |
|---|---|---|
| 1 | Brut Locion Dominant Milk Rose 3.4 Oz | 17.502 |
| 2 | The Body Shop Love & Plums Mini Gift Set Pampering Festive Skincare | 16.820 |
| 3 | The Body Shop Shea Ultimate Collection Gift Set 6pc Bath and Body | 15.710 |
| 4 | That Company Called If 6619 Little Miss Sunshine Keyring | 14.873 |
| 5 | Lab Series Clay Mask 3.4 oz | 14.705 |

**Semantic Top 5:**
| Rank | Title | Score |
|---|---|---|
| 1 | Gift Set 2-in-1 Spa-Quality Facial & Hair Care Moroccan Argan Oil | 0.6433 |
| 2 | Create Radiant Healthy Skin! Defy Your Age with Ultra Moisturizers | 0.6290 |
| 3 | Skinn Cosmetics Non-Negotiables AM & PM Cleanser 3-Piece Gift Set | 0.6201 |
| 4 | Bestpriceam 1pc Pro Nail Art Dust Cleaner Face Blush Brush Makeup | 0.6050 |
| 5 | The Body Shop Shea Ultimate Collection Gift Set | 0.5956 |

---

### Query 8: "best anti aging cream under 30 dollars"

**BM25 Top 5:**
| Rank | Title | Score |
|---|---|---|
| 1 | PENEBELLA Eye Cream Natural and Organic Anti Aging Under Eye Cream | 25.495 |
| 2 | The Best Eye Wrinkle Cream Anti Aging Eye Cream Skin Care Treatment | 24.573 |
| 3 | The Best Eye Anti-Wrinkle Cream By HeavenHerald81 | 24.440 |
| 4 | Best Selling Anti-Aging Skin Care Kits Krasa Anti-Aging Cream | 23.305 |
| 5 | PENEBELLA Hyaluronic Acid Eye Cream Anti-Aging Treatment | 21.925 |

**Semantic Top 5:**
| Rank | Title | Score |
|---|---|---|
| 1 | Best Anti Aging Cream For Men - Protect And Nourish Damaged Skin | 0.7875 |
| 2 | Best Anti Aging Vitamin C Creams with Super Fine Natural Hydrolized | 0.7058 |
| 3 | Best Anti Aging Face Cream With Both Retinol and Hyaluronic Acid | 0.7002 |
| 4 | Organic Excellence Silk Protein Anti-Aging Cream 2oz Fragrance-Free | 0.6720 |
| 5 | Premium Face Moisturizer Anti Aging Face Cream Best Hydrator | 0.6670 |

---

### Query 9: "help with hair loss and thinning hair"

**BM25 Top 5:**
| Rank | Title | Score |
|---|---|---|
| 1 | Veta Hair Stimulating Shampoo For Hair Loss Drug Free & Sulfate Free | 18.659 |
| 2 | Jardin Sante Organic Anti Hair Loss Shampoo with Argan Oil | 18.321 |
| 3 | Natural DHT Blocker Biotin Hair Loss Hair Growth Tonic Bee Propolis | 18.142 |
| 4 | Natural DHT Blocker Biotin Hair Loss Hair Growth Shampoo Bee Propolis | 18.102 |
| 5 | Hair Growth Serum Hair Growth Treatment Hair Growth oil For Hair Loss | 17.883 |

**Semantic Top 5:**
| Rank | Title | Score |
|---|---|---|
| 1 | Natural DHT Blocker Biotin Hair Loss Hair Growth Tonic Bee Propolis | 0.6616 |
| 2 | Natural DHT Blocker Biotin Hair Loss Hair Growth Shampoo Bee Propolis | 0.6595 |
| 3 | Biotin Hair Shampoo for Thinning Hair and Hair Moisturizer for Dry | 0.6098 |
| 4 | Hair Fibers for Thinning Hair 100% Undetectable Natural Formula | 0.5873 |
| 5 | PATRICKS SH1 Thickening Shampoo for Hair Loss in Men DHT Blocker | 0.5803 |

---

### Query 10: "CeraVe moisturizer"

**BM25 Top 5:**
| Rank | Title | Score |
|---|---|---|
| 1 | CeraVe Moisturizing Cream and Healing Ointment Bundle | 16.728 |
| 2 | CeraVe Hydrating Facial Cleanser 24 oz For Normal to Dry Skin | 16.281 |
| 3 | CeraVe AM Facial Moisturizing Lotion SPF 30 Oil-Free Face | 14.700 |
| 4 | CeraVe Daily Moisturizing Lotion for Normal to Dry Skin | 14.492 |
| 5 | CeraVe Moisturizing Cream 16 oz | 14.312 |

**Semantic Top 5:**
| Rank | Title | Score |
|---|---|---|
| 1 | Cerave Moisturizing Cream With Pump For Normal To Dry Skin | 0.8036 |
| 2 | CeraVe Hydrating Oil Cleanser 236ml | 0.7630 |
| 3 | CeraVe SA Cream for Rough Bumpy Skin 16 oz | 0.7439 |
| 4 | CeraVe Daily Moisturizing Lotion | 0.7201 |
| 5 | CeraVe Foaming Facial Cleanser | 0.7089 |

---

## 3. Comparison and Analysis (5 Selected Queries)

### Query 2: "vitamin C serum" — BM25 favored

**BM25** returns highly relevant results. Both *"vitamin"* and *"serum"* are specific enough terms that BM25 keyword matching works well. All top 5 results are genuine vitamin C serums.

**Semantic search** also returns relevant results but slightly  less precise — result 5 (*"Vitamin C Serum 20% with Vitamin E"*) drifts toward general anti-aging rather than specifically vitamin C focused products.

**Winner: Tie** — both methods perform well on exact product keyword queries. BM25 has a slight edge because the exact 
terms are rare and specific enough to be strong IDF signals.

---

### Query 5: "something to keep my face hydrated all day" — Semantic favored

**BM25 completely fails** on this query. The top results are:
- *"Grace My Face Minerals"* — matched *"my"*, *"face"*
- *"Not My Mama's Face Wash"* — matched *"my"*, *"face"*
- *"Kiss My Face Organic Lip Balm"* — matched *"my"*, *"face"*

None of these are relevant. BM25 matched common words (*"my"*, *"face"*) instead of understanding that the user 
wants a moisturizer or hydrating product.

**Semantic search succeeds** — returns:
- Hyaluronic Acid Serum (hydration-focused)
- Hydrating Face Mask
- 24-Hour Deep Hydration Face Cream

These are genuinely relevant even though the query shares almost no words with the product titles. The embedding model understood that *"hydrated all day"* maps to hydrating/moisturizing products.

**Winner: Semantic search** — this is the clearest example of semantic search's advantage over BM25. BM25 takes queries literally; semantic search understands intent.

---

### Query 6: "product for dark spots and uneven skin tone" — Mixed results

**BM25** returns one genuinely relevant result (*Garnier Dark Spot Corrector*) at rank 1, but rank 2 (*"Nose Pore Wax Strips"*) is completely irrelevant — it matched *"skin"* and *"spots"* without understanding context. BM25 scored it high because the query has many tokens (*"product"*, *"dark"*, *"spots"*, *"uneven"*, *"skin"*, *"tone"*) and this product happens to contain several.

**Semantic search** returns consistently relevant results — all five results are dark spot correctors or pigmentation treatments. It understood that *"uneven skin tone"* and *"dark spots"* relate to pigmentation products specifically.

**Winner: Semantic search** — longer, descriptive queries that describe a skin concern benefit from semantic understanding BM25 struggles when queries have many tokens because irrelevant products can score high by matching multiple common words.

---

### Query 7: "gift for someone who loves skincare" — Both struggle

**BM25** partially fails — rank 1 (*"Brut Locion Dominant"*) is completely irrelevant, matched on *"someone"* or common words. Ranks 2-3 are gift sets which are relevant. Rank 4 (*"Little Miss Sunshine Keyring"*) is a keychain — completely irrelevant.

**Semantic search** also partially fails — rank 4 (*"Bestpriceam Nail Art Dust Cleaner Brush"*) is irrelevant. However ranks 1, 3, 5 are skincare gift sets which are genuinely relevant.

**Winner: Semantic search marginally** — neither method handles this query well because *"gift"* is a conceptual intent rather than a product attribute. The dataset has limited gift set products and neither method can infer gift-giving intent reliably. This type of query would benefit significantly from a RAG system that could reason about intent.

---

### Query 8: "best anti aging cream under 30 dollars" — Price filtering impossible

**BM25** returns anti-aging eye creams at ranks 1-3, which are relevant to *"anti aging cream"* but the word *"best"* in product titles boosted irrelevant products. Cannot filter by price since 84.3% of prices are missing in our dataset.

**Semantic search** returns more consistently relevant anti-aging creams without the *"best"* keyword bias. However it also cannot filter by price.

**Winner: Semantic search marginally** — both methods completely ignore the price constraint (*"under 30 dollars"*) due to missing price data. This is a fundamental dataset limitation. A RAG system with a price lookup tool could address this in Milestone 2.

---

## 4. Summary of Insights

### Strengths and Weaknesses

| Method | Strengths | Weaknesses |
|---|---|---|
| BM25 | Fast, reliable for exact terms, excellent for brand names (*"CeraVe"*, *"L'Oreal"*), interpretable | Literal matching only, fails on descriptive/conceptual queries, sensitive to common words |
| Semantic | Understands meaning and intent, handles synonyms, works on descriptive queries | Slower (encoding), less precise on exact brand/product queries, lower confidence scores overall |

### Which query types challenge each method

**BM25 fails on:**
- Descriptive intent queries: *"something to keep my face hydrated"*
- Conceptual queries: *"gift for someone who loves skincare"*
- Queries where common words dominate: *"product for dark spots"*

**Semantic search fails on:**
- Queries requiring structured filtering: *"under 30 dollars"*
  (no model understands price constraints from missing data)
- Queries where a niche brand name has weak semantic signal

**Both methods struggle with:**
- Price-constrained queries — 84.3% of prices are missing,
  making price filtering impossible for either method
- Highly conceptual intent queries like gift recommendations
- Queries where product descriptions are too sparse to match
  against (78.8% of products have only title + details)

### Where advanced methods would help

**RAG (Milestone 2):**
- Could reason about price constraints using a live product 
  lookup tool
- Could synthesize multiple product results into a helpful 
  recommendation rather than just a ranked list
- Could handle *"best"* queries by combining retrieval with 
  rating-based reasoning

**Reranking:**
- A cross-encoder reranker could improve precision on complex 
  queries like *"product for dark spots and uneven skin tone"*
  by scoring query-document pairs more carefully

**Hybrid Search:**
- Combining BM25 + semantic scores would handle both 
  *"CeraVe moisturizer"* (BM25 strength) and 
  *"something hydrating"* (semantic strength) well in 
  a single system

---

*Full EDA findings and dataset statistics are documented in `notebooks/milestone1_exploration.ipynb`*
