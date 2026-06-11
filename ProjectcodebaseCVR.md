I've already read every file in the codebase. Let me now deliver the full analysis.

---
Complete Codebase Deep-Dive: promoted-stores-models

---
SECTION 1: What Is This Project?

Business Problem Being Solved

eBay has a product called Promoted Stores — a paid advertising placement where store owners pay to have their store/listings shown more prominently to users. When a user visits eBay, the ranking engine must decide: which ads to show, in what order, to maximize the chance the user clicks and buys something?

To do this well, eBay needs two ML predictions per ad request:

1. CTR (Click-Through Rate): Will this user click on this ad group? → determines ad visibility
2. CVR (Conversion Rate): If they click, will they buy? → determines ad quality & bid value

This repository contains the ML infrastructure to train, compute features for, and serve those two models.

Real-World Use Cases

User visits eBay homepage
    → Ad ranking engine fires
    → "Should I show Store A's ad or Store B's ad?"
    → CTR model: "User is 3.2% likely to click Store A, 1.1% on Store B"
    → CVR model: "Price ratio is 1.4, item has 847 sales, user likes electronics"
    → Store A wins, gets placed prominently
    → User clicks → eBay charges Store A a fee

Target Users (Who Uses This Code)

- ML Engineers: Train and update the CTR and CVR models
- Platform Engineers: Integrate the CVR feature JAR into the online ranking service
- Data Engineers: Run the offline Spark pipeline to generate training data
- Business Stakeholders: Track ad performance via model metrics (gAUC per placement)

Why This System Exists

Without these models, eBay would either:
- Show ads randomly (terrible ROI for advertisers, bad UX for users)
- Use simple rule-based ranking (misses personalization)

This system provides data-driven, personalized ad ranking at eBay scale.

---
SECTION 2: Architecture Overview

High-Level Architecture

┌─────────────────────────────────────────────────────────────────────────────┐
│                        PROMOTED STORES MODELS REPO                          │
│                                                                             │
│  ┌──────────────────────────┐    ┌──────────────────────────────────────┐  │
│  │   CTR MODELLING          │    │   CVR MODELLING                      │  │
│  │   (Python / TensorFlow)  │    │   (Scala / Spark / Maven)            │  │
│  │                          │    │                                      │  │
│  │  • Wide & Deep NN        │    │  • Feature extraction pipeline       │  │
│  │  • ~143 input features   │    │  • 70 features per ranking request   │  │
│  │  • Krylov distributed    │    │  • Spark (offline) + JVM (online)    │  │
│  │    training              │    │                                      │  │
│  └──────────┬───────────────┘    └──────────────────────────────────────┘  │
│             │                                  │                            │
│             ▼                                  ▼                            │
│  ┌──────────────────────┐    ┌─────────────────────────────────────────┐   │
│  │  DOCKER              │    │  HDFS / Krylov / YARN Cluster           │   │
│  │  train image         │    │  (eBay internal infrastructure)         │   │
│  │  serve image         │    └─────────────────────────────────────────┘   │
│  └──────────────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────────────┘

Complete System Data Flow

OFFLINE (Training Data Generation)
═══════════════════════════════════

HDFS Raw Impressions (Parquet)
│  Schema: requestId, eventTime, seedItem, creativeItems, payload(JSON), click, conversion
│
▼ CVR Pipeline (Spark)
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Parse JSON payload → item_* arrays + simplex_stage             │
│ Phase 1.5a: Combine fragments (same requestId = batch split of 1 req)  │
│ Phase 1.5b: Filter requests with no Simplex user data                  │
│ Phase 1.5c: Extract seed item scalars (match by listingId)             │
│ Phase 1.5d: Filter to surfaced items only (creativeItems)              │
│ Phase 2: Explode → Extract 70 features → Aggregate                     │
└─────────────────────────────────────────────────────────────────────────┘
│
▼ Feature Parquet (HDFS)
│  Schema: requestId, [70 float feature columns], click, conversion
│
├──────────────────────────────────────────►  CVR Model Training (elsewhere)
│
└──►  HDFS CTR Training Data (separate pipeline, not in this repo)
       │  Schema: AdsGroupId, seller_id, ..., 143 features..., click_flag
       │
       ▼ CTR Training (Krylov / Python)
       ┌───────────────────────────────────────────────┐
       │ WideAndDeepModel.fit(daily data, 20 days)     │
       │ Save checkpoint → Krylov EMS                  │
       └───────────────────────────────────────────────┘
       │
       ▼ TF SavedModel (zipped)
       │
       ▼ CTR Evaluation → gAUC per placement_id
       │
       ▼ Docker Serve Container (TF Serving)
         Ports: 1340 (REST/gRPC scoring), 5557 (monitoring)

ONLINE (Production Inference)
═══════════════════════════════

Ranking Service Call
│  RankingRequest(requestId, seedItem, N creativeItems, user context)
│
▼ OnlineFeatureService.extractFeatures()   [feature-transform-online JAR]
│
▼ buildDataRow(): maps RankingRequest → MapBasedDataRow
│  (column names match offline post-flatten schema exactly)
│
▼ FeatureTransformer.transformRow()
│  1. resetGlobalContext()
│  2. RowAdapter: explode 1 row → N item rows
│  3. SchemaBasedFeatureMapper: extract 70 features per item row
│  4. FeatureAggregator: aggregate N arrays → 1 Array[Float]
│
▼ FeatureVector(requestId, Array[Float](70), featureNames)
│
▼ CVR Model inference → conversion probability score

---
SECTION 3: Folder Structure — Purpose of Every Directory

promoted-stores-models/
│
├── ctr-modelling/                    # CTR model (Python + TensorFlow)
│   ├── script_train.py               # ENTRY POINT: submit training job to Krylov
│   ├── script_eval.py                # ENTRY POINT: submit evaluation job to Krylov
│   ├── krylov_tasks.py               # Core training + eval logic (runs on cluster)
│   ├── pyproject.toml                # Python project config (uv package manager)
│   ├── uv.lock                       # Dependency lock file
│   │
│   ├── core/                         # Model architecture + data pipeline
│   │   ├── __init__.py               # Exports: WideAndDeepModel, dataset
│   │   ├── feature_schema.py         # THE CONTRACT: all 143 input features + label
│   │   ├── feature_groups.py         # Groups features by processing type
│   │   ├── custom_model.py           # WideAndDeepModel class
│   │   ├── custom_block.py           # Processing blocks (6 types)
│   │   ├── custom_layer.py           # Low-level TF layers (7 types)
│   │   ├── dataset.py                # Parquet → tf.data.Dataset pipeline
│   │   ├── leaf_cat_mapping.npy      # Category ID → embedding index lookup
│   │   ├── leaf_cat_pretrained.npy   # Pre-trained category embeddings (raw)
│   │   ├── leaf_cat_pretrained_normal.npy  # Pre-trained embeddings (normalized)
│   │   ├── leaf_cat_mapping.csv      # Human-readable version of mapping
│   │   └── leaf_cat_pretrained.txt   # Human-readable version of embeddings
│   │
│   ├── utils/                        # Infrastructure utilities
│   │   ├── __init__.py               # Exports: initialise_hdfs, save_model, etc.
│   │   ├── hadoop.py                 # HDFS connection + file operations
│   │   ├── krylov_ems.py             # Model/metric save/load via Krylov EMS
│   │   └── date.py                   # Date range generation for training windows
│   │
│   └── ray/                          # Alternative: Ray-based distributed training
│       ├── script_train_v2.py        # Ray-based training script
│       └── dataset_v2.py             # Ray-compatible dataset
│
├── cvr-modelling/                    # CVR feature pipeline (Scala + Spark + Maven)
│   ├── pom.xml                       # Parent Maven POM (manages 3 sub-modules)
│   ├── README.md                     # Complete setup + operation guide
│   ├── submit-corp.sh                # Shell script to spark-submit on CORP cluster
│   │
│   ├── docs/                         # Technical documentation
│   │   ├── README.md                 # Doc index
│   │   ├── ARCHITECTURE.md           # System design deep-dive
│   │   ├── DESIGN_DECISIONS.md       # Why decisions were made (10 key decisions)
│   │   ├── DEVELOPMENT_NOTES.md      # Formulas + test commands (reference)
│   │   ├── IMPLEMENTATION_STATUS.md  # Per-feature status (47 real / 23 stub)
│   │   ├── IMPLEMENTATION_SUMMARY.md # Gap analysis + proxy table
│   │   ├── RAW_DATA_SCHEMA.md        # Payload JSON structure
│   │   └── all_features.md           # Canonical list of all 70 feature names
│   │
│   ├── local-test-data/              # Test data + verification scripts
│   │   ├── sample01.parquet          # 500 raw rows → 204 output rows
│   │   ├── sample02.parquet          # 500 raw rows → 195 output rows
│   │   ├── verify_features.py        # Re-implements 47 formulas in Python, compares
│   │   └── verify_schemas.py         # Checks input/output schema matches docs
│   │
│   ├── feature-transform-core/       # MODULE 1: Spark-free algorithms
│   │   ├── pom.xml                   # Only dependency: scala-library
│   │   └── src/main/scala/.../core/
│   │       ├── AdRankingFeatureNamespace.scala  # THE feature definitions (70 features)
│   │       ├── AggregationConfig.scala          # Per-feature aggregation strategy map
│   │       ├── FeatureAggregator.scala          # MAX/MIN/MEAN/SUM/FIRST logic
│   │       ├── FeatureTransformer.scala         # 3-stage pipeline orchestrator
│   │       ├── FeatureMapper.scala              # Trait + SchemaBasedFeatureMapper
│   │       ├── FeatureMappingContext.scala      # ThreadLocal 2-level cache
│   │       ├── FeatureNamespace.scala           # DSL trait ($$, $$map, $$ctx, $$const)
│   │       ├── Feature.scala                    # 64-bit feature identifier value class
│   │       ├── DataRow.scala                    # Abstract data row interface
│   │       ├── MapBasedDataRow.scala            # Map[String,Any] implementation
│   │       ├── RowAdapter.scala                 # Explode: 1 request → N item rows
│   │       └── TextUtils.scala                  # Tokenize, Jaccard, timestamp parsing
│   │   └── src/main/resources/
│   │       ├── placement_ids.csv       # 12 placement IDs → device type (mweb/native)
│   │       ├── L2_ptr.csv              # L2 category PTR lookup (bundled, pending data)
│   │       └── L3_ptr.csv              # L3 category PTR lookup (bundled, pending data)
│   │
│   ├── feature-transform-online/     # MODULE 2: Real-time inference service
│   │   ├── pom.xml                   # Depends on: feature-transform-core only
│   │   └── src/main/scala/.../online/
│   │       ├── OnlineFeatureService.scala  # HTTP service entry point
│   │       └── RankingRequest.scala        # Request/Item/FeatureVector data classes
│   │
│   └── feature-transform-pipeline/   # MODULE 3: Offline Spark batch ETL
│       ├── pom.xml                   # Depends on: core + Spark (provided)
│       └── src/main/scala/.../pipeline/
│       │   ├── FeatureTransformPipeline.scala  # spark-submit main()
│       │   ├── SparkFeatureOperators.scala     # DataFrame operations (explode/extract/agg)
│       │   └── SparkRowAdapter.scala           # Wraps Spark Row as DataRow
│       └── src/main/resources/
│           ├── input-corp.sql          # Hive SQL for CORP data pull
│           ├── input-sddz.sql          # Hive SQL for SDDZ data pull
│           └── input-schema.md         # Documents the expected input columns
│
├── docker/
│   ├── README.md                     # Build/push commands
│   ├── makefile                      # make build-train-image, make push-all, etc.
│   ├── train/
│   │   └── Dockerfile                # CUDA 12.2 + Python 3.10 + TF 2.21 + polars
│   ├── serve/
│   │   ├── Dockerfile                # TF Serving binary + Python 3.14 slim
│   │   ├── entrypoint.sh             # Launches tensorflow_model_server
│   │   ├── healthcheck.py            # Liveness probe
│   │   ├── batching.config           # TF Serving batch parameters
│   │   └── monitoring.config         # TF Serving metrics export
│   ├── test_payload_v2.json          # Sample serving request payload
│   └── test_payload_v3.json          # Sample serving request payload
│
└── _backup/                          # Old files (not active)
    ├── dataset.py                    # Prior version of CTR dataset
    ├── feature_schema.py             # Prior version of feature schema
    └── manual.ipynb                  # Exploration notebook

---
SECTION 4: Every Important File, Class, and Function

CTR Side — Python/TensorFlow

---
ctr-modelling/core/feature_schema.py

What is it? The contract between data pipeline and model. A list of 143 tuples: (column_name, dtype).

Why it exists? Both the Parquet reader and the model's input_signature need to agree on column names and types. Defined once here to prevent drift.

Simple English: "Here are the 143 fields I expect to see in the training data Parquet file, and here's the type for each one."

Technical: Creates FEATURE_SCHEMA (list of tuples) and LABEL = 'click_flag'. Used to:
- Build the PyArrow schema for reading Parquet (PA_SCHEMA)
- Build the TF Dataset signature (TF_SIGNATURE)
- Build the model's input_signature for TF Serving export

What breaks if removed? Everything. Dataset can't read data, model can't be built, serving export fails.

Feature categories in the schema:

┌───────────────────────────┬──────────────────────────────────────────────────┬───────┐
│         Category          │                     Examples                     │ Count │
├───────────────────────────┼──────────────────────────────────────────────────┼───────┤
│ Ad Group identity         │ AdsGroupId, ads_cmpgn_id                         │ 3     │
├───────────────────────────┼──────────────────────────────────────────────────┼───────┤
│ Seller info               │ seller_id, store_id, store_lvl, store_followers  │ 5     │
├───────────────────────────┼──────────────────────────────────────────────────┼───────┤
│ Item sales                │ item_sold_qty_{1,7,30}d_sum                      │ 3     │
├───────────────────────────┼──────────────────────────────────────────────────┼───────┤
│ Store engagement          │ item_vi_cnt_7d, add_to_cart_cnt_7d, etc.         │ 16    │
├───────────────────────────┼──────────────────────────────────────────────────┼───────┤
│ User behavior             │ vi, transaction, cart, search — 1d and 3d        │ 17    │
├───────────────────────────┼──────────────────────────────────────────────────┼───────┤
│ User purchase history     │ GMV, purchase counts, mean price                 │ 22    │
├───────────────────────────┼──────────────────────────────────────────────────┼───────┤
│ User category preferences │ long_term_clicked, purchased, seq — idx + weight │ 9     │
├───────────────────────────┼──────────────────────────────────────────────────┼───────┤
│ Ad performance            │ promoted clicks + impressions: 1d, 7d, 14d       │ 6     │
├───────────────────────────┼──────────────────────────────────────────────────┼───────┤
│ 360-degree item signals   │ price, impressions, views, sales: 7d, 14d        │ 7     │
├───────────────────────────┼──────────────────────────────────────────────────┼───────┤
│ Ad group embeddings       │ average, click, impression, similar-item         │ 5     │
├───────────────────────────┼──────────────────────────────────────────────────┼───────┤
│ Placement/time            │ placement_id, hour_in_day                        │ 2     │
├───────────────────────────┼──────────────────────────────────────────────────┼───────┤
│ Personalization flag      │ personalized_request                             │ 1     │
└───────────────────────────┴──────────────────────────────────────────────────┴───────┘

---
ctr-modelling/core/feature_groups.py

What is it? Categorizes the 143 features into processing groups so each Keras block knows which features to consume.

Why it exists? Different features need different preprocessing. Numerical features need batch norm. Categorical features need hashing + embedding. Pre-trained vectors need string-to-tensor parsing. This file is the routing table.

Key groups:

INDICATORS = [3 binary flags]              # → NumericalProcessor (no normalization)
NUMERICALS = [~90 continuous features]     # → NumericalProcessor (BatchNorm)
ACTUAL_CTR = [(click_col, imp_col) × 5]   # → CtrEstimator (Bayesian smoothing)
PRETRAINED_EMBEDDINGS = [6 vector cols]   # → EmbeddingParser (string → tensor)
LCF_SEED_SINGLE = 'trgtg_categ_id'        # → LeafCatEmbeddingParser (seed embed)
LCF_USER_SINGLE = [8 hot-purchase cats]   # → LeafCatEmbeddingParser (user embeds)
LCF_USER_MULTI = [(idx, weight) × 3]      # → LeafCatEmbeddingParser (weighted avg + cross)
LCF_SELLER_MULTI = ['store_dom_categ_list']# → LeafCatEmbeddingParser (seller embed + cross)
CATEGORICAL = [(col, hash_bins) × 6]      # → CategoricalProcessor (hashing + embed)
CATEGORICAL_INT_VOCAB = [(col, vocab) × 2]# → CategoricalProcessor (lookup + one-hot)
DATETIME = ['batch_feature_date', 'hour_in_day'] # → DatetimeProcessor
PERSONALISE_FLAG = 'personalized_request' # → Anonymiser
PERSONAL_INFO = [~60 user-identifying cols]# → Anonymiser (zeroed if not personalized)

---
ctr-modelling/core/custom_layer.py

What is it? Seven fundamental building blocks at the lowest level of the model.

Component-by-component:

VectorParser — Parses a comma-separated string into a fixed-size float/int tensor
Input:  "0.12,0.45,-0.33,...,0.08"  (stored as string in Parquet)
Output: tf.Tensor shape (batch, out_dim) dtype=float32
Used for: pre-trained embeddings (96-dim adgroup vectors, 64-dim user vector)

LeafCatEmbedding — Pre-trained, frozen category embedding table
Input:  category_id (int64)
Output: embedding vector (shape depends on pretrained size)
Loads leaf_cat_mapping.npy (category ID → embedding row index) and leaf_cat_pretrained_normal.npy (the actual embedding matrix). Frozen (trainable=False) — the team has pre-trained embeddings from another system and wants them fixed.

BatchMasker — Zeros out tensor values where mask=True
Input:  mask (bool tensor), data (any tensor)
Output: data with zeroed entries where mask=True
Used by Anonymiser for privacy: zero personal data when user is anonymous.

MonthOfYear — Extracts month integer from "YYYY-MM-DD" string
Input:  "2025-03-15"
Output: 3  (1-indexed)
Handles invalid/malformed dates by substituting Unix epoch (1970-01-01).

DayOfWeek — Computes ISO day-of-week from date string via Zeller's congruence
Input:  "2025-03-15"  (Saturday)
Output: 7  (ISO: 1=Monday, 7=Sunday)
Pure math — no Python date library dependency (works inside tf.function).

BayesianSmoother — Computes smoothed CTR from raw click/impression counts
Input:  clicks=5, impressions=1000
Output: (5 + 0.005) / (1000 + 1.0) = 0.005005 (very close to prior when data sparse)
Input:  clicks=0, impressions=0
Output: (0 + 0.005) / (0 + 1.0) = 0.005 (returns prior CTR when no data)
Business meaning: When an ad group has very few impressions, don't trust the raw CTR — shrink toward the global prior (0.5%). When data is abundant, trust the raw CTR.

---
ctr-modelling/core/custom_block.py

What is it? Six higher-level Keras layers that each process a category of features and return dense, cross, and/or sparse tensors.

Architecture of outputs:
- dense → fed into DNN (deep path)
- sparse → fed into Wide-Sparse linear path
- cross → fed into Wide-Cross linear path (cross-product features)

NumericalProcessor
Input:  all 3 INDICATORS + all ~90 NUMERICALS
Output: dense = BatchNorm(numericals) + indicators
        sparse = indicators only (for memorization)
Batch normalization only applied to continuous features, not binary indicators.

CtrEstimator
Input:  5 (click, impression) pairs
Output: dense = [BayesianSmoother(c1,i1), ..., BayesianSmoother(c5,i5)]
Converts raw counts into smoothed CTR estimates. Business meaning: "How has this ad been performing recently?"

EmbeddingParser
Input:  6 string columns (pre-trained 64/96-dim vectors)
Output: dense = concat of parsed vectors
Parses comma-separated embedding strings into dense tensors.

LeafCatEmbeddingParser
Input:  trgtg_categ_id (seed), hot_purchase_cat × 8, multi-hot cats × 3, store_dom_categ_list
Output: dense = [seed_embed, user_single_embeds..., user_multi_weighted_avg..., seller_multi_avg...]
        cross = [user_weighted_avg × seed_embed, seller_avg × seed_embed]
Business meaning:
- "What category is this ad in?" (seed embed)
- "What categories has this user bought from recently?" (user embeds)
- Cross product = "Does the ad's category match the user's purchase history?" → wide cross feature for memorization

CategoricalProcessor
Input:  AdsGroupId, seller_id, ads_cmpgn_id, site_id, store_id, payment_method (hash)
        store_lvl, placement_id (vocab lookup)
Output: dense = embeddings for all
        sparse = one-hot encodings for vocab-limited ones

DatetimeProcessor
Input:  batch_feature_date (string), hour_in_day (int)
Output: dense = [month_embed(8) + dow_embed(4) + hour_embed(16)]
        sparse = [one_hot_month(12) + one_hot_dow(7) + one_hot_hour(24) + is_weekday(1)]

Anonymiser
Input:  all features dict + personalized_request flag
Output: same dict, but PERSONAL_INFO features zeroed where personalized_request=0

---
ctr-modelling/core/custom_model.py

What is it? The final assembled model class.

BaseModel — Abstract base:
- Sets input_signature (for TF Serving export — this is critical for deployment)
- Defines serving_fn: applies logit_correction to convert raw logit to probability
- The logit_correction = log(prior_ctr / (1 - prior_ctr)) compensates for class imbalance during training

WideAndDeepModel — The full model:

inputs dict (143 features)
    │
    ▼ Anonymiser (privacy masking)
    │
    ├── NumericalProcessor  → dense, sparse
    ├── CtrEstimator        → dense
    ├── DatetimeProcessor   → dense, sparse
    ├── EmbeddingParser     → dense
    ├── LeafCatEmbeddingParser → dense, cross
    └── CategoricalProcessor → dense, sparse
    │
    ▼ concat dense → [all dense features]
    ▼ concat cross → [category cross features]
    ▼ concat sparse → [all one-hot/indicator features]
    │
    ▼ WideAndDeep
    ├── DNN([512, 128], dropout=0.5)(dense) → deep_logit
    ├── Dense(1, zeros)(sparse)             → wide_sparse_logit
    └── Dense(1, glorot)(cross)             → wide_cross_logit
    │
    ▼ logit = deep + wide_sparse + wide_cross
    ▼ probability = sigmoid(logit + logit_correction)

Why Wide & Deep? The deep path learns complex feature interactions (generalizes to new combinations). The wide path memorizes specific feature values (like "AdsGroupId 12345 almost always gets clicked"). Together, they outperform either alone.

---
ctr-modelling/krylov_tasks.py

What is it? Contains the actual training and evaluation logic that runs on the Krylov cluster.

task_train(as_of_date, num_days, parquet_template):

Setup:
  prior_ctr = 0.005 (0.5% — typical for promoted ads)
  class_weight = {0: 0.5/(1-0.005) ≈ 0.5025, 1: 0.5/0.005 = 100}
  # Class 1 (clicks) gets 100x weight — because only 0.5% of impressions lead to clicks

  model = WideAndDeepModel(hidden=(512,128), dropout=0.5, initial_bias=log(0.005/0.995))
  optimizer = AdamW(lr=1e-4, weight_decay=5e-3, clip_norm=1.0)

Training loop (incremental, one day at a time):
  for date in generate_dates(as_of_date, num_days):
    dataset = load_from_hdfs(parquet_template.format(date))
    history = model.fit(dataset, class_weight=class_weight)
    save_metrics(history, date)  # AUC + BCE logged per date
  save_model(model)  # checkpoint to Krylov EMS

Why incremental daily training? Each day's data is ~10-50GB. Training on 20 days at once would require ~200GB+ in memory. Incremental training keeps memory manageable while still exposing the model to recent patterns.

task_eval(checkpoint_id, parquet_path):

Load model from Krylov EMS
For each batch in evaluation dataset:
    y_pred = model(X)
    y_true = y
    group_id = X['placement_id']

Partition results by placement_id
For each placement group:
    auroc = roc_auc_score(y_true, y_pred)

gAUC = weighted_average(auroc, weight=n_samples)

Why gAUC? Regular AUC across all data mixes different placements (homepage vs. search results vs. mobile). gAUC evaluates the model within each placement independently and averages, giving a fairer picture of per-context performance.

---
ctr-modelling/core/dataset.py

What is it? Efficient data loading from HDFS Parquet to TF Dataset.

def dataset(parquet_path, filesystem) → tf.data.Dataset:
    paths = list all parquet files in directory

    return (
        from_tensor_slices(paths)               # iterate file by file
        .interleave(read_one_file, cycle_length=2)  # read 2 files in parallel
        .unbatch()                               # unpack table → individual rows
        .batch(2048, drop_remainder=True)        # re-batch at 2048
        .prefetch(AUTOTUNE)                      # overlap CPU prep with GPU training
    )

Key points:
- cycle_length=2: reads 2 Parquet files simultaneously (balance I/O and memory)
- drop_remainder=True: ensures all batches are exactly 2048 (no partial batch → consistent BatchNorm)
- filters=field(LABEL).isin([0,1]): drops any rows with invalid labels
- deterministic=False: allows reordering for performance

---
CVR Side — Scala/Spark/Maven

---
AdRankingFeatureNamespace.scala

What is it? The core of the CVR feature pipeline. Defines all 70 features using a DSL.

Why it exists? Feature definitions need to be shared between offline Spark training (millions of rows) and online real-time scoring (single request, milliseconds budget). By defining features as pure functions on DataRow, the same code works in both contexts.

The 4 DSL operators:

// 1. Direct field read with default — simplest
val FullRecoPrice = $$map[Double]("item_raw_NCalculatedTotalCost", Some(0.0))
// "Read the NCalculatedTotalCost field. If missing, return 0.0."

// 2. Custom handler — any computation on the row
val ItemSoldCount = $$(row => {
  val sold = row.getAs[Double]("item_raw_QuantitySold").getOrElse(0.0)
  if (sold > 0.0) math.log10(sold + 1.0).toFloat else 0.0f
})
// "Apply log10(sold+1) transformation — compresses large sale counts."

// 3. Context-aware handler — expensive computation cached across features
val PriceRatioNorm = $$ctx { row => ctx =>
  val recoListingPr = cachedRecoListingPrice(row, ctx)  // cached in ctx.localMap
  val seedListingPr = ...
  math.min(recoListingPr / seedListingPr, 30.0).toFloat
}
// "Price ratio, capped at 30. Listing price (total - shipping) cached per item row."

// 4. Placeholder — returns constant until upstream data available
val SeedRecoEbertCosineDistance = $$const(0.0f)
// "TODO: compute EBERT embedding cosine distance. Returns 0 until implemented."

Feature groups and business meaning:

PRICE FEATURES (10 features):
  FullRecoPrice     — How much does the reco item cost? (total price)
  FullSeedPrice     — How much does the seed item cost?
  PriceV2           — Cauchy-PDF similarity between seed and reco price
                      Business: "Is the reco item priced like the seed?"
  AbsoluteFullPriceDiff — How much more/less is reco vs seed? (signed)
  PriceRatioNorm    — Reco price / seed price, capped at 30
                      Business: "Is this item 3x more expensive than the reference?"
  BullseyeRVI*      — User's historical RVI (Recently Viewed Items) median price
                      Business: "Does this item price match what user usually browses?"

ENGAGEMENT FEATURES (11 features):
  ItemFastIMAViewCount7Day — How many times has this item been viewed in 7 days?
  ItemSalesOverView7Day    — What fraction of views led to sales? (log-smoothed)
  ItemWatchesOverView7Day  — What fraction of views led to watchlisting?
  SeedItemViewCount7Day    — How popular is the reference (seed) item?
  All variant versions     — Same metrics but for item variants

SELLER FEATURES (2 features):
  RecoPriceInteractSellerLstgConv  — log(price) × seller_conversion_rate
  RecoPriceInteractSellerMediaImp  — log(price) × log(seller_median_impressions)
  Business: "Is this a high-converting seller selling at this price point?"

PL (PROMOTED LISTINGS) PROXY FEATURES (5 features):
  PlDecayedSalesOverImpressions  — proxy: organic sales/impressions ratio
  PlClicksDecayed                — proxy: exp(logSalesOverImp) × viewCount
  PlImpressionsDecayed           — proxy: view count
  Business: "How has this item performed as a promoted listing?"

TEXT SIMILARITY FEATURES (4 features):
  MaxQueryKeywordItemTitleJaccard7Days  — Jaccard(item title, user queries last 7d)
  MaxViewedItemTitleJaccard             — Jaccard(item title, recently viewed titles)
  MaxViewedItemTitleJaccard2Days        — same, 2-day window
  MaxWatchedItemTitleJaccard7Days       — Jaccard(item title, watched items 7d)
  Business: "Does this item's title match what the user has been searching/viewing?"

---
AggregationConfig.scala

What is it? Maps each feature to how it should be combined across multiple items in an ad group.

Why it exists? An ad group has N items. Feature extraction produces one Array[Float] per item. The model needs one vector per request. Aggregation reduces N arrays to 1.

The strategies and their business rationale:

MEAN: FullRecoPrice, PriceRatioNorm, ItemConditionOrdinal
  → "What's the average price/condition across this ad group?"

MAX: BothFreeShipping, CassiniRecallSource, MaxQueryKeywordItemTitleJaccard7Days
  → "Does ANY item qualify? Is ANY item title relevant to user's query?"

SUM: ItemSoldCount, PlImpressionsDecayed, UltimatelyBoughtV2CoviewCosaleCount
  → "How many total sales across all items in this ad group?"

FIRST: UserPricePrefDuplicate, DeviceIsMWeb, SeedItemViewCount7DayDecayDomestic
  → "These are request-level features — same for every item, just take the first"

MIN: SeedRecoEbertCosineDistance (and other embedding distances)
  → "The CLOSEST item to the seed is what matters for similarity"

---
FeatureTransformer.scala

What is it? The orchestrator of the 3-stage pipeline.

Why it exists? Separates "what do we compute" (features) from "how do we process N items" (pipeline structure).

def transformRow(row: DataRow): Array[Float] = {
  mapper.resetGlobalContext()     // Step 0: clear per-request cache

  val itemCount = RowAdapter.itemCount(row)  // how many items in this request?

  val perItemFeatures = (0 until itemCount).map { idx =>
    val itemRow = RowAdapter.extractItemRow(row, idx)  // Step 1: EXPLODE
    mapper.map(itemRow)                                 // Step 2: EXTRACT
  }

  FeatureAggregator.aggregate(                          // Step 3: AGGREGATE
    perItemFeatures, featureNames, aggregationConfig, unifiedAggregation
  )
}

---
FeatureMappingContext.scala

What is it? A two-level ThreadLocal cache for feature computation.

Why it exists? Consider: a request has 8 candidate items. MaxQueryKeywordItemTitleJaccard7Days needs to tokenize the user's search queries. Without caching, this tokenization happens 8 times. With loadGlobal, it happens once per request.

Similarly, PriceRatioNorm and RecoPriceInteractSellerLstgConv both need recoListingPrice = totalCost - shippingCost. With loadLocal, this subtraction is done once per item row.

Global cache (request-scoped):
  Shared across all 8 item rows in the same request.
  Reset once before processing each new request (resetGlobal).

  Cached values:
  - parsedSearchQueries7D   (tokenized user search queries)
  - parsedViewedTitlesFlat  (pipe-delimited viewed titles)
  - parsedViewedTitles2D    (2-day windowed viewed titles)
  - parsedViewedTitles15D   (15-day windowed viewed titles)
  - parsedWatchedTitles7D   (7-day windowed watched titles)
  - tokenizedSearchQueries7D (Set[String] for each query)
  - tokenizedViewedTitlesFlatKey (Set[String] for each viewed title)
  - tokenizedViewedTitles2DKey
  - tokenizedWatchedTitles7DKey
  - placementDevice (device type from placement_ids.csv)

Local cache (row-scoped):
  Reset before each item row (resetLocal).

  Cached values:
  - recoListingPrice   (totalCost - shippingCost)
  - rviMedianCents     (user_rvi_median_price × 100)
  - itemTitleTokens    (tokenized item title Set[String])

ThreadLocal means each thread (Spark task or web server thread) has its own copy — no locking needed.

---
FeatureTransformPipeline.scala

What is it? The main() entry point for the Spark batch job that produces CVR training data.

Complete data transformation trace:

Input: Raw Parquet row
{
  requestId: "drTiUhn6rTiK/102307",
  eventTime: 1748000000000,
  seedItem: "123456789",
  creativeItems: "123456789;987654321;555555555",
  payload: '{"stages": [
    {"listingId": 123456789, "stageName": null, "rawFeatures": {"Item_NCalculatedTotalCost": "79999.0", ...}},
    {"listingId": 987654321, "stageName": null, "rawFeatures": {...}},
    {"listingId": null, "stageName": "Bullseye Simplex Data", "simplexFeatures": {...}}
  ]}',
  click: 0,
  conversion: 0
}

Phase 1: parseAndFlattenPayload()
→ {
    requestId: "drTiUhn6rTiK/102307",
    placementId: 102307,
    seedItem: 123456789,
    item_listingIds: [123456789, 987654321],
    item_raw_NCalculatedTotalCost: [79999.0, 59999.0],
    item_raw_NCalculatedShippingCost: [0.0, 999.0],
    item_raw_ViewCount_7Day: [847.0, 123.0],
    ... (25+ item_* arrays) ...,
    simplex_stage: [{simplexFeatures: {SearchModel103UserItemAffinity: {price_pref: -0.52, ...}}}],
    click: 0, conversion: 0
  }

Phase 1.5a: combineRequestFragments()
  (if multiple rows share requestId, merge item arrays; dedup by listingId)
  → same schema but deduplicated

Phase 1.5b: filterMissingUserFeatures()
  → drop row if simplex_stage is empty (no user data)

Phase 1.5c: extractUserFeatures()
→ {
    ... plus ...
    user_price_pref: -0.52,
    user_pred_price_mean: 89.99,
    user_rvi_median_price: 75.0,
    user_search_queries_ts: "vintage lamp:1747900000000|desk lamp:1747800000000",
    user_viewed_titles: "Antique Brass Lamp|Vintage Table Lamp",
    user_viewed_titles_ts: "Antique Brass Lamp:1747950000000|...",
    user_watched_titles_ts: "..."
  }

Phase 1.5d: extractSeedFeatures()
  seed item = listingId 123456789, found at index 0
→ {
    ... plus ...
    seed_total_cost: 79999.0,
    seed_shipping_cost: 0.0,
    seed_delivery_max_days: 3.0,
    seed_view_count_7day: 847.0
  }

Phase 1.5e: filterToSurfacedItems()
  creativeItems = "123456789;987654321;555555555"
  555555555 not in item_listingIds → filtered out
→ item arrays trimmed to items 123456789 and 987654321

Phase 2a: explodeToItemRows()
→ Row 1: {requestId, user_*, seed_*, item_raw_NCalculatedTotalCost=79999.0, ...}  ← item 123456789
→ Row 2: {requestId, user_*, seed_*, item_raw_NCalculatedTotalCost=59999.0, ...}  ← item 987654321

Phase 2b: extractFeatures (UDF)
→ Row 1: Array[Float](70) = [79999.0, 0.982, 79999.0, ..., 1.0, 0.0, ...]
→ Row 2: Array[Float](70) = [59999.0, 0.734, 79999.0, ..., 0.0, 1.0, ...]

Phase 2c: aggregateFeatures()
→ {
    requestId: "drTiUhn6rTiK/102307",
    FullRecoPrice: MEAN(79999.0, 59999.0) = 69999.0,
    BothFreeShipping: MAX(1.0, 0.0) = 1.0,
    ItemSoldCount: SUM(log10(847+1), log10(123+1)) = 5.856,
    UserPricePrefDuplicate: FIRST(-0.52, -0.52) = -0.52,
    DeviceIsNative: FIRST(1.0, 1.0) = 1.0,
    ... 70 features total ...
    click: 0,
    conversion: 0
  }

---
OnlineFeatureService.scala

What is it? The JAR that online ranking services link against for real-time feature extraction.

Who calls it? The production ranking service (not in this repo) calls it for every ad ranking request.

What it calls: FeatureTransformer.transformRow() from feature-transform-core.

What data enters: A RankingRequest Scala case class.

What data leaves: A FeatureVector (ArrayFloat (70) + feature names).

Critical design point: The buildDataRow method translates the RankingRequest object into a MapBasedDataRow with column names that exactly match the offline pipeline's post-flatten schema. This is the training-serving consistency guarantee.

// Online: seed scalars come from the explicit seedItem field
"seed_total_cost" -> seed.totalCost.getOrElse(0.0)

// Offline: seed scalars extracted by finding seed listingId in merged arrays
// Both result in the same "seed_total_cost" field in DataRow

---
SparkRowAdapter.scala

What is it? A bridge between org.apache.spark.sql.Row and the DataRow interface.

Why it exists? Core feature logic uses DataRow. Spark gives us Row. We can't change either, so we adapt.

Key optimization: The schema field name → column index map is built once outside the UDF (for the entire DataFrame) and passed in. Inside the UDF, each get(fieldName) is O(1) map lookup instead of O(n) scan.

// BAD: rebuilds schema index for every row
val schemaIndex = row.schema.fieldNames.zipWithIndex.toMap  // inside UDF = O(n) per row

// GOOD: build once outside, reuse for all rows
val schemaIndex = df.schema.fieldNames.zipWithIndex.toMap  // outside UDF
val adapter = new SparkRowAdapter(row, schemaIndex)         // O(1) per row

---
TextUtils.scala

What is it? Text processing utilities for the Jaccard similarity features.

Business meaning: "Does this item's title overlap with what the user has been searching for?"

tokenize("Vintage Brass Desk Lamp - 1960s Style")
→ {"vintage", "brass", "desk", "lamp", "1960s", "style"}

tokenize("vintage brass lamp")
→ {"vintage", "brass", "lamp"}

jaccardSimilarity({"vintage","brass","desk","lamp","1960s","style"}, {"vintage","brass","lamp"})
= |intersection| / |union|
= 3 / 6
= 0.5

MaxQueryKeywordItemTitleJaccard7Days = max(jaccard(title, query) for each query in last 7 days)
→ "Best overlap between item title and any recent search" = 0.5

parseWithTimestamps: parses "query text:timestamp|query text:timestamp" and returns texts whose timestamp falls within the time window (7 days, 2 days, 15 days from eventTime).

---
SECTION 5: Component Communication Map

CTR SIDE DEPENDENCIES:
═══════════════════════

script_train.py
    ↓ imports pykrylov + krylov_tasks.py
krylov_tasks.py
    ↓ imports core (WideAndDeepModel, dataset)
    ↓ imports utils (initialise_hdfs, save_model, save_metrics, generate_dates)
core/custom_model.py
    ↓ imports core/custom_block.py (all 6 blocks)
    ↓ imports core/feature_schema.py (for input_signature)
core/custom_block.py
    ↓ imports core/custom_layer.py (7 layers)
    ↓ imports core/feature_groups.py (all group constants)
core/dataset.py
    ↓ imports core/feature_schema.py (PA_SCHEMA, TF_SIGNATURE)
    ↓ uses pyarrow.fs.HadoopFileSystem
utils/krylov_ems.py
    ↓ uses pykrylov.ems (save/load model artifacts)
    ↓ uses tensorflow.saved_model
utils/hadoop.py
    ↓ uses pyarrow.fs.HadoopFileSystem
    ↓ subprocess: hadoop classpath --glob

CVR SIDE DEPENDENCIES:
═══════════════════════

FeatureTransformPipeline (main)
    ↓ uses SparkFeatureOperators
    ↓ uses FeatureTransformer (from core)
    ↓ uses AdRankingFeatureNamespace (from core)
    ↓ uses SchemaBasedFeatureMapper (from core)
SparkFeatureOperators
    ↓ uses SparkRowAdapter (bridges Spark Row → DataRow)
    ↓ calls mapper.map() → SchemaBasedFeatureMapper
SparkRowAdapter
    ↓ implements DataRow (from core)
OnlineFeatureService
    ↓ uses FeatureTransformer (from core)
    ↓ creates MapBasedDataRow (from core)
FeatureTransformer
    ↓ uses SchemaBasedFeatureMapper
    ↓ uses RowAdapter (for explode)
    ↓ uses FeatureAggregator
SchemaBasedFeatureMapper
    ↓ uses AdRankingFeatureNamespace (calls each feature handler)
    ↓ uses FeatureMappingContext (caching)
AdRankingFeatureNamespace
    ↓ uses TextUtils (Jaccard, tokenize, parseWithTimestamps)
    ↓ reads placement_ids.csv (device detection)
    ↓ reads L2_ptr.csv + L3_ptr.csv (category PTR — stubs, pending categoryPath field)

---
SECTION 6: Execution Flow Traces

CTR Training — Complete Trace

User runs:
  $ python script_train.py --version Apr11a --as-of-date 2025-07-31 --num-days 20 --env corp

script_train.py:main()
  1. Parse CLI args → version="Apr11a", as_of_date=date(2025,7,31), num_days=20, env="corp"
  2. TaskVariables.from_env("corp") → namespace="ebay", parquet_template=PARQUET_TEMP_CORP, docker_image=...
  3. krylov.Task(task_object=task_train, args=[date, num_days, template], docker_image=...)
  4. task.add_packages(['core', 'utils'])  ← bundles local packages into Krylov job
  5. task.run_on_hadoop(batch_user='b_ebayadvertising')
  6. task.add_cpu(16), task.add_memory(128)
  7. task.add_task_parameter("exp_version", "Apr11a")
  8. session.submit_experiment(project='dis-train-pdpctr-vi', name='TRAIN-Apr11a-AS-OF-2025-07-31')

[Krylov schedules and runs task_train on a 16-CPU/128GB node]

krylov_tasks.py:task_train(as_of_date, num_days, parquet_template)
  1. Build model: WideAndDeepModel(hidden=(512,128), dropout=0.5, initial_bias=log(0.005/0.995))
  2. Set class_weight = {0: 0.5025, 1: 100}  ← click events weighted 100x heavier
  3. Compile: AdamW(lr=1e-4, wd=5e-3, clip=1.0), BinaryCrossentropy(from_logits=True)
  4. hdfs = initialise_hdfs()  ← sets up HADOOP_HOME, LD_LIBRARY_PATH, CLASSPATH
  5. For date in [2025-07-12, ..., 2025-07-31] (20 days, skipping missing):
     a. dataset = core.dataset('/apps/.../pd_ctr_feature_data_v6/dt=2025-07-12', hdfs)
        → list parquet files in HDFS dir
        → interleave(read 2 files parallel)
        → filter(click_flag in [0,1])
        → batch(2048, drop_remainder=True)
        → prefetch(AUTOTUNE)
     b. model.fit(dataset, verbose=2, class_weight={0:0.5025, 1:100})
        → For each batch of 2048 rows:
           i.   Anonymiser: mask personal fields if personalized_request=0
           ii.  6 processing blocks → dense, cross, sparse tensors
           iii. WideAndDeep: deep + wide_sparse + wide_cross logits → sum
           iv.  Loss = weighted BCE(logit, label)
           v.   Backprop AdamW
     c. save_metrics({'auc': 0.7234, 'binary_crossentropy': 0.0123}, '2025-07-12')
        → ems.record_metrics([{name:'auc', value:0.7234, dimensions:{date:20250712}}, ...])
  6. save_model(model)
     → tf.saved_model.save(model, export_dir, signatures=model.serving_fn)
     → shutil.make_archive(...)  → model.zip
     → ems.record_model(name='model.zip', file_path=...)

---
CVR Online Inference — Complete Trace

Ranking service receives HTTP request:
  RankingRequest(
    requestId = "drTiUhn6rTiK/102307",
    seedItem = RankingItem(listingId=123, totalCost=Some(79999.0), shippingCost=Some(0.0),
                           viewCount7Day=Some(847.0)),
    creativeItems = Seq(
      RankingItem(listingId=123, totalCost=Some(79999.0), shippingCost=Some(0.0)),
      RankingItem(listingId=456, totalCost=Some(59999.0), shippingCost=Some(999.0))
    ),
    userPricePref = Some(-0.52),
    userSearchQueriesTs = Some("vintage lamp:1747900000000|desk lamp:1747800000000")
  )

OnlineFeatureService.extractFeatures(request)
  1. buildDataRow(request):
     MapBasedDataRow(Map(
       "requestId"   → "drTiUhn6rTiK/102307",
       "placementId" → 102307L,   ← parsed from requestId suffix
       "eventTime"   → System.currentTimeMillis(),
       "item_listingIds" → Array(123L, 456L),
       "item_raw_NCalculatedTotalCost"    → Array(79999.0, 59999.0),
       "item_raw_NCalculatedShippingCost" → Array(0.0, 999.0),
       "item_raw_ViewCount_7Day"          → Array(847.0, 0.0),
       ... all other item_* arrays ...
       "seed_total_cost"      → 79999.0,
       "seed_shipping_cost"   → 0.0,    ← conditionally included (not 0.0 for "unknown")
       "seed_view_count_7day" → 847.0,
       "user_price_pref"      → -0.52,
       "user_search_queries_ts" → "vintage lamp:1747900000000|desk lamp:1747800000000"
     ))

  2. transformer.transformRow(row):

     a. mapper.resetGlobalContext()
        → globalMap.clear()  ← ensures no bleed from previous request on this thread

     b. RowAdapter.itemCount(row) → 2 (len of item_listingIds array)

     c. For idx=0 (item 123):
        RowAdapter.extractItemRow(row, 0):
          → MapBasedDataRow({
              ...all user_* and seed_* scalars copied from row...,
              "item_raw_NCalculatedTotalCost"    → 79999.0,  ← array[0]
              "item_raw_NCalculatedShippingCost" → 0.0,
              "item_raw_ViewCount_7Day"          → 847.0,
              "item_titles"                      → null,
              ... all other item fields at index 0 ...
            })

        mapper.map(itemRow0):
          ctx.resetLocal()  ← clear local cache
          For each of 70 features in implementedFeatures:

            "FullRecoPrice": $$map → row.getAs[Double]("item_raw_NCalculatedTotalCost").getOrElse(0.0)
              → 79999.0

            "PriceRatioNorm": $$ctx
              ctx.loadLocalAs[Double]("recoListingPrice") {
                row.getAs[Double]("item_raw_NCalculatedTotalCost").getOrElse(0.0) -
                row.getAs[Double]("item_raw_NCalculatedShippingCost").getOrElse(0.0)
              }  → 79999.0 - 0.0 = 79999.0  (cached in local map)
              seedListingPr = 79999.0 - 0.0 = 79999.0
              math.min(79999.0 / 79999.0, 30.0) → 1.0f

            "DeviceIsNative": $$ctx
              ctx.loadGlobalAs[Option[(Boolean,Boolean)]]("placementDevice") {
                placementDeviceMap.get(102307)  → Some((false, true))
              }  → cached globally (same for all items in request)
              → 1.0f (isNative=true)

            "MaxQueryKeywordItemTitleJaccard7Days": $$ctx
              cachedItemTitleTokens: tokenize(null) → Set() [empty, item has no title]
              → -1.0f (default: empty title)

            "ItemSoldCount": $$
              sold = row.getAs[Double]("item_raw_QuantitySold").getOrElse(0.0) → 0.0
              → 0.0f

          → features0 = Array[Float](70) = [79999.0f, ..., 1.0f, -1.0f, 0.0f, ...]

     d. For idx=1 (item 456):
        (similar, reuses global cache — DeviceIsNative lookup NOT repeated)

        "PriceRatioNorm": recoListingPrice = 59999.0 - 999.0 = 58999.0
          min(58999.0 / 79999.0, 30.0) → 0.7375f  ← different from item 0

        → features1 = Array[Float](70) = [59999.0f, ..., 1.0f, ...]

     e. FeatureAggregator.aggregate([features0, features1], featureNames, AggregationConfig):
        FullRecoPrice:    MEAN(79999.0, 59999.0) = 69999.0
        BothFreeShipping: MAX(1.0, 0.0) = 1.0  ← item 123 has free shipping
        PriceRatioNorm:   MEAN(1.0, 0.7375) = 0.8688
        DeviceIsNative:   FIRST(1.0, 1.0) = 1.0
        → aggregated = Array[Float](70)

  3. FeatureVector("drTiUhn6rTiK/102307", aggregated, featureNames)
     .get("PriceRatioNorm") → Some(0.8688f)
     .get("DeviceIsNative") → Some(1.0f)

---
SECTION 7: AI/ML Components Deep Dive

CTR Model — Wide & Deep

ARCHITECTURE RATIONALE:

Wide & Deep (Google, 2016) addresses the memorization vs. generalization tradeoff:

WIDE (memorization):
  Linear model on sparse features (one-hot categories, indicators)
  "If AdsGroupId=12345 → usually clicked" (memorizes specific patterns)
  sparse → Dense(1, zeros)  [zero init = no bias at start]
  sparse → Dense(1, glorot) [cross-product features]

DEEP (generalization):
  DNN on dense features (embeddings, normalized numericals)
  "Items with high category affinity + low price ratio tend to convert"
  (generalizes to unseen combinations)
  [512 → BN → ReLU → Dropout(0.5)] → [128 → BN → ReLU] → Dense(1)

COMBINED:
  logit = deep_logit + wide_sparse_logit + wide_cross_logit
  probability = sigmoid(logit + prior_logit)

  prior_logit = log(0.005 / 0.995) ≈ -5.3
  (Initial bias so model starts predicting 0.5% rather than 50%)
  (Also applied at serving time via logit_correction)

CTR Model — Training Strategy

CLASS IMBALANCE:
  Clicks are ~0.5% of impressions.
  Without correction: model learns to predict "no click" always.

  Solution: class_weight = {0: 0.5025, 1: 100}
  Click samples contribute 100x more to the loss.
  This is equivalent to upsampling clicks or downsampling non-clicks.

INCREMENTAL TRAINING:
  Train on day1, save weights.
  Train on day2 starting from day1 weights (not from scratch).

  This allows the model to:
  1. Remember patterns from historical data (weights persist)
  2. Adapt to recent trends (trained on fresh data last)
  3. Handle data volume (20 days × ~5GB/day = 100GB total processed in 20 steps)

AdamW vs Adam:
  weight_decay=5e-3 adds L2 regularization properly (fixed decoupled from adaptive lr).
  clip_norm=1.0 prevents gradient explosions from rare high-weight features.

BAYESIAN SMOOTHING (CTR Estimator):
  smoothed_ctr = (clicks + μ) / (impressions + 1)
  where μ = 0.005 (prior_ctr)

  Why: New ad groups have 0 impressions.
  Without smoothing: CTR = 0/0 = undefined.
  With smoothing: CTR = 0.005/1 = 0.5% (prior).
  When impressions grow: converges to actual CTR.

CVR Features — Key ML Design Choices

LEAF CATEGORY EMBEDDINGS (pre-trained, frozen):
  Category ID → 32-dim embedding from another system.
  "Frozen" means weights don't update during training.
  Why frozen: Category semantics (electronics vs. fashion) are stable.
  Why pre-trained: eBay already has high-quality category representations.

  Cross-product (user_cat_embed × seed_embed):
  If user browses electronics and seed item is electronics → high cross value.
  This is an explicit user-item category match signal.

JACCARD TEXT SIMILARITY:
  Simple but effective: no ML required.
  "User searched 'vintage lamp' → item titled 'Vintage Brass Desk Lamp' → 0.5 Jaccard"
  Works for zero-shot recall (item is new, no impression history).

PROXY IMPLEMENTATIONS:
  PL (Promoted Listings) specific features not in payload → use organic signals.
  This is a known approximation. When PL-specific data becomes available,
  the stub features will be updated to exact implementations.

CAUCHY PDF (PriceV2):
  Modeled after FeaturesRepertoire.priceV2Feature in simplark.
  Cauchy distribution is heavy-tailed → less sensitive to price outliers than Gaussian.
  "How likely is this reco price given the seed price?" → high score = similar price.

---
SECTION 8: Production Perspective

How CTR Serving Works in Production

Docker Container (hub.tess.io/promoted_display/ctr-model-serve:2026-04-12):
  Base: python:3.14.3-slim + tensorflow_model_server binary

  Entrypoint: /ml-app/entrypoint.sh
    → launches tensorflow_model_server
    → Port 1340: REST/gRPC scoring API
    → Port 5557: monitoring/metrics

  Config files:
    batching.config: enables request batching (accumulate N requests → single GPU inference)
    monitoring.config: exports latency + throughput metrics

  Health check: healthcheck.py
    → polls the TF Serving REST API
    → returns 0 (healthy) or 1 (unhealthy) for container orchestration

  TF SavedModel:
    model.serving_fn decorated with @tf.function(input_signature=...)
    → Fixed computation graph, compiled for fast inference
    → Input: {"AdsGroupId": [[id]], "seller_id": [[id]], ...}
    → Output: {"output": [[probability]]}

How CVR Feature Serving Works in Production

The feature-transform-online JAR is linked into the ranking service.
No separate network call needed — it's a library, not a microservice.

Typical usage:
  val service = OnlineFeatureService.create()  // called once at startup

  // For each ranking request:
  val features = service.extractFeatures(rankingRequest)
  val cvrScore = cvrModel.predict(features.toTensor)

Latency characteristics:
  - RowAdapter (explode): O(N_items) memory allocation
  - Feature extraction: O(N_items × 70 features)
  - Text parsing (Jaccard): O(|queries| × |title_tokens|) — but cached globally
  - Aggregation: O(N_items × 70)
  - Total target: <50ms for typical N_items=5-20

Thread safety:
  FeatureMappingContext is ThreadLocal — each request-processing thread is isolated.
  OnlineFeatureService itself is stateless (transformer is immutable after creation).
  Safe for concurrent use from multiple threads.

Offline Pipeline — Production Characteristics

Spark job configuration:
  --num-executors 50
  --executor-cores 2
  --executor-memory 8g
  --driver-memory 4g
  --conf spark.yarn.executor.memoryOverhead=2048

Why these numbers:
  - 2 cores per executor: limits concurrent heap pressure during JSON parsing
  - 8g executor memory: JSON parsing of large payloads is heap-intensive
  - 2048 MB overhead: native/JVM overhead from Jackson JSON parser
  - 50 executors: parallelism for the groupBy(requestId) shuffle

Performance bottlenecks:
  1. combineRequestFragments(): groupBy + collect_list → shuffle (most expensive)
  2. extractFeatures UDF: JSON string parsing + feature computation
  3. aggregateFeatures(): second groupBy → second shuffle

Data volume:
  ~500 input rows → ~200 output rows per sample (60% filter rate)
  Production: likely millions of rows → hundreds of millions of feature vectors

Logging and Observability

CTR side:
  - Training: model.fit(verbose=2) → per-batch loss/AUC to stdout (captured by Krylov)
  - Per-date metrics saved to Krylov EMS (queryable via dashboard)
  - Per-group AUROC printed to stdout at eval time

CVR side:
  - Pipeline: println() statements at each phase boundary
  - --debug flag: intermediate row counts (triggers full Spark jobs, production: off)
  - Warning: AdRankingFeatureNamespace prints to stderr if placement_ids.csv missing

Production gaps (identified in docs):
  - No per-feature value distribution monitoring
  - No feature extraction latency tracking
  - No alerting on data volume drops
  - No feature drift detection

Failure Modes and Risk Areas

1. TRAINING-SERVING MISMATCH (highest risk)
   If --aggregation MEAN used offline but not set online:
   → Training features and serving features will differ
   → Model silently degrades (no error, just wrong predictions)
   Mitigation: validate AggregationConfig matches between offline runs and online deployments

2. MISSING SIMPLEX STAGE (data quality)
   Requests without user signal are filtered out (filterMissingUserFeatures)
   → If Simplex pipeline has an outage, most training data disappears
   → Model trains on partial data, may overfit to non-personalized requests

3. STUB FEATURES RETURNING 0.0
   23 features always return 0.0
   → Model can't learn from them
   → When real implementations added, model must be retrained (feature values change dramatically)
   → A/B test required before production deployment of real implementations

4. NON-DETERMINISTIC FRAGMENT ORDERING
   combineRequestFragments uses first() on non-sorted data
   → "First" listingId occurrence may differ across Spark runs
   → Training data is slightly different each run (acceptable for training, not for offline evaluation)

5. GLOBAL CONTEXT BLEED (fixed, but must maintain)
   Fixed in June 2026 per DESIGN_DECISIONS.md
   If resetGlobalContext() is ever removed or bypassed:
   → Text parsing results from Request A bleed into Request B
   → Features for B are computed using A's user data
   → Silent wrong predictions

6. CATEGORY EMBEDDING STALENESS
   leaf_cat_pretrained_normal.npy is static (frozen in training docker image)
   → If eBay's category taxonomy changes, embeddings become stale
   → No automatic update mechanism visible in this repo

---
SECTION 9: Design Patterns and Architectural Decisions

Why Was It Designed This Way?

Pattern 1: DataRow Abstraction (Adapter Pattern)
Problem: Feature logic needs to run on Spark Row (offline) AND Map (online).
         Spark is 500MB of JARs — too heavy for online inference.

Solution: Define DataRow trait (4 methods).
          Core depends only on DataRow.
          SparkRowAdapter adapts Spark Row → DataRow (in pipeline module only).
          MapBasedDataRow wraps Map (in core module).

Result: Core JAR = ~250KB. Online JAR = ~500KB. No Spark.

Pattern 2: Strategy Pattern (Aggregation)
Problem: FullRecoPrice should be averaged, BothFreeShipping should be max'd,
         ItemSoldCount should be summed. Same operation (aggregate), different semantics.

Solution: AggregationConfig is a Map[FeatureName, Strategy].
          FeatureAggregator dispatches on strategy at runtime.

Benefit: Adding a feature = adding one line to AggregationConfig.
         Changing a feature's strategy = changing one line.
         Fail-fast validation: validateFeatures() at startup.

Pattern 3: DSL for Feature Definitions
Problem: How to make feature definitions readable to non-Scala developers?
         How to enable caching transparently without changing feature code?

Solution: 4 operators: $$, $$map, $$ctx, $$const
          Features written as pure functions on DataRow.
          Caching ($$ctx) is explicit but isolated from the feature logic itself.

Why not annotations or reflection magic?
  → Explicit caching ($$ctx) shows exactly which features share computation.
  → Pure functions are easy to unit test.
  → No runtime surprises.

Pattern 4: Thread-Safe Caching (ThreadLocal)
Problem: Parsing user search queries is expensive. Same queries appear on every item.
         Multiple threads process different requests concurrently.

Solution: ThreadLocal[Map] means each thread has its own cache.
          No locking, no contention.
          Reset between requests (online) or at requestId boundaries (offline).

Trade-off: Module-level lastRequestId ThreadLocal needed for Spark UDF serializability.
           (anonymous ThreadLocal subclass isn't serializable → class-level definition)

Pattern 5: Fail-Fast Validation
AggregationConfig.validateFeatures() called at FeatureTransformer creation.

Why: If you add a feature but forget its aggregation strategy, you want to know:
  - Offline: before the 10-minute Spark job runs
  - Online: at service startup, not during traffic

How: throws IllegalArgumentException with the missing feature name list.

---
SECTION 10: System Summary

What This System Does (Business Summary)

eBay runs a Promoted Stores advertising product. Store owners bid to appear prominently on the platform. To decide which ads to show and how prominently, eBay uses ML models that predict:

1. CTR: Probability this specific user clicks this specific ad group → used for ranking and impression allocation
2. CVR (feature pipeline): Computes 70 features per ranking request → feeds into a CVR model that predicts purchase probability

This repo builds and maintains:
- The Wide & Deep neural network for CTR prediction
- The complete feature transformation infrastructure for CVR prediction
- Training pipelines, evaluation, Docker images for deployment

---
Complete Dependency Map

┌─────────────────────────────────────────────────────────────────────────┐
│ PYTHON (CTR)                                                             │
│                                                                          │
│  feature_schema.py ←── dataset.py ←── krylov_tasks.py ←── script_train │
│         ↓                                      ↑                         │
│  custom_model.py ←── custom_block.py ←── feature_groups.py             │
│         ↓                    ↓                                           │
│  serving_fn          custom_layer.py                                     │
│                                                                          │
│  utils/hadoop.py ──→ krylov_tasks.py ←── utils/krylov_ems.py           │
│  utils/date.py ──────────────┘                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ SCALA (CVR)                                                              │
│                                                                          │
│  [CORE — no Spark]                                                       │
│  Feature.scala                                                           │
│  FeatureNamespace.scala (DSL)                                            │
│  DataRow.scala (trait) ←── MapBasedDataRow.scala                        │
│  FeatureMappingContext.scala                                             │
│  TextUtils.scala                                                         │
│  AdRankingFeatureNamespace.scala ←── placement_ids.csv, L2/L3_ptr.csv  │
│  AggregationConfig.scala                                                 │
│  FeatureAggregator.scala                                                 │
│  FeatureMapper.scala (SchemaBasedFeatureMapper)                          │
│  RowAdapter.scala                                                        │
│  FeatureTransformer.scala                                                │
│          ↗                           ↑                                   │
│  [ONLINE]                      [PIPELINE — Spark]                        │
│  RankingRequest.scala          SparkRowAdapter.scala ←── DataRow.scala  │
│  OnlineFeatureService.scala    SparkFeatureOperators.scala               │
│                                FeatureTransformPipeline.scala (main)    │
│                                   ← input-corp.sql, input-sddz.sql      │
└─────────────────────────────────────────────────────────────────────────┘

---
Request Lifecycle Diagram

OFFLINE (Training Data Generation):

  HDFS Parquet file
       │ (requestId, eventTime, seedItem, creativeItems, payload JSON, click, conversion)
       ▼
  FeatureTransformPipeline.main()
       │
       ▼ parseAndFlattenPayload()
  from_json(payload) → item_* arrays + simplex_stage
       │
       ▼ combineRequestFragments()
  groupBy(requestId) + flatten + dedup by listingId
       │
       ▼ filterMissingUserFeatures()
  drop if simplex_stage is empty
       │
       ▼ extractUserFeatures()
  simplex_stage(0) → user_price_pref, user_search_queries_ts, etc.
       │
       ▼ extractSeedFeatures()
  find seedItem in item_listingIds → seed_total_cost, seed_shipping_cost, etc.
       │
       ▼ filterToSurfacedItems()
  keep only items in creativeItems string
       │
       ▼ SparkFeatureOperators.explodeToItemRows()
  1 row (N items) → N rows (1 item each)
       │
       ▼ SparkFeatureOperators.extractFeatures(UDF)
  [per partition: sort by requestId]
  [at requestId boundary: mapper.resetGlobalContext()]
  for each row: SchemaBasedFeatureMapper.map() → Array[Float](70)
       │
       ▼ SparkFeatureOperators.aggregateFeatures()
  groupBy(requestId) + per-feature AggregationStrategy → Array[Float](70)
       │
       ▼ join(labels)
  → output: requestId, [70 float cols], click, conversion
       │
       ▼ write.parquet(outputPath)
  HDFS output

ONLINE (Real-Time Inference):

  Ranking Service
       │ new RankingRequest(...)
       ▼
  OnlineFeatureService.extractFeatures()
       │
       ▼ buildDataRow()
  RankingRequest → MapBasedDataRow (offline-compatible column names)
       │
       ▼ FeatureTransformer.transformRow()
       │
       ├─ mapper.resetGlobalContext() ← clear thread-local cache
       │
       ├─ for each item (0..N-1):
       │   │ RowAdapter.extractItemRow(row, i)
       │   │   → MapBasedDataRow (array[i] values as scalars)
       │   │ SchemaBasedFeatureMapper.map(itemRow)
       │   │   ctx.resetLocal()
       │   │   for each of 70 features:
       │   │       call handler ($$, $$map, or $$ctx)
       │   │   → Array[Float](70)
       │   └────────────────────────────────►  perItemFeatures
       │
       └─ FeatureAggregator.aggregate(perItemFeatures, featureNames, strategies)
              for each feature: apply MAX/MIN/MEAN/SUM/FIRST across N arrays
              → Array[Float](70)
       │
       ▼ FeatureVector(requestId, Array[Float](70), featureNames)
       │
       ▼ CVR Model Inference (external)
  conversion probability score

---
Top 20 Most Important Files

┌──────┬─────────────────────────────────────────────────────────────────────────────────┬─────────────────────────────────┐
│ Rank │                                      File                                       │        Why It's Critical        │
├──────┼─────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│ 1    │ cvr-modelling/feature-transform-core/src/.../AdRankingFeatureNamespace.scala    │ Defines all 70 CVR features.    │
│      │                                                                                 │ The heart of the system.        │
├──────┼─────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│      │                                                                                 │ The contract for all 143 CTR    │
│ 2    │ ctr-modelling/core/feature_schema.py                                            │ features. Change this and       │
│      │                                                                                 │ everything breaks.              │
├──────┼─────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│      │                                                                                 │ The actual training and         │
│ 3    │ ctr-modelling/krylov_tasks.py                                                   │ evaluation logic running on the │
│      │                                                                                 │  cluster.                       │
├──────┼─────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│ 4    │ cvr-modelling/feature-transform-pipeline/src/.../FeatureTransformPipeline.scala │ The offline Spark batch job.    │
│      │                                                                                 │ Produces all CVR training data. │
├──────┼─────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│      │                                                                                 │ The production CVR feature      │
│ 5    │ cvr-modelling/feature-transform-online/src/.../OnlineFeatureService.scala       │ service. Used in real-time      │
│      │                                                                                 │ ranking.                        │
├──────┼─────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│      │                                                                                 │ Maps every feature to its       │
│ 6    │ cvr-modelling/feature-transform-core/src/.../AggregationConfig.scala            │ aggregation strategy. Must be   │
│      │                                                                                 │ kept in sync.                   │
├──────┼─────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│ 7    │ ctr-modelling/core/custom_model.py                                              │ The full WideAndDeepModel       │
│      │                                                                                 │ class.                          │
├──────┼─────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│ 8    │ ctr-modelling/core/custom_block.py                                              │ The 6 processing blocks that    │
│      │                                                                                 │ build the feature tensors.      │
├──────┼─────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│ 9    │ cvr-modelling/feature-transform-core/src/.../FeatureTransformer.scala           │ Orchestrates Explode → Extract  │
│      │                                                                                 │ → Aggregate.                    │
├──────┼─────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│      │                                                                                 │ Thread-safe 2-level caching.    │
│ 10   │ cvr-modelling/feature-transform-core/src/.../FeatureMappingContext.scala        │ Critical for performance and    │
│      │                                                                                 │ correctness.                    │
├──────┼─────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│ 11   │ ctr-modelling/core/feature_groups.py                                            │ Routes each feature to the      │
│      │                                                                                 │ right processing block.         │
├──────┼─────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│ 12   │ cvr-modelling/feature-transform-pipeline/src/.../SparkFeatureOperators.scala    │ DataFrame operations: explode,  │
│      │                                                                                 │ extract UDF, aggregate.         │
├──────┼─────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│ 13   │ cvr-modelling/feature-transform-core/src/.../FeatureNamespace.scala             │ The DSL ($$, $$map, $$ctx,      │
│      │                                                                                 │ $$const).                       │
├──────┼─────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│ 14   │ cvr-modelling/feature-transform-core/src/.../TextUtils.scala                    │ Jaccard similarity + text       │
│      │                                                                                 │ parsing for 4 features.         │
├──────┼─────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│      │                                                                                 │ Low-level TF layers including   │
│ 15   │ ctr-modelling/core/custom_layer.py                                              │ BayesianSmoother and            │
│      │                                                                                 │ LeafCatEmbedding.               │
├──────┼─────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│ 16   │ cvr-modelling/feature-transform-online/src/.../RankingRequest.scala             │ The API contract for online     │
│      │                                                                                 │ feature requests.               │
├──────┼─────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│      │                                                                                 │ SchemaBasedFeatureMapper:       │
│ 17   │ cvr-modelling/feature-transform-core/src/.../FeatureMapper.scala                │ dispatches handlers per         │
│      │                                                                                 │ feature.                        │
├──────┼─────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│ 18   │ ctr-modelling/core/dataset.py                                                   │ HDFS Parquet → tf.data.Dataset  │
│      │                                                                                 │ pipeline.                       │
├──────┼─────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│      │                                                                                 │ Explains WHY every              │
│ 19   │ cvr-modelling/docs/DESIGN_DECISIONS.md                                          │ architectural choice was made.  │
│      │                                                                                 │ Read before refactoring.        │
├──────┼─────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┤
│      │                                                                                 │ The ground truth for CVR        │
│ 20   │ cvr-modelling/local-test-data/verify_features.py                                │ feature correctness. Run after  │
│      │                                                                                 │ every change.                   │
└──────┴─────────────────────────────────────────────────────────────────────────────────┴─────────────────────────────────┘

---
Interview-Style Q&A

Q: Why is there no normalization in the CVR feature pipeline?
A: Deliberate design choice (Decision #6 in DESIGN_DECISIONS.md). Feature transformation outputs raw values so different downstream models can apply their own scaling strategy. The CVR model training pipeline is responsible for normalization, not the feature extraction step. This maintains clear separation of concerns.

Q: What happens if you add a new CVR feature and forget to add its aggregation strategy?
A: AggregationConfig.validateFeatures() is called inside FeatureTransformer.createAdRankingTransformer(). This method throws IllegalArgumentException at startup with the missing feature name. The Spark job or online service fails immediately at startup — not mid-job.

Q: Why does FeatureMappingContext use containsKey instead of Option(map.get(key))?
A: Because map.get(key) returns null for a missing key AND for a stored null value. Option(null) is None, which would cause the compute block to re-execute even though null was intentionally stored. containsKey distinguishes "key absent" from "key present with null value."

Q: Why does the CTR model use logit_correction at serving time?
A: During training, the model is calibrated to a 50% click rate (via class_weight), not the real 0.5% rate. Without correction, it would predict absurdly high CTRs in production. logit_correction = log(0.005/0.995) ≈ -5.3 shifts the logit back to align with the real prior CTR.

Q: What's the difference between loadGlobal and loadLocal in FeatureMappingContext?
A: loadGlobal caches values that are the same for all items in a request (e.g., parsed user search queries — same user, same queries, for all N candidate items). loadLocal caches values that are specific to one item row but shared across multiple features within that item (e.g., tokenized item title — shared by 4 Jaccard features on the same item).

Q: Why are the seed item scalars extracted separately rather than just using items[0]?
A: The dump order of items in the payload is not guaranteed to match the seed position. The seed item is identified by seedItem listingId, which must be found by searching item_listingIds. The offline pipeline uses a UDF to find this index; the online service uses the explicit request.seedItem field. Using items[0] would produce wrong seed features whenever the seed wasn't dumped first.

Q: Why does the Spark UDF need a module-level ThreadLocal rather than a local anonymous class?
A: Spark serializes UDFs to send to executors. Anonymous classes (including anonymous ThreadLocal subclasses) are not serializable in Scala because they capture a reference to the enclosing scope. A module-level class (object companion) is fully serializable.

Q: What are the 23 stub features waiting for?
A: Each category needs different upstream data: (1) Embedding distances need an embedding join pipeline that adds item embeddings to the payload. (2) Recall source indicators need a recallSource field per item in the payload. (3) Category PTR features have their CSV files bundled already — they just need categoryPath added to each item's payload fields. (4) Merch features need merch-channel imp/click data. (5) Co-view/co-sale needs an item graph. (6) Boolean features (EPID match, RVI, cart) need their respective data in the Simplex user stage.

Q: How does the online service guarantee training-serving feature consistency?
A: OnlineFeatureService.buildDataRow() explicitly maps RankingRequest fields to column names that exactly match the offline pipeline's post-flatten schema (user_price_pref, item_raw_NCalculatedTotalCost, seed_total_cost, etc.). The same FeatureTransformer and AdRankingFeatureNamespace code runs on both paths. Aggregation strategy must be set identically (default or override).

Q: Why is LeafCatEmbedding frozen (trainable=False)?
A: The category embeddings come from a pre-trained model (not in this repo) that has learned rich semantic representations across all of eBay's category taxonomy. Freezing them prevents the CTR model's training (with limited labeled data) from corrupting these high-quality representations. The model learns how to use the embeddings (via the cross-product and dense layers), not the embeddings themselves.

---
Roadmap to Full Productivity

Week 1: Foundations

Day 1-2: Read and understand
  - README.md (top-level)
  - ctr-modelling/README.md
  - cvr-modelling/README.md
  - cvr-modelling/docs/DESIGN_DECISIONS.md  ← most important doc

Day 3: CTR Model
  - Read feature_schema.py (understand all 143 features)
  - Read feature_groups.py (understand processing categories)
  - Read custom_layer.py (especially BayesianSmoother, LeafCatEmbedding)
  - Run the CTR evaluation locally if you have access to a checkpoint

Day 4-5: CVR Feature Pipeline
  - Read AdRankingFeatureNamespace.scala (scan all 70 features)
  - Read AggregationConfig.scala
  - Run the local Spark test: spark-submit --env LOCAL --input-path sample01.parquet
  - Run verify_features.py and confirm ALL CHECKS PASSED

Week 2: Deep Dive

Day 1-2: CVR Pipeline internals
  - Read FeatureTransformPipeline.scala (trace all 5 phases)
  - Read SparkFeatureOperators.scala
  - Read FeatureMappingContext.scala (understand 2-level caching)

Day 3: Online module
  - Read OnlineFeatureService.scala
  - Read RankingRequest.scala
  - Manually trace a request through buildDataRow → transformRow → aggregate

Day 4-5: CTR Model training
  - Read krylov_tasks.py in full
  - Read dataset.py
  - Read custom_block.py
  - Study how class_weight and logit_correction work together


Week 3: Contributing

Try adding a new CVR feature (start with a simple $$map feature):
  1. Add val in AdRankingFeatureNamespace
  2. Add to implementedFeatures
  3. Add to AggregationConfig.strategies
  4. Build: mvn clean install -DskipTests
  5. Run spark-submit against sample01.parquet
  6. Add formula to verify_features.py
  7. Run verify_features.py → confirm 0 mismatches

Try running CTR evaluation on an existing checkpoint (if you have cluster access)

Read the docs/ folder systematically:
  - RAW_DATA_SCHEMA.md (understand the payload JSON structure)
  - IMPLEMENTATION_STATUS.md (see status of all 70 features)
  - ARCHITECTURE.md (architecture diagrams)

Key Questions to Investigate When You Join

1. Where does the CTR training data (pd_ctr_feature_data_v6) come from? What pipeline generates it?
2. Which service consumes the feature-transform-online JAR? How does it integrate?
3. When is the CVR pipeline run — daily? Weekly? How is it scheduled?
4. What's the deployment process for a new model version? Is there an A/B testing framework?
5. Which of the 23 stub features are actively being developed upstream?
6. What are the current production gAUC numbers by placement? What's the target?