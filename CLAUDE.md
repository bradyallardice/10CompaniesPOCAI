# AI Tool Usage Extraction: Allardice/Kurer Roadmap (Hamptone Approach)

This methodology follows a three-stage approach to identify AI applications and exposure in job advertisements, adapted from Hamptone et al.

---

## Overview

The approach consists of three main stages:

1. **Simple keyword-matching** ("AI keywords") - Focus on AI development roles
2. **Tasks related to AI development** ("AI applications") - Identify AI use cases affecting broader workforce
3. **Classify AI exposure at task level** - Map all occupational tasks to exposure levels

---

## Stage 1: Simple Keyword-Matching ("AI Keywords") of Subset

### Rationale
Collect job ads concerned with active AI development, focusing on "AI workers."

### Empirical Approach

#### Data Processing:
1. Load 100 company sample from PostgreSQL database of Swiss jobs, credentials in config.env. 
2. Clean text to remove any special characters
3. Develop multilingual keyword list
3. Search text for multilingual keywords for AI

Use multilingual keyword list based on Hamptone et al. keywords to catch German/French/Italian in addition to English.

### Multilingual Keywords List

```
ai_development_keywords = [
        # ------------------------------------------------------------------
        # 0. UNIVERSAL ABBREVIATIONS & PROPER NAMES
        # ------------------------------------------------------------------
        "ai", "ml", "dl", "rl", "nlp", "nlu", "llm", "mlops",
        # libraries & clouds
        "pytorch", "tensorflow", "keras", "mxnet", "theano", "torch",
        "torch7", "caffe", "caffe2", "cntk", "deeplearning4j", "lasagne", "chainer",
        "scikit-learn", "sklearn", "weka", "mahout", "spark mllib", "h2o",
        "sagemaker", "vertex ai", "amazon machine learning", "google cloud ml",
        "azure ml", "ibm watson", "huggingface", "hugging face",
        # big-data / infra
        "hadoop", "spark", "mapreduce", "kafka", "storm", "flink",
        "mlflow", "kubeflow",
        "onnx", "onnxruntime",
        "gpu", "cuda", "cudnn", "opencl", "fpga",

        # ------------------------------------------------------------------
        # 1. ENGLISH CORE CONCEPTS, MODELS & ROLES
        # ------------------------------------------------------------------
        "artificial intelligence", "machine learning", "deep learning",
        "neural network", "neural networks", "convolutional neural network",
        "large language model", "language model",
        "computer vision", "natural language processing",
        "generative ai", "transformer", "transformers",
        "gpt", "bert", "clip",
        "cnn", "rnn", "lstm", "gru",
        "gan", "generative adversarial network", "vae", "autoencoder",
        "diffusion model", "stable diffusion", "Next Best Action",
        # roles & titles
        "ai engineer", "ai developer", "ai programmer",
        "machine learning engineer", "ml engineer", "deep learning engineer",
        "data scientist", "ml scientist", "research scientist", "applied scientist",
        "data mining engineer", "predictive modeler", "statistical modeler",
        "computer vision engineer", "nlp engineer", "mlops engineer",
        "model ops engineer", "cognitive computing engineer", "watson engineer",

        # ------------------------------------------------------------------
        # 2. LEARNING PARADIGMS
        # ------------------------------------------------------------------
        "supervised learning", "unsupervised learning", "semi-supervised learning",
        "self-supervised learning", "reinforcement learning",
        "few-shot learning", "zero-shot learning",
        "transfer learning", "meta learning",
        "active learning", "online learning",

        # ------------------------------------------------------------------
        # 3. CLASSICAL ML ALGORITHMS
        #  (English first – then German / IT / FR forms that recruiters use)
        # ------------------------------------------------------------------
        "logistic regression", "logistische regression", "regressione logistica",
        "régression logistique",
        "linear regression", "lineare regression", "regressione lineare",
        "régression linéaire",
        "decision tree", "entscheidungsbaum", "albero decisionale",
        "arbre de décision",
        "random forest", "zufallswald", "forêt aléatoire",
        "gradient boosting",
        "gboost", "xgboost", "lightgbm", "catboost", "adaboost", "bagging",
        "support vector machine", "svm", "machine à vecteurs de support",
        "naive bayes", "naiver bayes", "naïve bayes",
        "knn", "k-nearest neighbors", "k-nächste nachbarn",
        "k-means", "clustering",
        "principal component analysis", "pca",
        "hauptkomponentenanalyse", "analisi delle componenti principali",
        "analyse en composantes principales",
        "hidden markov model", "hmm",
        "verstecktes markov-modell", "modello di markov nascosto",
        "modèle de markov caché",
        "latent dirichlet allocation", "lda",
        "genetic algorithm", 
        "genetischer algorithmus", "algoritmo genetico", "algorithme génétique",

        # ------------------------------------------------------------------
        # 4. EARLY DEEP-LEARNING BUZZ (2010-2016)
        # ------------------------------------------------------------------
        "deep belief network", "dbn",
        "restricted boltzmann machine", "rbm",
        "self-organizing map", "som", "selbstorganisierende karte",
        "mappa auto-organizzante", "carte auto-organisatrice",

        # ------------------------------------------------------------------
        # 5. DEPLOYMENT / MLOps KEYWORDS
        # ------------------------------------------------------------------
        "model deployment", "modellbereitstellung",
        "deploy del modello", "déploiement de modèle",
        "model serving", "serving del modello", "serving de modèle",
        "model monitoring", "modellüberwachung",
        "monitoraggio del modello", "surveillance du modèle",

        # ------------------------------------------------------------------
        # 6. DOMAIN-SPECIFIC TASKS
        # ------------------------------------------------------------------
        "speech recognition", "spracherkennung",
        "riconoscimento vocale", "reconnaissance vocale",
        "asr",
        "audio processing", "audioverarbeitung",
        "elaborazione audio", "traitement audio",
        "text mining", "fouille de texte",
        "sentiment analysis", "sentiment-analyse",
        "analisi del sentiment", "analyse de sentiment",
        "information extraction", "informationsextraktion",
        "estrazione di informazioni", "extraction d'information",
        "recommendation system", "recommender",
        "empfehlungssystem", "sistema di raccomandazione",
        "système de recommandation",
        "predictive analytics", "prädiktive analytik",
        "analisi predittiva", "analytique prédictive",
        "object detection", "objekterkennung",
        "rilevamento oggetti", "détection d'objets",
        "image segmentation", "bildsegmentierung",
        "segmentazione delle immagini", "segmentation d'image",
        "hyperparameter tuning", "hyperparameteroptimierung",
        "ottimizzazione degli iperparametri",
        "optimisation des hyperparamètres",

        # ------------------------------------------------------------------
        # 7. RESPONSIBLE & TRUSTWORTHY AI 
        # ------------------------------------------------------------------
        "explainable ai", "xai", "interpretability",
        "responsible ai", "ethical ai", "fairness", "model governance",
        "erklärbare ki", "erklärbare künstliche intelligenz",
        "verantwortungsvolle ki", "ki-ethik",
        "intelligenza artificiale spiegabile", "ia spiegabile",
        "ia responsabile", "etica ia", "governance dei modelli",
        "intelligence artificielle explicable", "ia explicable",
        "ia responsable", "éthique de l'ia", "gouvernance des modèles",

        # ------------------------------------------------------------------
        # 8. LEGACY / MARKETING BUZZWORDS
        # ------------------------------------------------------------------
        "cognitive computing", "kognitives computing",
        "computing cognitivo", "informatique cognitive",
        "expert system", "expertensystem", "sistema esperto", "système expert",
        "knowledge engineering", "wissensengineering",
        "ingegneria della conoscenza", "ingénierie des connaissances",
        "predictive modeling", "prädiktive modellierung",
        "modellazione predittiva", "modélisation prédictive",
        "pattern recognition", "mustererkennung",
        "riconoscimento di pattern", "reconnaissance de formes"
    ]

```

### Caveat
"ai" in Italian creates many false positives ("assicurare il monitoraggio delle acque ai fini della sostenibilità..."). Need simple solution to drop these cases early.

### Expected Outcome
Identifies small subsample of job ads (~1%, depending on final keyword list). Most jobs are tech-intensive roles in IT development and similar fields. 

---

## Stage 2: Hand-Code Selected Job ads 

### Rationale
Create an evaluation set for future use

### Empirical Approach
1. Load in subsample of jobs from prior step
2. Deduplicate Jobs
3. Translate Raw text from source language to English
4. Output file for hand-coding that includes the Raw translated text and the raw untranslated text
5. Hand-code job ads based on the same instructions to be given to the LLM in the next phase.

### Expected Outcome
An file of hand-coded job tasks to be used as an evaluation for the next step

## Stage 3: Extract AI Tasks from Selected Job Ads 

Tasks Related to AI Development ("AI Applications")


### Rationale
Identify roles where AI may be utilized and applied, shifting focus from AI development to AI use cases affecting broader "non-AI workers."

### Empirical Approach
1. Use LLM-based identification and cleaning of relevant tasks (multi-step prompt adapted from Hamptone et al.) in 3 steps:
1. Extract AI applications and translate to English
2. Filter and Clean Applications
3. Filter once more


### Expected Outcome
List of AI applications in harmonized format. Average ~2 tasks per AI-developing job ad (range 1-5). Rough estimation: 150,000 unique tasks (11M job ads, <1% AI-developing jobs, average 2 tasks each, not all unique).

## Stage 4: Evaluate

### Rationale
Ensure that the model is performing well on the evaluation set

### Empirical Approach
1. Compare the number of tasks extracted from each job ad to the manual approach
2. Compare the overall similarity of the tasks extracted from each job to the manual approach
3. Manually inspect the differences between the manual and LLM approach

## Stage 5: Iterate Stages 3-4 Until Happy with Results

## Stage 6: Classify AI Exposure at Task Level

### Rationale
Classify all occupational tasks (regardless of AI relation) into exposed or not-exposed categories on this sample.

### Empirical Approach
- **Method**: Word embeddings, cosine similarity between identified "AI applications" and O*NET tasks
- **Benchmark**: Below/above 95th percentile similarity (following Hamptone approach)

### Expected Outcome
Every O*NET task classified as exposed or not-exposed. Enables calculation of exposure probability across:
- **Occupations** (bundle of tasks from O*NET, weighted by importance)
- **Firms** (within-occupation variation due to firm-level "AI applications")  
- **Time** (temporal variation in exposure)

### Key Equations

#### Equation 23: Firm-Level AI Application Similarity
Number of AI applications present in a given firm that are similar to a specific O*NET task.

#### Equation 24: Weighted Average Exposure  
Weighted average exposure across bundle of tasks relevant for specific occupation (in given firm, at given time).

#### Equation 26: AI Exposure Average
Multiply Equation 24 with logged number of AI applications by firm to get "AI Exposure Average" varying at occupationXfirmXtime level.

## Stage 7: Evaluate Results and Make Changes Accordingly

### Rationale
Before scaling up, we need to see if our results have face validity

## Stage 8: Repeat with Full Dataset

### Rationale 
The full dataset is what we are interested in, so once we are happy with the small results, we can scale it up further

---

## Applications and Analysis

### Data Integration
Merge "AI Exposure Average" with Swiss Household Panel individual respondents at occupationXfirmXtime level.

### Analysis Dimensions

#### Within occupations between firms [and over time]:
- Descriptive analysis demonstrating importance of firm-level variation
- Justifies job vacancy data approach vs. simple occupational measures

#### Within individual over time and firm:
- How individuals respond to changing AI exposure
- Outcomes: objective/subjective economic effects, political preferences, vote choice

#### Sources of Variation:
1. **Over time**: True "stayers" within same occupation and firm
2. **Across firms**: Individual changes firm but stays within occupation  
3. **Across occupations**: Job transitions over time

---

## Implementation Notes

- Use PostgreSQL database for Swiss jobs 
- Implement multilingual text processing
- Create labeled evaluation dataset (100 companies, ~302 jobs)
- Apply three-step LLM processing pipeline
- Calculate embeddings and similarity measures for O*NET integration
- Merge with Swiss Household Panel for individual-level analysis

## Important Implementation Guidelines

### Keyword List Authority
**CRITICAL**: The `ai_development_keywords` list in `step_1_keyword_match.py` is the ONLY authoritative keyword list for this project. No other keyword files (Data/keywords/*.txt) should be used or considered. The ai_development_keywords list contains the complete multilingual keyword set based on Hamptone et al. methodology.

### Code Execution and Changes
**MANDATORY**: All code execution and changes must be approved by the user before implementation. This includes:
- Running any scripts or pipeline steps
- Modifying existing code files
- Creating new files
- Making changes to keyword lists or detection logic
- Database queries or data processing

Always consult with the user before proceeding with any code execution or modifications.

### File Management
**CRITICAL**: Never create new files. Always modify existing files instead. When improving functionality:
- Edit the original file (e.g., `step_1_keyword_match.py`)
- Do not create new versions or variations
- Make incremental improvements to existing code
- Preserve the original file structure and naming

**CRITICAL**: Never rerun step 0 in a way that would overwrite the sample of jobs from 100 companies. Always just use the already saved .csv instead.

### Chat‑Model Interaction Efficiency
**MANDATORY**: When requesting code edits from Claude (or any chat LLM), enforce an *efficient‑diff* workflow to avoid unnecessary token usage.

1. **Diff‑Only Output** – Claude must return a unified diff (`--- a/… +++ b/…`) showing *only* the modified lines and their immediate context. Unchanged code blocks are never re‑printed.  
2. **Minimal Prompts** – After each diff, the model should ask a single yes/no question (“Apply? (y/n)”) with no additional commentary unless explicitly requested.  
3. **Batch Identical Edits** – If the same change is needed in multiple locations, Claude must aggregate them into one combined diff and ask for approval once.  
4. **No File Re‑Inlining** – Once the source file has been uploaded or previously shown, Claude must reference it by path or name rather than re‑sending its full contents.  
5. **Patch‑Only Workflow** – Upon approval, Claude supplies a standalone patch that applies cleanly with `git apply`, and does **not** resend the full updated file.

---

**Version**: Updated methodology based on Hamptone et al. approach
**Date**: July 2025