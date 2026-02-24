# AI RAG & Agents Course

A comprehensive beginner-to-advanced curriculum covering AI/ML fundamentals, NLP, transformers, RAG systems, and AI agents across 14 modules.

## Curriculum

```
Beginner                    Intermediate                      Advanced
────────────────────────────────────────────────────────────────────────
M1 AI/ML Fundamentals
M2 Neural Networks
M3 NLP Fundamentals
  M4 Word Vectors            M5 Modern Embeddings
  M6 Text Generation         M7 Transformers (a + b)
                              M8 Prompt Engineering
                              M9 RAG Foundations              M11 Advanced RAG
                              M10 RAG Pipeline
                              M12 Intro to Agents             M13 Advanced Agents
                                                              M14 Agentic RAG (Capstone)
```

## Modules

### Module 1: AI & Machine Learning Fundamentals
**`01_ai_ml_fundamentals.ipynb`** | Beginner

AI landscape from rule-based systems to generative AI. Supervised vs unsupervised vs reinforcement learning. Linear regression from scratch with gradient descent, including loss curve and contour plot visualizations. Logistic regression on the Iris dataset. Evaluation metrics: accuracy, precision, recall, F1, confusion matrices. Comparison across train/test split ratios.

### Module 2: Neural Networks & Deep Learning Basics
**`02_neural_networks_basics.ipynb`** | Beginner

The perceptron and multi-layer networks. Activation functions (sigmoid, ReLU, tanh) visualized with derivatives. Forward pass and backpropagation walkthrough. Loss functions (MSE, cross-entropy). PyTorch introduction: tensors, autograd, nn.Module. MNIST digit classification with a simple MLP. Exercises: XOR with a 2-layer network from scratch, experimenting with hidden layer sizes, comparing activation functions.

### Module 3: NLP Fundamentals
**`03_nlp_fundamentals.ipynb`** | Beginner

Text preprocessing pipeline: lowercasing, punctuation removal, tokenization (whitespace, regex, NLTK). Stop words, Porter stemming, WordNet lemmatization. Bag-of-Words from scratch and with CountVectorizer. TF-IDF from scratch and with TfidfVectorizer. N-grams (unigrams, bigrams, trigrams). Text classification with TF-IDF + logistic regression. Limitations of classical NLP as motivation for word vectors.

### Module 4: Word Vectors & Embeddings
**`word_vec_visualization.ipynb`** | Beginner-Intermediate

Why dense vectors beat sparse representations. Pre-trained GloVe embeddings (100d) via Gensim. Semantic similarity (`most_similar`). Word analogies with vector arithmetic (man:king :: woman:queen). t-SNE visualization of word clusters by category. Exercises: finding analogies that fail, exploring gender bias in embeddings with profession projections.

### Module 5: Modern Embeddings
**`llm-embeddings.ipynb`** | Intermediate

Sentence and document-level embeddings using API models (Cohere `embed-english-v3.0`, OpenAI `text-embedding-ada-002`) and local models (E5-base-v2). Batch processing 41K arxiv paper chunks. t-SNE + K-means clustering on IMDB reviews. Cosine similarity comparison across text pairs. Semantic search exercise over the arxiv dataset. Bridge to RAG: how these embeddings power retrieval systems.

### Module 6: LLM Text Generation
**`llm_text_generation.ipynb`** | Intermediate

How LLMs generate text token by token using GPT-2. Tokenization, logits, softmax probabilities. Greedy decoding with step-by-step trace. Temperature sampling and its effect on creativity vs coherence. Top-k sampling. Exercises: top-p (nucleus) sampling, stopping criteria (EOS, sentence boundaries), batch generation of multiple continuations.

### Module 7a: Self-Attention Mechanism
**`self-attention.ipynb`** | Intermediate-Advanced

Self-attention from scratch, step by step. Dot-product attention scores and softmax normalization. Context vectors as weighted sums. Query/Key/Value projections with learnable weight matrices. Scaled dot-product attention (dividing by sqrt(d_k)). Causal masking for autoregressive models. Multi-head attention: splitting into heads, parallel computation, concatenation. Visualization of attention weight heatmaps. Comparison with PyTorch's built-in MultiheadAttention.

### Module 7b: Transformer Architecture
**`07b_transformer_architecture.ipynb`** | Intermediate-Advanced

Sinusoidal positional encoding with heatmap visualization. Layer normalization vs batch normalization. Residual connections for gradient flow. Feed-forward network layer (expansion and projection). Complete encoder block assembly in PyTorch. Decoder block with masked self-attention and cross-attention. Architecture variants: BERT (encoder-only) vs GPT (decoder-only) vs T5 (encoder-decoder). Pre-training objectives: masked LM, causal LM, span corruption.

### Module 8: Prompt Engineering
**`08_prompt_engineering.ipynb`** | Intermediate

Prompt anatomy: system, user, and assistant messages. Zero-shot and few-shot prompting with comparison. Chain-of-thought prompting for math and reasoning problems. Role prompting to control output style. Structured output: getting reliable JSON from LLMs. Common failures (hallucination, instruction following) and mitigations. Iterative prompt refinement workflow. Prompt evaluation harness comparing multiple variants.

### Module 9: RAG Foundations
**`09_rag_foundations.ipynb`** | Intermediate

Why RAG: knowledge cutoff, hallucination, domain-specific needs. RAG architecture overview. Document chunking strategies: fixed-size, sentence-based, recursive character splitting. Embedding documents with sentence-transformers. ChromaDB: creating collections, adding documents, semantic search. Distance metrics: cosine similarity, dot product, Euclidean. Metadata filtering. Overview of other vector databases (Pinecone, FAISS, Weaviate).

### Module 10: RAG Pipeline
**`10_rag_pipeline.ipynb`** | Intermediate-Advanced

End-to-end RAG built from scratch (no framework): query embedding, vector search, context building, LLM generation. Chain strategies: "stuff" (concatenate all), "map-reduce" (summarize individually, combine), and "refine" (iterative improvement). Source attribution with numbered references. Multi-turn conversation with memory. Comparison with LangChain's RetrievalQA.

### Module 11: Advanced RAG
**`11_advanced_rag.ipynb`** | Advanced

RAG evaluation metrics: precision@k, recall@k, MRR, faithfulness scoring. Evaluation dataset with ground-truth relevant documents. HyDE (Hypothetical Document Embeddings): generating hypothetical answers for better retrieval. Cross-encoder re-ranking with `ms-marco-MiniLM-L-6-v2` for two-stage retrieval. Hybrid search: BM25 keyword matching + vector search combined via Reciprocal Rank Fusion. Failure analysis and debugging checklist.

### Module 12: Introduction to AI Agents
**`12_agents_introduction.ipynb`** | Intermediate

What agents are: autonomous systems using tools to achieve goals. The agent loop: Observe, Think, Act, Observe. Building tools with LangChain's `@tool` decorator. Agent types: ReAct agents and tool-calling agents. ReAct pattern: Thought-Action-Observation traces. Error handling and retry logic. Comparison of deprecated vs modern LangChain APIs.

*Historical reference: `langchain_agent.ipynb` (deprecated APIs)*

### Module 13: Multi-Agent Systems & Advanced Patterns
**`13_advanced_agents.ipynb`** | Advanced

Multi-agent patterns: sequential, parallel, hierarchical, debate. ReAct implementation from scratch without frameworks. Agent memory: buffer, summary, and entity memory. Agent planning: decomposing complex tasks into sub-tasks. Self-reflection: generate, critique, revise loop. LangGraph introduction: workflows as state machines with conditional routing. Debate system: two agents argue, one judges.

*Existing example: `ai_trading_agent.ipynb` (5-agent sequential chain using Perplexity AI)*

### Module 14: Agentic RAG (Capstone)
**`14_agentic_rag.ipynb`** | Advanced

Retrieval as an agent tool with OpenAI function calling. Multi-source retrieval: agent chooses between knowledge base, web search, and calculator. Query routing: classifying questions to select the right source. Corrective RAG (CRAG): retrieve, evaluate relevance, re-retrieve or fall back if needed. Self-RAG: deciding when retrieval is needed and checking answer faithfulness. Full agentic RAG pipeline combining routing, retrieval, evaluation, and self-checking. Capstone exercise: end-to-end AI assistant.

## Setup

### Environment

The notebooks use a conda environment with Python 3.12:

```bash
conda create -n llm python=3.12
conda activate llm
```

### Dependencies

```bash
pip install numpy matplotlib scikit-learn torch torchvision transformers \
    sentence-transformers gensim nltk datasets chromadb rank-bm25 \
    openai cohere langchain langchain-openai langchain-community langgraph \
    python-dotenv tqdm beautifulsoup4 requests
```

Each notebook also includes its own `!pip install` cell for specific dependencies.

### API Keys

Create a `.env` file at `/home/amir/source/.env` with:

```
OPENAI_API_KEY=your-key-here
CO_API_KEY=your-cohere-key-here
PPLX_API_KEY=your-perplexity-key-here
```

Modules 1-7 run entirely locally (no API keys needed). Modules 8-14 require an OpenAI API key. Module 5 optionally uses Cohere.

## Notebook Conventions

- **Exercise pattern**: Markdown explanation → code cell with `TODO` stubs and `None` placeholders → `### Solution` heading → working solution code
- **First cell**: `!pip install -q ...` for dependencies
- **API keys**: loaded via `python-dotenv`
- **Device detection**: `device = "cuda" if torch.cuda.is_available() else "cpu"`
- **Visualizations**: matplotlib
- **Progress bars**: tqdm for batch loops
