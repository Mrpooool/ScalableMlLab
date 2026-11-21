# **Lab Assignment 2** Retrieval-Augmented Generation for Question Answering

Large language models can answer many questions from what they learned during training, but they hallucinate, go out of date, and cannot point to a source. Retrieval-augmented generation (RAG) addresses this by retrieving relevant passages from a document collection and giving them to the model as context. Your task is to build a RAG system for question answering, measure each of its components separately, and extend it to questions that need information from more than one document.

You will work in the provided [notebook](https://colab.research.google.com/drive/1q9LyIjUNKE2r26RBEgOIfbbz78Yu1IQo?authuser=1#scrollTo=2ea1BiB333VS) on a free Colab T4 GPU. Data loading, evaluation metrics, the LLM wrapper and the result tables are provided; you implement the components marked TODO. You are provided with the following datasets, which the notebook downloads automatically.

| Dataset | Type | Characteristics | Link |
| ----- | ----- | ----- | :---: |
| SQuAD v1.1 (dev) | Single-hop QA | 300 questions over 48 Wikipedia articles. Each answer is found in one paragraph. | [Link](https://huggingface.co/datasets/rajpurkar/squad) |
| HotpotQA (distractor, dev) | Multi-hop QA | 200 questions over about 5,000 Wikipedia paragraphs. Each answer needs two paragraphs. | [Link](https://huggingface.co/datasets/hotpotqa/hotpot_qa) |

Single-hop questions look like *“When was Isaac Newton born?”*: one document contains the answer. Multi-hop questions look like *“In what year was the university where Sergei Aleksandrovich Tokarev was a professor founded?”*: you first need Tokarev’s page to learn that the university is Moscow State University, and only then can you find the page that says it was founded in 1755\.

At the end of this week, your system should

> * index a Wikipedia corpus in a Chroma vector database,  
> * retrieve evidence for a question with dense, keyword and hybrid search,  
> * answer questions with an LLM served by vLLM, with and without retrieved context,  
> * report retrieval quality, answer quality, latency and cost for every system,  
> * answer multi-hop questions by retrieving iteratively.

## Task 1\. Chunk the Corpus

Before documents can be indexed, they must be split into chunks: the units the retriever returns and the LLM reads. Paragraph chunking is provided. Implement two more strategies:

> * groups of *n* sentences, where consecutive chunks share *m* sentences,  
> * fixed-size windows of about *N* characters with an overlap, without cutting words in half.

In your report, discuss what can go wrong at chunk boundaries, and how overlap addresses it.

## Task 2\. Build the Vector Database

Store the chunked corpus in a vector database so that it can be searched by meaning. Implement

> * an indexing function that chunks the corpus, embeds every chunk, and stores the embeddings with their metadata (source document and position) in a Chroma collection that uses cosine distance,  
> * a retrieval function that returns the top-*k* chunks for a batch of queries.

Compare at least two of the provided embedding models (MiniLM, BGE-small and E5-small). In your report, explain why some embedding models expect “query:” and “passage:” prefixes, and what happens if you leave them out.

## Task 3\. Evaluate Retrieval

Every question comes with its gold evidence, so retrieval can be evaluated before any LLM is involved. For at least three chunking strategies (including overlap versus no overlap) and at least two embedding models, measure on both datasets

> * recall@*k*: the share of the evidence found in the top *k* chunks,  
> * all@*k*: the share of questions for which all of the evidence was found,  
> * mean reciprocal rank (MRR),  
> * the amount of text retrieved.

Also implement neighbour expansion, which adds the chunks before and after each retrieved chunk. In your report, describe and justify the following:

> 1. Which chunking strategy and embedding model work best for each dataset, and why?  
> 2. Larger chunks always get a higher recall@*k*. How do the strategies compare when they retrieve a similar amount of text?  
> 3. Why is all@*k* much lower than recall@*k* on HotpotQA?  
> 4. Which value of *k* do you choose, with and without neighbour expansion?

## Task 4\. Build Question Answering Systems

Implement and compare the following systems on both datasets. All of them use the same reader LLM (Qwen2.5-3B-Instruct, served with vLLM):

> * an LLM-only baseline without retrieval (provided),  
> * RAG with top-*k* dense retrieval, with *k* chosen by answer quality,  
> * RAG with re-ranking: retrieve 20 to 30 candidates and re-rank them with a cross-encoder,  
> * RAG with query rewriting: the LLM rewrites the question into a search query before retrieval,  
> * RAG with hybrid retrieval: BM25 keyword search and dense retrieval, fused with Reciprocal Rank Fusion.

In your report, describe and justify the following:

> 1. Which value of *k* did you choose, and does more context ever hurt?  
> 2. Which retrieval improvement helps most on SQuAD, and which on HotpotQA? Why?

## Task 5\. Evaluate the Systems

A single accuracy number hides where a RAG system fails. For every system in Task 4, measure

> * retrieval quality of the context given to the LLM,  
> * answer quality: exact match, token F1 and lenient accuracy,  
> * end-to-end accuracy, split by whether the evidence was retrieved,  
> * latency per question (median and 95th percentile),  
> * cost: tokens per question and GPU-hours, extrapolated to the full dev set.

Then analyse the errors. Take at least three incorrect answers and classify each as a retrieval error, a reasoning error, a long-context error or an evaluation artefact (a correct answer phrased differently), with evidence for each. Run two controlled ablations:

> * a reader ablation, which gives the same LLM no context, the retrieved context, or only the gold paragraphs,  
> * a long-context ablation, which places the gold paragraphs among a growing number of distractors, at the start, middle or end of the context.

In your report, describe and justify the following:

> 1. Is your best system worth its extra latency and cost?  
> 2. What does the accuracy on questions where retrieval missed the evidence tell you about the model?  
> 3. How much accuracy is lost to retrieval, and how much to the reader?  
> 4. Does the reader get worse with more context, or with the position of the evidence?

## Task 6\. Answer Multi-hop Questions

Single-shot retrieval often misses the second paragraph of a multi-hop question, because that paragraph can only be identified after reading the first one. Implement a system that, for each question,

> 1. writes a search query for the first missing piece of information and retrieves,  
> 2. summarises the retrieved passages into notes and writes the next query,  
> 3. repeats step 2 once more, then answers from the notes and all retrieved passages.

Run each step for all questions in one batched LLM call. In your report, describe and justify the following:

> 1. Does iterative retrieval improve retrieval and accuracy over single-shot RAG with a similar number of retrieved chunks?  
> 2. How do the results differ between bridge and comparison questions?  
> 3. What does the loop cost in latency and tokens?  
> 4. Show one trace where the loop found the missing paragraph, and one where it went wrong.

## Deliverables

Each group should submit

> 1. The completed notebook, with all outputs, runnable from top to bottom on a Colab T4.  
> 2. A \[3-5\] page report that answers the questions above. It should contain the retrieval comparison, the final results table for both datasets, the error analysis with at least three worked examples, the ablation results, and the multi-hop results with example traces.