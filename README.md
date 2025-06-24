This repo is created with chat-langchain as its starter

# property-rag is a semantic search chatbot designed for real estate property search in London. It leverages a hybrid retrieval approach to deliver highly relevant property matches:

Hybrid Search Workflow:
The system first applies pre-filtering using essential property metadata such as the number of bedrooms, bathrooms, and other basic attributes to narrow down the results set. This ensures that only properties meeting the user's must-have criteria are considered.

Dense Semantic Search:
After initial filtering, property descriptions and user queries are embedded using the text-embedding-3-small model. A dense vector search is then performed against the filtered set, capturing nuanced intent and context from the user's query for more accurate matches.

RAG Database with Weaviate:
Weaviate serves as the Retrieval-Augmented Generation (RAG) database, efficiently storing and retrieving property embeddings to enable fast and scalable semantic search.

Reranking:
The system reranks the candidate properties based on semantic similarity, ensuring that the most relevant listings appear at the top of the results.

This architecture enables property-rag to provide fast, accurate, and context-aware property recommendations, streamlining the search experience for buyers and renters in London
