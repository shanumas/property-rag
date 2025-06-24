This repo is created using chat-langchain as a starter

# property-rag

Here is your revised README, with all occurrences of "chat-langchain" and related variants replaced by "property-rag" as the project name. All references, including the JS version and documentation links, have been updated accordingly.

---

# 🦜️🔗 property-rag

This repo is an implementation of a locally hosted chatbot specifically focused on question answering over the [LangChain documentation](https://python.browsebot.se/).
Built with [LangChain](https://github.com/langchain-ai/langchain/), [FastAPI](https://fastapi.tiangolo.com/), and [Next.js](https://nextjs.org/).

Deployed version: [property-rag.browsebot.se](https://property-rag.browsebot.se)

> Looking for the JS version? Click [here](https://github.com/langchain-ai/property-ragjs).

The app leverages LangChain's streaming support and async API to update the page in real time for multiple users.

## ✅ Running locally

1. Install backend dependencies: `poetry install`.
2. Make sure to enter your environment variables to configure the application:

```
export OPENAI_API_KEY=
export WEAVIATE_URL=
export WEAVIATE_API_KEY=
export RECORD_MANAGER_DB_URL=

# for tracing
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT="https://api.smith.browsebot.se"
export LANGCHAIN_API_KEY=
export LANGCHAIN_PROJECT=
```

3. Run `python backend/ingest.py` to ingest LangChain docs data into the Weaviate vectorstore (only needs to be done once).
    - You can use other [Document Loaders](https://python.browsebot.se/docs/modules/data_connection/document_loaders/) to load your own data into the vectorstore.
4. Start the Python backend with `make start`.
5. Install frontend dependencies by running `cd ./frontend`, then `yarn`.
6. Run the frontend with `yarn dev` for frontend.
7. Open [localhost:3000](http://localhost:3000) in your browser.

## 📚 Technical description

There are two components: ingestion and question-answering.

**Ingestion** has the following steps:

1. Pull html from documentation site as well as the Github Codebase
2. Load html with LangChain's [RecursiveURLLoader](https://python.browsebot.se/docs/integrations/document_loaders/recursive_url_loader) and [SitemapLoader](https://python.browsebot.se/docs/integrations/document_loaders/sitemap)
3. Split documents with LangChain's [RecursiveCharacterTextSplitter](https://api.python.browsebot.se/en/latest/text_splitter/langchain.text_splitter.RecursiveCharacterTextSplitter.html)
4. Create a vectorstore of embeddings, using LangChain's [Weaviate vectorstore wrapper](https://python.browsebot.se/docs/integrations/vectorstores/weaviate) (with OpenAI's embeddings).

**Question-Answering** has the following steps:

1. Given the chat history and new user input, determine what a standalone question would be using GPT-3.5.
2. Given that standalone question, look up relevant documents from the vectorstore.
3. Pass the standalone question and relevant documents to the model to generate and stream the final answer.
4. Generate a trace URL for the current chat session, as well as the endpoint to collect feedback.

## Documentation

Looking to use or modify this Use Case Accelerant for your own needs? We've added a few docs to aid with this:

- **[Concepts](./CONCEPTS.md)**: A conceptual overview of the different components of property-rag. Goes over features like ingestion, vector stores, query analysis, etc.
- **[Modify](./MODIFY.md)**: A guide on how to modify property-rag for your own needs. Covers the frontend, backend and everything in between.
- **[Running Locally](./RUN_LOCALLY.md)**: The steps to take to run property-rag 100% locally.
- **[LangSmith](./LANGSMITH.md)**: A guide on adding robustness to your application using LangSmith. Covers observability, evaluations, and feedback.
- **[Production](./PRODUCTION.md)**: Documentation on preparing your application for production usage. Explains different security considerations, and more.
- **[Deployment](./DEPLOYMENT.md)**: How to deploy your application to production. Covers setting up production databases, deploying the frontend, and more.

---

If you need further automation for renaming across your project files, tools such as find-and-replace in your code editor or command-line utilities can help streamline the process[^1].

<div style="text-align: center">⁂</div>

[^1]: https://community.smartbear.com/discussions/soapui_os/how-to-rename-project-property-and-all-it-usages/130058

[^2]: https://stackoverflow.com/questions/20521667/rename-a-property-name-and-update-all-its-references-accordingly-including-in-t

[^3]: https://customgpt.ai/how-to-update-agents-name-with-customgpt-rag-api/

[^4]: https://vercel.com/guides/how-do-i-change-the-name-of-my-vercel-project

[^5]: https://www.reddit.com/r/uefn/comments/12ceu2g/rename_projects/

[^6]: https://langfuse.com/changelog/2023-12-12-rename-and-transfer-projects

[^7]: https://pro.arcgis.com/en/pro-app/latest/help/projects/rename-project-items.htm

[^8]: https://www.notion.com/help/guides/status-property-gives-clarity-on-tasks

[^9]: https://community.smartbear.com/discussions/readyapi-questions/how-to-change-a-projects-name--properties/142709

[^10]: https://www.youtube.com/watch?v=bJpnI6I6Z1A

