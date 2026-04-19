# RAG - Retrival Augmented Generation

<div style="background-color: #f0f0f0; padding: 15px;">
<strong><h2>Introduction</h2></strong>
</div>

Technique that enhances LLM responses by ; 
 - **retrieving relevant information from external knowledge sources**
 - using that retrieved context to generate more accurate, grounded answers.
 - The system converts queries into embeddings, performs similarity search to fetch the most relevant documents, and injects this context into the LLM prompt before generation.

This approach **reduces hallucinations**, enables access to current or proprietary information without retraining, and provides verifiable sources for model outputs. It's essentially giving the LLM access to a dynamic external memory it can reference before responding.

<div style="background-color: #f0f0f0; padding: 15px;">
<strong><h2>Challenges</h2></strong>
</div>

 - How do we deal with the missing content ? Indexing
 - Missed the top ranked documents - saving metadata of document; chunk size, embedding strategy, retrieval strategy, context size
 - Several documents are retrieved but only few make it into a final context - Training a retrieval model and use high context window model.
 - Context is not being extracted - fine tune the model
 - Wrong format ( table for example)
 - Incorrect specificity - interactive query generation
 - Incomplete retrieval/response - additional training on summarization.
 - Speed of retrieval(optimizing tokenization, encoding documents)
 - Safety - rag with poisoned documents
 - Bias & Privacy - Documents with personal details or biases. 


<div style="background-color: #f0f0f0; padding: 15px;">
<strong><h2>Prompting Techniques to reduce Hallucinations</h2></strong>
</div>

### Chain of Thought 

**Method**:Guiding the model through a series of logical operations or steps helps it better understand how to approach each of the queries, compared to a standard prompt, which fails to give the correct response. 
 - issue : Result is depending on the intermediary steps. So if any of the steps are flawed, you will end up with incorrect response
 - error propagates through the chain

   
* https://www.prompthub.us/blog/chain-of-thought-prompting-guide


### Thread of Thought 
 - works well when retrieved information is chaotic
 - **method**
   - go step by step
   - summarize each step
   - analyse each step
   
* https://learnprompting.org/docs/advanced/thought_generation/thread_of_thought

### Chain of Note
**Need of Chain of Note**
 - not retrieving irrelevant information
 - failing to use their inherent knowledge
 - struggle to determine when they have sufficient information for accurate answers, often not indicating when knowledge is lacking.

**Method**: 
CON constructs reading notes based on the relevance of the documents to the query. It directly answers from relevant documents, infers answers using context from partially relevant documents, and defaults to “unknown” if documents are irrelevant or insufficient for a response.


