def retriever_agent(query):
    return retrieve(query, index, texts)

def analyzer_agent(docs):
    return "\n".join(docs[:3])

def explainer_agent(query, context):
    return generate_answer(query, context)
