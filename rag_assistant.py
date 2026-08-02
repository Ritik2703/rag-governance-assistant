from anthropic import Anthropic

class RAGAssistant:
    def __init__(self, docs: list):
        self.client = Anthropic()
        self.kb = "\n\n".join(docs)
        self.history = []
    
    def retrieve_docs(self, query: str, top_k: int = 3):
        terms = query.lower().split()
        lines = self.kb.split('\n')
        scored = [(line, sum(1 for t in terms if t in line.lower())) for line in lines]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [line for line, _ in scored[:top_k]]
    
    def answer(self, query: str):
        docs = self.retrieve_docs(query)
        context = "\n".join(docs)
        self.history.append({"role": "user", "content": query})
        
        resp = self.client.messages.create(
            model="claude-opus-4-5", max_tokens=1024,
            system=f"Answer using this context:\n{context}",
            messages=self.history
        )
        
        ans = resp.content[0].text
        self.history.append({"role": "assistant", "content": ans})
        return ans
