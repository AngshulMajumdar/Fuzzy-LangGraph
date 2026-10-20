from fuzzy_langgraph import DocumentRecord, FuzzyIndex
from fuzzy_langgraph.graph import FuzzyRAGGraph

docs = [
    DocumentRecord("old", "The current portal is OldDesk.", source="cache", metadata={"authority":0.4,"freshness":0.1,"reliability":0.5}),
    DocumentRecord("new", "The current portal is PeopleHub.", source="handbook", metadata={"authority":0.98,"freshness":1.0,"reliability":0.95}),
]
graph = FuzzyRAGGraph(FuzzyIndex(docs))
print(graph.invoke("What is the current portal?"))
