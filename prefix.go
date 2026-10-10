package embedchunk

// Task prefixes from the EmbeddingGemma 2 model card. They are applied by the
// caller: llama.cpp and Ollama embedding endpoints add no prefix of their own.
type Task string

const (
	// SearchResult is the asymmetric pair for document search: use Task.Query for
	// queries and Doc for corpus items.
	SearchResult Task = "search result"
	// QuestionAnswering is the asymmetric pair for question answering.
	QuestionAnswering Task = "question answering"
	// FactChecking is the asymmetric pair for fact checking.
	FactChecking Task = "fact checking"
	// CodeRetrieval is the asymmetric pair for code search.
	CodeRetrieval Task = "code retrieval"
	// Classification is symmetric: apply the same prefix to everything compared.
	Classification Task = "classification"
	// Clustering is symmetric.
	Clustering Task = "clustering"
	// SentenceSimilarity is symmetric.
	SentenceSimilarity Task = "sentence similarity"
)

// Query renders the query side of a task's prefix.
func (t Task) Query(q string) string {
	return "task: " + string(t) + " | query: " + q
}

// Doc renders the document side of a task's prefix.
func Doc(title, body string) string {
	if title == "" {
		title = "none"
	}
	return "title: " + title + " | text: " + body
}

// DocWrapper returns a Wrap function for Chunker that prefixes every chunk with
// the document-side prefix for the given title.
func DocWrapper(title string) func(body string) string {
	return func(body string) string { return Doc(title, body) }
}

// QueryWrapper is the symmetric-task counterpart of DocWrapper for the query
// side of a retrieval or question-answering task.
func QueryWrapper(t Task) func(body string) string {
	return func(body string) string { return t.Query(body) }
}
