// Package embedchunk prepares text for an OpenAI-compatible embedding endpoint
// (llama.cpp /v1/embeddings serving EmbeddingGemma 2) so that requests never
// exceed the server's physical batch size and never come back as HTTP 500.
//
// Two facts drive the design:
//
//   - llama.cpp rejects an input larger than the physical batch instead of
//     truncating it. With -b/-ub N the accepted input is at most N-2 tokens,
//     because the server prepends BOS and appends EOS.
//   - The token budget applies to the final string that is sent, including the
//     task prefix, so counting must happen after the prefix is applied.
package embedchunk

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	sentencepiece "github.com/eliben/go-sentencepiece"
)

// SpecialTokens is the number of tokens the server adds around an input
// (BOS and EOS for gemma-embedding2).
const SpecialTokens = 2

// BudgetFor returns the largest number of caller-supplied tokens an input may
// have when the server runs with the given -ub (physical batch size).
func BudgetFor(ubatch int) int { return ubatch - SpecialTokens }

// Counter reports how many tokens an embedding request will charge for s,
// excluding the BOS/EOS the server adds itself.
type Counter interface {
	Count(s string) (int, error)
}

// SentencePieceCounter counts tokens locally using the model's own
// tokenizer.model. It reproduces llama.cpp's /tokenize exactly for
// EmbeddingGemma 2 (see the package README for the verification run).
type SentencePieceCounter struct {
	proc *sentencepiece.Processor
}

// NewSentencePieceCounter loads a SentencePiece .model file.
func NewSentencePieceCounter(modelPath string) (*SentencePieceCounter, error) {
	proc, err := sentencepiece.NewProcessorFromPath(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer %s: %w", modelPath, err)
	}
	return &SentencePieceCounter{proc: proc}, nil
}

// Count returns the number of tokens in s.
func (c *SentencePieceCounter) Count(s string) (int, error) {
	return len(c.proc.Encode(s)), nil
}

// VocabSize returns the vocabulary size of the loaded model.
func (c *SentencePieceCounter) VocabSize() int { return c.proc.ModelInfo().VocabularySize }

// Tokenize returns the token ids for s.
func (c *SentencePieceCounter) Tokenize(s string) []int {
	toks := c.proc.Encode(s)
	ids := make([]int, len(toks))
	for i, t := range toks {
		ids[i] = t.ID
	}
	return ids
}

// HTTPCounter counts by calling the server's /tokenize endpoint. It needs no
// tokenizer file and is authoritative by construction, at the cost of one round
// trip per call.
type HTTPCounter struct {
	BaseURL string
	Client  *http.Client
}

// NewHTTPCounter returns a counter bound to a llama-server base URL.
func NewHTTPCounter(baseURL string) *HTTPCounter {
	return &HTTPCounter{BaseURL: baseURL, Client: &http.Client{Timeout: 60 * time.Second}}
}

// Count asks the server how many tokens s becomes.
func (c *HTTPCounter) Count(s string) (int, error) {
	body, err := json.Marshal(map[string]string{"content": s})
	if err != nil {
		return 0, err
	}
	resp, err := c.Client.Post(c.BaseURL+"/tokenize", "application/json", bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("/tokenize returned HTTP %d", resp.StatusCode)
	}
	var out struct {
		Tokens []int `json:"tokens"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return 0, err
	}
	return len(out.Tokens), nil
}
