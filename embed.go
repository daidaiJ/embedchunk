package embedchunk

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client talks to a llama.cpp /v1/embeddings endpoint.
type Client struct {
	BaseURL string
	HTTP    *http.Client
	// BatchSize is the number of inputs sent per request. Batching raises
	// throughput without changing results; 16 is a good default.
	BatchSize int
	// MaxTokens, when positive, rejects an oversized input locally instead of
	// letting the server answer HTTP 500. Set it to BudgetFor(ubatch) minus the
	// input's own prefix cost, or leave it to Counter.
	MaxTokens int
	// Counter, when set, is used to enforce MaxTokens.
	Counter Counter
}

type embeddingResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// Embed returns one vector per input, in the same order. A batch that the server
// rejects with HTTP 500 is retried once split in half, and a single failed input
// is reported with its token count so the caller can see why it was too large.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if c.BaseURL == "" {
		return nil, fmt.Errorf("BaseURL is empty")
	}
	if c.HTTP == nil {
		c.HTTP = &http.Client{Timeout: 10 * time.Minute}
	}
	batch := c.BatchSize
	if batch <= 0 {
		batch = 1
	}
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += batch {
		end := start + batch
		if end > len(texts) {
			end = len(texts)
		}
		vectors, err := c.embedBatch(ctx, texts[start:end])
		if err != nil {
			return nil, fmt.Errorf("inputs [%d:%d]: %w", start, end, err)
		}
		out = append(out, vectors...)
	}
	return out, nil
}

func (c *Client) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if c.MaxTokens > 0 && c.Counter != nil {
		for i, text := range texts {
			tokens, err := c.Counter.Count(text)
			if err != nil {
				return nil, err
			}
			if tokens > c.MaxTokens {
				return nil, fmt.Errorf("input %d is %d tokens, over the %d-token budget: %s",
					i, tokens, c.MaxTokens, truncate(text, 40))
			}
		}
	}
	vectors, err := c.post(ctx, texts)
	if err == nil {
		return vectors, nil
	}
	if len(texts) == 1 {
		return nil, err
	}
	// The server rejects an oversized input with 500 and no retry hint, so fall
	// back to halves to isolate the offending input.
	mid := len(texts) / 2
	left, leftErr := c.embedBatch(ctx, texts[:mid])
	if leftErr != nil {
		return nil, leftErr
	}
	right, rightErr := c.embedBatch(ctx, texts[mid:])
	if rightErr != nil {
		return nil, rightErr
	}
	return append(left, right...), nil
}

func (c *Client) post(ctx context.Context, texts []string) ([][]float32, error) {
	body, err := json.Marshal(map[string]any{"input": texts})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.BaseURL+"/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 300))
		return nil, fmt.Errorf("/v1/embeddings returned HTTP %d: %s",
			resp.StatusCode, bytes.TrimSpace(detail))
	}
	var parsed embeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, err
	}
	if len(parsed.Data) != len(texts) {
		return nil, fmt.Errorf("asked for %d embeddings, got %d", len(texts), len(parsed.Data))
	}
	ordered := make([][]float32, len(parsed.Data))
	for _, item := range parsed.Data {
		if item.Index < 0 || item.Index >= len(ordered) {
			return nil, fmt.Errorf("response index %d out of range", item.Index)
		}
		ordered[item.Index] = item.Embedding
	}
	return ordered, nil
}
