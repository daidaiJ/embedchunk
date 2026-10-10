package embedchunk

import (
	"errors"
	"fmt"
	"strings"
)

// Chunk is one unit of text ready to be embedded.
type Chunk struct {
	Body   string // the chunk text without any task prefix
	Text   string // exactly what is sent: Wrap(Body)
	Tokens int    // Counter.Count(Text), excluding the server's BOS/EOS
}

// Options controls how text is split.
type Options struct {
	// MaxTokens is the hard budget for Text, that is, for the wrapped chunk. It
	// must not exceed BudgetFor(ubatch).
	MaxTokens int
	// Wrap applies the task prefix. It is called with a chunk body and must be
	// cheap and deterministic. Nil means no prefix.
	Wrap func(body string) string
	// OverlapTokens repeats the tail of the previous chunk at the start of the
	// next one, so a passage split across a boundary stays reachable.
	OverlapTokens int
	// MinTokens merges a trailing chunk shorter than this into its predecessor,
	// as long as the merged chunk still fits MaxTokens.
	MinTokens int
	// MergeSections packs across heading boundaries instead of restarting at every
	// heading. It fills the budget better and so sends fewer, larger chunks, at
	// the cost of chunks that mix topics.
	MergeSections bool
}

// Chunker splits a document into chunks that each fit the token budget.
type Chunker struct {
	counter     Counter
	opt         Options
	pieceTokens map[string]int
}

// NewChunker validates the options against the counter's cost for the prefix.
func NewChunker(counter Counter, opt Options) (*Chunker, error) {
	if counter == nil {
		return nil, errors.New("counter is nil")
	}
	if opt.MaxTokens <= 0 {
		return nil, errors.New("MaxTokens must be positive")
	}
	if opt.Wrap == nil {
		opt.Wrap = func(s string) string { return s }
	}
	if opt.OverlapTokens < 0 {
		return nil, errors.New("OverlapTokens must not be negative")
	}
	if opt.OverlapTokens >= opt.MaxTokens {
		return nil, errors.New("OverlapTokens must be smaller than MaxTokens")
	}
	prefixTokens, err := counter.Count(opt.Wrap(""))
	if err != nil {
		return nil, err
	}
	if prefixTokens >= opt.MaxTokens {
		return nil, fmt.Errorf("budget %d leaves no room for the prefix (%d tokens)",
			opt.MaxTokens, prefixTokens)
	}
	return &Chunker{
		counter:     counter,
		opt:         opt,
		pieceTokens: make(map[string]int),
	}, nil
}

// PrefixTokens reports what the task prefix costs on its own.
func (c *Chunker) PrefixTokens() (int, error) { return c.count("") }

// Split returns the chunks of text in order. Every returned chunk satisfies
// Count(Text) <= MaxTokens: each chunk is measured with the real counter before
// it is appended to the result.
func (c *Chunker) Split(text string) ([]Chunk, error) {
	var out []Chunk
	if c.opt.MergeSections {
		var all []string
		for _, section := range splitSections(text) {
			pieces, err := c.prepare(section)
			if err != nil {
				return nil, err
			}
			all = append(all, pieces...)
		}
		if len(all) > 0 {
			if err := c.pack(all, &out); err != nil {
				return nil, err
			}
		}
	} else {
		for _, section := range splitSections(text) {
			pieces, err := c.prepare(section)
			if err != nil {
				return nil, err
			}
			if len(pieces) == 0 {
				continue
			}
			if err := c.pack(pieces, &out); err != nil {
				return nil, err
			}
		}
	}
	if c.opt.MinTokens > 0 {
		var err error
		out, err = c.mergeShortTail(out)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// prepare turns a section into pieces that each fit the budget on their own.
func (c *Chunker) prepare(section string) ([]string, error) {
	var pieces []string
	for _, sentence := range splitSentences(section) {
		if strings.TrimSpace(sentence) == "" {
			continue
		}
		split, err := c.hardSplit(sentence)
		if err != nil {
			return nil, err
		}
		pieces = append(pieces, split...)
	}
	return pieces, nil
}

// hardSplit breaks s into the fewest pieces such that every piece fits the budget
// once wrapped. It fails when not even one rune fits.
func (c *Chunker) hardSplit(s string) ([]string, error) {
	remaining := s
	var out []string
	for {
		fits, n, err := c.largestFittingPrefix(remaining)
		if err != nil {
			return nil, err
		}
		if n == 0 {
			return nil, fmt.Errorf("budget %d is too small for the wrap of %q",
				c.opt.MaxTokens, truncate(strings.TrimSpace(remaining), 20))
		}
		if fits {
			return append(out, remaining), nil
		}
		out = append(out, remaining[:n])
		remaining = remaining[n:]
	}
}

// largestFittingPrefix reports whether the whole string fits. If it does not, it
// returns the byte length of the longest rune-aligned prefix that does.
func (c *Chunker) largestFittingPrefix(s string) (fits bool, n int, err error) {
	tokens, err := c.count(s)
	if err != nil {
		return false, 0, err
	}
	if tokens <= c.opt.MaxTokens {
		return true, len(s), nil
	}
	runes := []rune(s)
	lo, hi := 0, len(runes)-1 // the full rune slice is known not to fit
	for lo < hi {
		mid := (lo + hi + 1) / 2
		count, err := c.count(string(runes[:mid]))
		if err != nil {
			return false, 0, err
		}
		if count <= c.opt.MaxTokens {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return false, len(string(runes[:lo])), nil
}

// pack fills chunks greedily, closing a chunk only when the next piece no longer
// fits. Chunks reach out exclusively through emit, which measures with the real
// counter, so the budget holds even where the concatenation of sentences merges
// into fewer or more tokens than its parts.
func (c *Chunker) pack(pieces []string, out *[]Chunk) error {
	var cur []string
	var overlap string

	// emit appends exactly one verified chunk and leaves anything it could not
	// use in cur.
	emit := func() error {
		if len(cur) == 0 {
			return nil
		}
		chunk, err := c.makeChunk(strings.Join(cur, ""))
		if err != nil {
			return err
		}
		if chunk.Tokens <= c.opt.MaxTokens {
			*out = append(*out, chunk)
			overlap, err = c.overlapTail(chunk.Body)
			cur = nil
			return err
		}
		// Sentence concatenation produced a body over budget: cut it at the rune
		// level. Every part fits alone, so the first part is a valid chunk.
		parts, err := c.hardSplit(chunk.Body)
		if err != nil {
			return err
		}
		first, err := c.makeChunk(parts[0])
		if err != nil {
			return err
		}
		*out = append(*out, first)
		overlap = ""
		cur = parts[1:]
		return nil
	}

	for _, piece := range pieces {
		pieceTokens, err := c.countPiece(piece)
		if err != nil {
			return err
		}
		for len(cur) > 0 {
			current, err := c.currentTokens(cur)
			if err != nil {
				return err
			}
			if current+pieceTokens <= c.opt.MaxTokens {
				break
			}
			if err := emit(); err != nil {
				return err
			}
			after, err := c.currentTokens(cur)
			if err != nil {
				return err
			}
			if len(cur) > 0 && after >= current {
				return fmt.Errorf("chunker made no progress: %d -> %d tokens", current, after)
			}
		}
		if len(cur) == 0 {
			cur = []string{piece}
			total, err := c.currentTokens(cur)
			if err != nil {
				return err
			}
			if overlap != "" && total+len([]rune(overlap)) <= c.opt.MaxTokens {
				// Measure the pair before trusting the rune-count shortcut.
				candidate := []string{overlap, piece}
				measured, err := c.currentTokens(candidate)
				if err != nil {
					return err
				}
				if measured <= c.opt.MaxTokens {
					cur = candidate
				}
			}
			continue
		}
		cur = append(cur, piece)
	}
	for len(cur) > 0 {
		if err := emit(); err != nil {
			return err
		}
	}
	return nil
}

// currentTokens measures the chunk being accumulated. It is called once per
// piece, so packing a document costs one tokenisation per sentence on top of the
// per-chunk verification.
func (c *Chunker) currentTokens(pieces []string) (int, error) {
	if len(pieces) == 0 {
		return 0, nil
	}
	return c.count(strings.Join(pieces, ""))
}

// overlapTail returns the longest suffix of body that stays within OverlapTokens,
// moved forward to a sentence boundary when one falls inside it. Overlapping by
// whole sentences alone would do nothing whenever a single sentence costs more
// than OverlapTokens, which is the common case.
func (c *Chunker) overlapTail(body string) (string, error) {
	if c.opt.OverlapTokens <= 0 {
		return "", nil
	}
	runes := []rune(body)
	if len(runes) == 0 {
		return "", nil
	}
	prefixCost, err := c.count("")
	if err != nil {
		return "", err
	}
	suffixCost := func(n int) (int, error) {
		if n <= 0 {
			return 0, nil
		}
		total, err := c.count(string(runes[len(runes)-n:]))
		if err != nil {
			return 0, err
		}
		return total - prefixCost, nil
	}
	lo, hi, best := 1, len(runes), 0
	for lo <= hi {
		mid := (lo + hi) / 2
		cost, err := suffixCost(mid)
		if err != nil {
			return "", err
		}
		if cost <= c.opt.OverlapTokens {
			best = mid
			lo = mid + 1
		} else {
			hi = mid - 1
		}
	}
	if best == 0 {
		return "", nil
	}
	start := len(runes) - best
	for i := start; i < len(runes); i++ {
		if strings.ContainsRune(sentenceEnders+"\n", runes[i]) {
			start = i + 1
			break
		}
	}
	if start >= len(runes) {
		return "", nil
	}
	return string(runes[start:]), nil
}

func (c *Chunker) count(body string) (int, error) { return c.counter.Count(c.opt.Wrap(body)) }

func (c *Chunker) countPiece(piece string) (int, error) {
	if tokens, ok := c.pieceTokens[piece]; ok {
		return tokens, nil
	}
	tokens, err := c.count(piece)
	if err != nil {
		return 0, err
	}
	c.pieceTokens[piece] = tokens
	return tokens, nil
}

func (c *Chunker) makeChunk(body string) (Chunk, error) {
	text := c.opt.Wrap(body)
	tokens, err := c.counter.Count(text)
	if err != nil {
		return Chunk{}, err
	}
	return Chunk{Body: body, Text: text, Tokens: tokens}, nil
}

// mergeShortTail folds a too-short trailing chunk into its predecessor when the
// merged chunk still fits the budget.
func (c *Chunker) mergeShortTail(chunks []Chunk) ([]Chunk, error) {
	for i := len(chunks) - 1; i > 0; i-- {
		if chunks[i].Tokens >= c.opt.MinTokens {
			continue
		}
		merged, err := c.makeChunk(chunks[i-1].Body + chunks[i].Body)
		if err != nil {
			return nil, err
		}
		if merged.Tokens > c.opt.MaxTokens {
			continue
		}
		chunks[i-1] = merged
		chunks = append(chunks[:i], chunks[i+1:]...)
	}
	return chunks, nil
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "..."
}
