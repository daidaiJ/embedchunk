// Command embedchunk chunks markdown for an EmbeddingGemma 2 endpoint and can
// verify the result by embedding every chunk.
//
//	embedchunk -model model/tokenizer.model -path notes -ub 1024 -embed
//
// It measures every chunk with the model's own tokenizer, so a request can
// never exceed the target endpoint's physical batch size.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/daidaiJ/embedchunk"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		modelPath = flag.String("model", "", "path to the model's tokenizer.model (required)")
		root      = flag.String("path", "", "markdown file or directory to chunk")
		server    = flag.String("server", "http://127.0.0.1:8080", "llama-server base URL")
		ubatch    = flag.Int("ub", 1024, "server physical batch size (-ub); the budget becomes ub-2")
		overlap   = flag.Int("overlap", 0, "overlap in tokens between consecutive chunks")
		minTokens = flag.Int("min", 0, "merge a trailing chunk shorter than this many tokens")
		merge     = flag.Bool("merge-sections", false, "pack across heading boundaries for fewer, fuller chunks")
		embed     = flag.Bool("embed", false, "embed every chunk to verify the token budget")
		batch     = flag.Int("batch", 16, "inputs per embedding request")
		countOnly = flag.Bool("count", false, "read stdin and print its token count")
		verbose   = flag.Bool("v", false, "list per-file results")
	)
	flag.Parse()

	if *modelPath == "" {
		return fmt.Errorf("-model is required (the tokenizer.model shipped next to the GGUF)")
	}
	counter, err := embedchunk.NewSentencePieceCounter(*modelPath)
	if err != nil {
		return err
	}
	budget := embedchunk.BudgetFor(*ubatch)
	if budget <= 0 {
		return fmt.Errorf("-ub %d leaves no room for the server's BOS/EOS", *ubatch)
	}

	if *countOnly {
		data, err := io.ReadAll(os.Stdin)
		if err != nil {
			return err
		}
		tokens, err := counter.Count(string(data))
		if err != nil {
			return err
		}
		fmt.Println(tokens)
		return nil
	}
	if *root == "" {
		return fmt.Errorf("-path is required")
	}

	paths, err := markdownPaths(*root)
	if err != nil {
		return err
	}
	if len(paths) == 0 {
		return fmt.Errorf("no markdown files under %s", *root)
	}

	var (
		allChunks []embedchunk.Chunk
		fileLines []string
		totalDoc  int
		oversized int
	)
	start := time.Now()
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		title := documentTitle(string(data), path)
		chunker, err := embedchunk.NewChunker(counter, embedchunk.Options{
			MaxTokens:     budget,
			Wrap:          embedchunk.DocWrapper(title),
			OverlapTokens: *overlap,
			MinTokens:     *minTokens,
			MergeSections: *merge,
		})
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		docTokens, err := counter.Count(string(data))
		if err != nil {
			return err
		}
		totalDoc += docTokens
		if docTokens > budget {
			oversized++
		}
		chunks, err := chunker.Split(string(data))
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		for i, chunk := range chunks {
			if chunk.Tokens > budget {
				return fmt.Errorf("%s chunk %d: %d tokens exceeds the %d-token budget",
					path, i, chunk.Tokens, budget)
			}
		}
		allChunks = append(allChunks, chunks...)
		if *verbose {
			fileLines = append(fileLines, fmt.Sprintf("%5d doc tokens -> %3d chunks  %s",
				docTokens, len(chunks), filepath.ToSlash(path)))
		}
	}
	chunking := time.Since(start)

	for _, line := range fileLines {
		fmt.Println(line)
	}
	fmt.Printf("\nfiles=%d  document tokens=%d\n", len(paths), totalDoc)
	fmt.Printf("budget=%d tokens (from -ub %d, reserving BOS+EOS)\n", budget, *ubatch)
	fmt.Printf("chunks=%d  (%.1f per file)\n", len(allChunks), float64(len(allChunks))/float64(len(paths)))
	fmt.Printf("files whose text alone exceeds the budget: %d/%d\n", oversized, len(paths))
	report(allChunks)
	fmt.Printf("chunking took %s\n", chunking.Round(time.Millisecond))

	if !*embed {
		fmt.Println("\n(pass -embed to send every chunk to the server and confirm none is rejected)")
		return nil
	}

	client := &embedchunk.Client{
		BaseURL:   *server,
		BatchSize: *batch,
		MaxTokens: budget,
		Counter:   counter,
	}
	texts := make([]string, len(allChunks))
	for i, chunk := range allChunks {
		texts[i] = chunk.Text
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Hour)
	defer cancel()
	start = time.Now()
	vectors, err := client.Embed(ctx, texts)
	elapsed := time.Since(start)
	if err != nil {
		return err
	}
	dims := 0
	if len(vectors) > 0 {
		dims = len(vectors[0])
	}
	fmt.Printf("\nembedded %d chunks -> dim=%d in %s\n", len(vectors), dims, elapsed.Round(time.Millisecond))
	fmt.Printf("server throughput ~%.0f tokens/s at batch=%d\n", float64(totalDoc)/elapsed.Seconds(), *batch)
	fmt.Println("no input was rejected: the token guard holds end to end")
	return nil
}

func report(chunks []embedchunk.Chunk) {
	if len(chunks) == 0 {
		return
	}
	tokens := make([]int, len(chunks))
	total := 0
	for i, chunk := range chunks {
		tokens[i] = chunk.Tokens
		total += chunk.Tokens
	}
	sort.Ints(tokens)
	fmt.Printf("chunk tokens: min=%d median=%d p90=%d max=%d mean=%.0f\n",
		tokens[0], tokens[len(tokens)/2], tokens[(len(tokens)*9)/10], tokens[len(tokens)-1],
		float64(total)/float64(len(tokens)))
}

func markdownPaths(root string) ([]string, error) {
	info, err := os.Stat(root)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return []string{root}, nil
	}
	var paths []string
	err = filepath.Walk(root, func(path string, entry os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() && strings.HasSuffix(strings.ToLower(path), ".md") {
			paths = append(paths, path)
		}
		return nil
	})
	sort.Strings(paths)
	return paths, err
}

// documentTitle prefers the first level-one heading, then any heading, then the
// file name. EmbeddingGemma 2 wants a title in the document prefix and reads
// "title: none" as "this note has no title".
func documentTitle(text, path string) string {
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "# ") {
			return strings.TrimSpace(trimmed[2:])
		}
	}
	return strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
}
