package embedchunk

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const testTitle = "测试文档"

func testWrap(body string) string { return Doc(testTitle, body) }

// runeCounter counts runes, which is close enough to a real tokenizer to exercise
// the chunker's arithmetic without loading a model.
type runeCounter struct{}

func (runeCounter) Count(s string) (int, error) { return len([]rune(s)), nil }

// cliffCounter counts runes below a threshold and reports a wildly larger count
// above it, so that a concatenation looks cheaper than its parts. It exercises the
// path where packing overshoots and the body has to be cut at the rune level.
type cliffCounter struct{ cliff int }

func (c cliffCounter) Count(s string) (int, error) {
	n := len([]rune(s))
	if n > c.cliff {
		return n * 10, nil
	}
	return n, nil
}

func stripSpace(s string) string {
	return strings.Map(func(r rune) rune {
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			return -1
		}
		return r
	}, s)
}

const sampleDoc = `# 出网设计

OpenSandbox 的 egress 通过统一网关出网，网关侧做鉴权和审计。这条策略在 pod 启动时注入，
并且在清理阶段回收。注意 ` + "`higress`" + ` 的配置是热更新的。

## 代码示例

` + "```go" + `
// # 这一行是注释而不是标题
func main() {
	println("hello")
}
` + "```" + `

## 长段落

这一段没有任何标点所以不会被切开它会被当作一个很长的句子来处理直到触发按字符切分为止
这一段的目的是验证超长句子的兜底路径能否保证不超过预算并且不丢内容。

收尾一句。`

func TestBudgetRespected(t *testing.T) {
	for _, maxTokens := range []int{60, 120, 400} {
		ch, err := NewChunker(runeCounter{}, Options{MaxTokens: maxTokens, Wrap: testWrap, OverlapTokens: 8})
		if err != nil {
			t.Fatalf("maxTokens=%d: %v", maxTokens, err)
		}
		chunks, err := ch.Split(sampleDoc)
		if err != nil {
			t.Fatalf("maxTokens=%d: %v", maxTokens, err)
		}
		if len(chunks) == 0 {
			t.Fatalf("maxTokens=%d: no chunks", maxTokens)
		}
		for i, c := range chunks {
			if c.Tokens > maxTokens {
				t.Errorf("maxTokens=%d chunk %d: %d tokens over budget\n%s", maxTokens, i, c.Tokens, c.Text)
			}
			if c.Tokens != len([]rune(c.Text)) {
				t.Errorf("maxTokens=%d chunk %d: Tokens=%d does not match the counter", maxTokens, i, c.Tokens)
			}
			if !strings.HasPrefix(c.Text, "title: "+testTitle+" | text: ") {
				t.Errorf("maxTokens=%d chunk %d: prefix missing\n%s", maxTokens, i, c.Text)
			}
			if strings.TrimSpace(c.Body) == "" {
				t.Errorf("maxTokens=%d chunk %d: empty body", maxTokens, i)
			}
		}
	}
}

func TestNoContentLoss(t *testing.T) {
	ch, err := NewChunker(runeCounter{}, Options{MaxTokens: 100, Wrap: testWrap})
	if err != nil {
		t.Fatal(err)
	}
	chunks, err := ch.Split(sampleDoc)
	if err != nil {
		t.Fatal(err)
	}
	var joined strings.Builder
	for _, c := range chunks {
		joined.WriteString(c.Body)
	}
	if got, want := stripSpace(joined.String()), stripSpace(sampleDoc); got != want {
		t.Errorf("content changed by chunking:\ngot  %q\nwant %q", got, want)
	}
}

func TestLongSentenceIsSplit(t *testing.T) {
	sentence := strings.Repeat("没有标点的长句", 60) // 420 runes, no terminator
	ch, err := NewChunker(runeCounter{}, Options{MaxTokens: 80, Wrap: testWrap})
	if err != nil {
		t.Fatal(err)
	}
	chunks, err := ch.Split(sentence)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) < 5 {
		t.Fatalf("expected the sentence to be cut into several chunks, got %d", len(chunks))
	}
	var joined strings.Builder
	for i, c := range chunks {
		if c.Tokens > 80 {
			t.Fatalf("chunk %d over budget: %d", i, c.Tokens)
		}
		joined.WriteString(c.Body)
	}
	if got := stripSpace(joined.String()); got != stripSpace(sentence) {
		t.Errorf("long sentence lost content:\ngot  %q\nwant %q", got, stripSpace(sentence))
	}
}

func TestOverlapRepeatsTail(t *testing.T) {
	ch, err := NewChunker(runeCounter{}, Options{MaxTokens: 90, Wrap: testWrap, OverlapTokens: 30})
	if err != nil {
		t.Fatal(err)
	}
	chunks, err := ch.Split(sampleDoc)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) < 2 {
		t.Fatalf("need several chunks to observe overlap, got %d", len(chunks))
	}
	overlapped := 0
	for i := 1; i < len(chunks); i++ {
		prev, cur := chunks[i-1].Body, chunks[i].Body
		// The longest k where the previous body ends with exactly the first k
		// runes of the current body: the repetition overlap is supposed to create.
		if border(prev, cur) >= 8 {
			overlapped++
		}
	}
	if overlapped == 0 {
		t.Errorf("no chunk repeated the tail of its predecessor:\n%s", dump(chunks))
	}
}

// border returns the length in runes of the longest suffix of a that is also a
// prefix of b.
func border(a, b string) int {
	ar, br := []rune(a), []rune(b)
	limit := len(ar)
	if len(br) < limit {
		limit = len(br)
	}
	for k := limit; k > 0; k-- {
		if string(ar[len(ar)-k:]) == string(br[:k]) {
			return k
		}
	}
	return 0
}

func TestCodeFenceHeadingDoesNotSplit(t *testing.T) {
	// The "#" inside the Go comment must not start a new section, so the whole
	// fenced block stays in one chunk when the budget allows it.
	ch, err := NewChunker(runeCounter{}, Options{MaxTokens: 500, Wrap: testWrap})
	if err != nil {
		t.Fatal(err)
	}
	chunks, err := ch.Split(sampleDoc)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, c := range chunks {
		if strings.Contains(c.Body, "```go") && strings.Contains(c.Body, "println(\"hello\")") &&
			strings.Contains(c.Body, "```") {
			found = true
		}
	}
	if !found {
		t.Errorf("the fenced code block was split across chunks:\n%s", dump(chunks))
	}
}

func TestOptionValidation(t *testing.T) {
	if _, err := NewChunker(nil, Options{MaxTokens: 10}); err == nil {
		t.Error("a nil counter must be rejected")
	}
	if _, err := NewChunker(runeCounter{}, Options{MaxTokens: 0}); err == nil {
		t.Error("a zero budget must be rejected")
	}
	// The prefix alone is 24 runes, so a 20-token budget cannot hold it.
	if _, err := NewChunker(runeCounter{}, Options{MaxTokens: 20, Wrap: testWrap}); err == nil {
		t.Error("a budget smaller than the prefix must be rejected")
	}
	if _, err := NewChunker(runeCounter{}, Options{MaxTokens: 50, Wrap: testWrap, OverlapTokens: 50}); err == nil {
		t.Error("an overlap as large as the budget must be rejected")
	}
}

func TestAdversarialCounterKeepsBudget(t *testing.T) {
	ch, err := NewChunker(cliffCounter{cliff: 40}, Options{MaxTokens: 100, Wrap: testWrap})
	if err != nil {
		t.Fatal(err)
	}
	chunks, err := ch.Split(sampleDoc)
	if err != nil {
		t.Fatal(err)
	}
	for i, c := range chunks {
		if c.Tokens > 100 {
			t.Errorf("chunk %d is %d tokens, over the 100-token budget even though the counter is pathological", i, c.Tokens)
		}
	}
}

func TestMergeSectionsFillsBudget(t *testing.T) {
	var doc strings.Builder
	for i := 0; i < 12; i++ {
		doc.WriteString("\n## 小节 ")
		doc.WriteString(string(rune('A' + i)))
		doc.WriteString("\n\n")
		doc.WriteString(strings.Repeat("一小段内容。", 8)) // ~48 runes per section
	}
	const budget = 200
	measure := func(merge bool) []Chunk {
		ch, err := NewChunker(runeCounter{}, Options{
			MaxTokens: budget, Wrap: testWrap, MergeSections: merge,
		})
		if err != nil {
			t.Fatal(err)
		}
		chunks, err := ch.Split(doc.String())
		if err != nil {
			t.Fatal(err)
		}
		return chunks
	}
	split, merged := measure(false), measure(true)
	if len(merged) >= len(split) {
		t.Errorf("merging sections should produce fewer chunks: %d vs %d", len(merged), len(split))
	}
	for i, c := range merged {
		if c.Tokens > budget {
			t.Errorf("merged chunk %d is %d tokens, over budget", i, c.Tokens)
		}
	}
	if mean(merged) <= mean(split) {
		t.Errorf("merged chunks should be larger on average: %.0f vs %.0f", mean(merged), mean(split))
	}
	var joined strings.Builder
	for _, c := range merged {
		joined.WriteString(c.Body)
	}
	if stripSpace(joined.String()) != stripSpace(doc.String()) {
		t.Error("merging sections lost or reordered content")
	}
}

func mean(chunks []Chunk) float64 {
	if len(chunks) == 0 {
		return 0
	}
	total := 0
	for _, c := range chunks {
		total += c.Tokens
	}
	return float64(total) / float64(len(chunks))
}

func TestMinTokensMergesShortTail(t *testing.T) {
	doc := "第一段内容足够长可以构成一个块。" + strings.Repeat("填充内容", 10) + "\n尾。"
	ch, err := NewChunker(runeCounter{}, Options{MaxTokens: 200, Wrap: testWrap, MinTokens: 10})
	if err != nil {
		t.Fatal(err)
	}
	chunks, err := ch.Split(doc)
	if err != nil {
		t.Fatal(err)
	}
	for i, c := range chunks {
		if c.Tokens < 10 {
			t.Errorf("chunk %d is %d tokens, below MinTokens, so it should have been merged", i, c.Tokens)
		}
		if c.Tokens > 200 {
			t.Errorf("chunk %d exceeds the budget after merging: %d", i, c.Tokens)
		}
	}
}

// TestRealTokenizer compares the local SentencePiece counter with llama.cpp's
// /tokenize. It runs only when the environment provides the paths:
//
//	EMBEDCHUNK_TOKENIZER=D:\...\model\tokenizer.model
//	EMBEDCHUNK_SERVER=http://127.0.0.1:8170        (optional)
func TestRealTokenizer(t *testing.T) {
	modelPath := os.Getenv("EMBEDCHUNK_TOKENIZER")
	if modelPath == "" {
		t.Skip("set EMBEDCHUNK_TOKENIZER to the tokenizer.model path")
	}
	counter, err := NewSentencePieceCounter(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	if got := counter.VocabSize(); got != 262144 {
		t.Errorf("vocab size = %d, want 262144", got)
	}
	server := os.Getenv("EMBEDCHUNK_SERVER")
	httpCounter := (*HTTPCounter)(nil)
	if server != "" {
		httpCounter = NewHTTPCounter(server)
	}
	inputs := []string{
		"", " ", "的", "你好，世界。", "task: search result | query: 沙箱出网策略怎么配",
		Doc("测试", strings.Repeat("很长的一段中文文本，", 40)), "func main() {}",
	}
	for _, in := range inputs {
		local, err := counter.Count(in)
		if err != nil {
			t.Fatalf("%q: %v", in, err)
		}
		if httpCounter == nil {
			continue
		}
		remote, err := httpCounter.Count(in)
		if err != nil {
			t.Fatalf("%q: %v", in, err)
		}
		if local != remote {
			t.Errorf("count mismatch for %q: local=%d server=%d", truncate(in, 30), local, remote)
		}
	}
}

// TestSplitRealWikiFile chunks real notes from the wiki when EMBEDCHUNK_WIKI
// points at a directory of markdown files.
func TestSplitRealWikiFile(t *testing.T) {
	wiki := os.Getenv("EMBEDCHUNK_WIKI")
	if wiki == "" {
		t.Skip("set EMBEDCHUNK_WIKI to a directory of markdown files")
	}
	counter := Counter(runeCounter{})
	if modelPath := os.Getenv("EMBEDCHUNK_TOKENIZER"); modelPath != "" {
		real, err := NewSentencePieceCounter(modelPath)
		if err != nil {
			t.Fatal(err)
		}
		counter = real
	}
	const budget = 1000 // what -ub 1024 leaves for the prefixed input
	files := 0
	for _, path := range mustGlob(t, wiki) {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		title := filepath.Base(strings.TrimSuffix(path, ".md"))
		ch, err := NewChunker(counter, Options{
			MaxTokens:     budget,
			Wrap:          DocWrapper(title),
			OverlapTokens: 0,
			MinTokens:     0,
		})
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		chunks, err := ch.Split(string(data))
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		for i, c := range chunks {
			if c.Tokens > budget {
				t.Fatalf("%s chunk %d: %d tokens over budget", path, i, c.Tokens)
			}
		}
		files++
	}
	t.Logf("chunked %d files with budget %d without exceeding it", files, budget)
}

func mustGlob(t *testing.T, root string) []string {
	t.Helper()
	var paths []string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(path, ".md") {
			paths = append(paths, path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return paths
}

func dump(chunks []Chunk) string {
	var b strings.Builder
	for i, c := range chunks {
		b.WriteString("--- chunk ")
		b.WriteString(string(rune('0' + i%10)))
		b.WriteString(" ---\n")
		b.WriteString(c.Body)
		b.WriteString("\n")
	}
	return b.String()
}
