# embedchunk

Prepare text for an EmbeddingGemma 2 embedding endpoint so that no request is
ever rejected. Chunking happens on token boundaries measured with the model's own
tokenizer, and the token budget covers the task prefix as well as the body.

## The failure this prevents

llama.cpp answers an oversized input with **HTTP 500 and no truncation**:

```
input (1025 tokens) is too large to process. increase the physical batch size
(current batch size: 1024)
```

Three rules follow, all measured on llama.cpp b11540 with
`embeddinggemma-2-Q8_0.gguf`:

| rule | detail |
| --- | --- |
| input limit | `tokens(text) <= min(-c, -ub) - 2`; the server adds BOS and EOS itself |
| hard failure | over the limit is an error, not a silent cut, so an ingest pipeline loses the document |
| prefix counts | the retrieval prefixes consume the same budget as the body |

The prefixes are **not** added by the endpoint. Embedding the same sentence raw
and with `title: none | text: ` gives vectors with cosine similarity 0.84, so
omitting them changes the result rather than being harmless.

`POST /tokenize` adds no special tokens (an empty string returns 0 tokens), which
is how the `-2` is confirmed: a 1022-token input is accepted at `-ub 1024`, a
1023-token input is rejected.

## Measured on a GTX 1650 (4 GB), Windows, CUDA 12.4 build

VRAM is the server's own footprint, over a ~570 MiB desktop baseline, with
`-ngl 99` and the 295.5 MiB Q8_0 model.

| `-ub` | VRAM (`-fa auto`) | VRAM (`-fa on`) | ingest tok/s (`-fa auto`) | ingest tok/s (`-fa on`) |
| --- | --- | --- | --- | --- |
| 512 | ~250–295 MiB | | ~2600 | |
| 1024 | ~351 MiB | ~348 MiB | 2459–2670 | 1875–2052 |
| 2048 | ~479 MiB | | | |
| 4096 | ~912 MiB | ~728 MiB | 2328–2650 | 1948–2084 |
| 8192 | ~2.4 GB | ~1.3 GB | | 1933–2102 |

Other measurements at `-ub 1024`:

- a short query takes 33 ms median; a 417-token chunk takes 179 ms
- `--parallel 1` and `--parallel 4` measure identically
- `-c 512` through `-c 8192` differ by ~10 MiB, i.e. nothing

Conclusions for this machine:

- **Flash Attention is a 20–25 % throughput loss here.** Turing has no native
  support for the f16 path llama.cpp uses, so the fallback costs more than it
  saves. Its only real benefit is capping the attention buffer, which is what
  makes `-ub 8192` cost 1.3 GB instead of 2.4 GB.
- **`-c` costs nothing**, so keep it at the model's real context window (8192) and
  control memory with `-ub` instead. Lowering `-c` to 4096 saves no VRAM, does not
  change latency, and lowers the single-request ceiling to 4094 tokens.
- **Batching beats concurrency.** Sending 8 or 32 inputs per request reaches
  ~2600 tok/s; four concurrent connections reach the same 2670 tok/s, so there is
  nothing to gain from `--parallel` on this GPU.

Recommended server command, with a 1024-token physical batch:

```
llama-server.exe -m model/embeddinggemma-2-Q8_0.gguf --embedding ^
  -ngl 99 -fa auto -c 8192 -b 1024 -ub 1024 --port 8080
```

Raise `-ub` to 4096 (~900 MiB) if single chunks must be larger, and add `-fa on`
when you need `-ub 8192`.

## Counting tokens in Go

`github.com/eliben/go-sentencepiece` loading the `tokenizer.model` that ships next
to the GGUF reproduces llama.cpp's `/tokenize` exactly: 30 hand-picked inputs
(CJK, English, markdown, emoji, tabs, newlines, empty string) returned identical
counts, and the vocabulary matches at 262144 pieces with bos=2, eos=1, unk=3,
pad=0. It runs at ~2.7 M tokens/s, so a document can be measured many times over.

The two other ways to count:

- `POST /tokenize` on the same server: authoritative by construction, one round
  trip per call, and it needs no tokenizer file at all (`HTTPCounter`).
- `github.com/sugarme/tokenizer` or `github.com/daulet/tokenizers` reading
  `tokenizer.json`: plausible but **not verified here**, because this machine's
  proxy blocked the download. Validate any alternative against `/tokenize` with
  `TestRealTokenizer` before trusting it in a pipeline.

## Usage

```
go get github.com/daidaiJ/embedchunk
```

```go
import (
	embedchunk "github.com/daidaiJ/embedchunk"
)

counter, err := embedchunk.NewSentencePieceCounter(`model\tokenizer.model`)
if err != nil {
    return err
}

// -ub 1024 leaves 1022 tokens for the whole input, prefix included.
chunker, err := embedchunk.NewChunker(counter, embedchunk.Options{
    MaxTokens:     embedchunk.BudgetFor(1024),
    Wrap:          embedchunk.DocWrapper("出网设计"),
    OverlapTokens: 64,
    MinTokens:     32,
    MergeSections: true,
})
if err != nil {
    return err
}
chunks, err := chunker.Split(note)
```

`chunk.Text` is exactly what to send. `chunk.Body` is the same text without the
prefix, for display or highlighting. `chunk.Tokens` is the measured cost.

Embedding, with the budget checked locally before anything is sent:

```go
client := &embedchunk.Client{
    BaseURL:   "http://127.0.0.1:8080",
    BatchSize: 16,
    MaxTokens: embedchunk.BudgetFor(1024),
    Counter:   counter,
}
vectors, err := client.Embed(ctx, texts)
```

Command line:

```
embedchunk -model model\tokenizer.model -path path\to\notes -ub 1024 -embed
embedchunk -model model\tokenizer.model -count < note.md      # token count only
echo "task: search result | query: 出网策略" | embedchunk -model model\tokenizer.model -count
```

`-merge-sections` packs across headings for fewer, fuller chunks. On the
167-file wiki it changes the shape of the index substantially:

| mode | chunks | median tokens | mean tokens | budget used |
| --- | --- | --- | --- | --- |
| default (heading-scoped) | 2239 | 239 | 315 | 31 % |
| `-merge-sections` | 791 | 973 | 880 | 86 % |

Heading-scoped chunks keep each chunk inside one topic; merging cuts the number
of embedding requests by ~2.8x. Pick per corpus.

## How it is verified

- `go test ./...` covers the budget invariant, that chunking loses no content,
  that a single long sentence is cut, that overlap really repeats the tail, that
  a `#` inside a fenced code block does not start a section, and that a
  deliberately inconsistent counter cannot push a chunk over budget.
- `EMBEDCHUNK_TOKENIZER=... EMBEDCHUNK_SERVER=... go test -run TestRealTokenizer`
  compares the local counter with the live server.
- `EMBEDCHUNK_WIKI=... go test -run TestSplitRealWikiFile` chunks a real corpus
  and asserts the budget holds on every chunk.
- `-embed` pushes every chunk through `/v1/embeddings`, so a full ingest is the
  test: any over-budget chunk would come back as HTTP 500.
