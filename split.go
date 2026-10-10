package embedchunk

import "strings"

// splitSections cuts markdown at heading boundaries, keeping each heading with
// the text that follows it. Lines inside fenced code blocks never start a section.
func splitSections(text string) []string {
	lines := strings.Split(text, "\n")
	var out []string
	var cur []string
	fence := ""

	flush := func() {
		if len(cur) > 0 {
			out = append(out, strings.Join(cur, "\n"))
			cur = nil
		}
	}
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if isFence(trimmed) {
			if fence == "" {
				fence = fenceMarker(trimmed)
			} else if strings.HasPrefix(trimmed, fence) {
				fence = ""
			}
			cur = append(cur, line)
			continue
		}
		if fence == "" && isHeading(trimmed) && len(cur) > 0 {
			flush()
		}
		cur = append(cur, line)
	}
	flush()
	return out
}

func isFence(trimmed string) bool {
	return strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~")
}

func fenceMarker(trimmed string) string {
	if strings.HasPrefix(trimmed, "```") {
		return "```"
	}
	return "~~~"
}

func isHeading(trimmed string) bool {
	if !strings.HasPrefix(trimmed, "#") {
		return false
	}
	i := 0
	for i < len(trimmed) && trimmed[i] == '#' {
		i++
	}
	if i > 6 {
		return false
	}
	return i == len(trimmed) || trimmed[i] == ' ' || trimmed[i] == '\t'
}

// sentenceEnders split a section into sentence-sized pieces. CJK punctuation ends
// a sentence on its own; an ASCII period or comma only ends one when whitespace
// follows, so decimals, version numbers and dotted identifiers stay intact.
const sentenceEnders = "。！？；"
const asciiEnders = ".!?;"

func splitSentences(section string) []string {
	runes := []rune(section)
	var out []string
	var b strings.Builder

	for i := 0; i < len(runes); i++ {
		r := runes[i]
		b.WriteRune(r)
		if r == '\n' {
			out = appendSentence(out, b.String())
			b.Reset()
			continue
		}
		endsHere := strings.ContainsRune(sentenceEnders, r)
		if !endsHere && strings.ContainsRune(asciiEnders, r) {
			endsHere = i+1 >= len(runes) || isSpaceRune(runes[i+1])
		}
		if !endsHere {
			continue
		}
		// Absorb closing punctuation that belongs to this sentence.
		for i+1 < len(runes) && strings.ContainsRune("”’」』）)]》〉", runes[i+1]) {
			i++
			b.WriteRune(runes[i])
		}
		out = appendSentence(out, b.String())
		b.Reset()
	}
	if b.Len() > 0 {
		out = appendSentence(out, b.String())
	}
	return out
}

func isSpaceRune(r rune) bool { return r == ' ' || r == '\t' || r == '\n' || r == '\r' }

func appendSentence(out []string, s string) []string {
	if strings.TrimSpace(s) == "" {
		return out
	}
	return append(out, s)
}
