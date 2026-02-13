package ollama

import (
	"io"
	"strings"
	"testing"
)

// nopCloser wraps an io.Reader as io.ReadCloser.
type nopCloser struct{ io.Reader }

func (nopCloser) Close() error { return nil }

func TestSplitScanner_NewlineDelimited(t *testing.T) {
	input := "line1\nline2\nline3\n"
	sc := NewSplitScanner(nopCloser{strings.NewReader(input)}, "\n")

	var lines []string
	for sc.Scan() {
		lines = append(lines, sc.Text())
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scanner error: %v", err)
	}

	want := []string{"line1", "line2", "line3"}
	if len(lines) != len(want) {
		t.Fatalf("got %d lines, want %d: %v", len(lines), len(want), lines)
	}
	for i, l := range lines {
		if l != want[i] {
			t.Errorf("line[%d] = %q, want %q", i, l, want[i])
		}
	}
}

func TestSplitScanner_CustomDelimiter(t *testing.T) {
	input := "a||b||c"
	sc := NewSplitScanner(nopCloser{strings.NewReader(input)}, "||")

	var parts []string
	for sc.Scan() {
		parts = append(parts, sc.Text())
	}

	want := []string{"a", "b", "c"}
	if len(parts) != len(want) {
		t.Fatalf("got %d parts, want %d: %v", len(parts), len(want), parts)
	}
	for i, p := range parts {
		if p != want[i] {
			t.Errorf("part[%d] = %q, want %q", i, p, want[i])
		}
	}
}

func TestSplitScanner_EmptyInput(t *testing.T) {
	sc := NewSplitScanner(nopCloser{strings.NewReader("")}, "\n")

	var count int
	for sc.Scan() {
		count++
	}
	if count != 0 {
		t.Errorf("expected 0 tokens from empty input, got %d", count)
	}
}

func TestSplitScanner_JSONLines(t *testing.T) {
	// Simulate actual Ollama NDJSON stream
	input := `{"response":"hello"}` + "\n" + `{"response":"world","done":true}` + "\n"
	sc := NewSplitScanner(nopCloser{strings.NewReader(input)}, "\n")

	var jsons []string
	for sc.Scan() {
		text := sc.Text()
		if text != "" {
			jsons = append(jsons, text)
		}
	}

	if len(jsons) != 2 {
		t.Fatalf("got %d JSON lines, want 2", len(jsons))
	}
}
