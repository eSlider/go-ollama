package ollama

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDSN_resolveAPI(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		dsn  DSN
		want APIBackend
	}{
		{"explicit openai", DSN{API: APIOpenAI}, APIOpenAI},
		{"explicit ollama", DSN{API: APIOllama, URL: "http://x/v1/chat/completions"}, APIOllama},
		{"detect v1 path", DSN{URL: "http://127.0.0.1:18434/v1"}, APIOpenAI},
		{"detect chat", DSN{URL: "http://host/v1/chat/completions"}, APIOpenAI},
		{"detect completions", DSN{URL: "http://127.0.0.1:8102/v1/completions"}, APICompletions},
		{"default ollama", DSN{URL: DefaultGenerateURL}, APIOllama},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := tc.dsn.resolveAPI(); got != tc.want {
				t.Fatalf("resolveAPI = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestChatCompletionsURL(t *testing.T) {
	t.Parallel()
	cases := []struct {
		in, want string
	}{
		{"", DefaultOpenAIChatURL},
		{"http://h:18434/v1", "http://h:18434/v1/chat/completions"},
		{"http://h:18434/v1/", "http://h:18434/v1/chat/completions"},
		{"http://h/v1/chat/completions", "http://h/v1/chat/completions"},
		{"http://h:18434", "http://h:18434/v1/chat/completions"},
	}
	for _, tc := range cases {
		if got := chatCompletionsURL(tc.in); got != tc.want {
			t.Fatalf("chatCompletionsURL(%q)=%q want %q", tc.in, got, tc.want)
		}
	}
}

func TestQueryOpenAI_SSE_OnCodeBlockYAML(t *testing.T) {
	// Use-case: llama-server SSE stream → assemble text → extract yaml fence
	mk := func(content, finish string) string {
		type delta struct {
			Content string `json:"content,omitempty"`
		}
		type choice struct {
			Index        int    `json:"index"`
			Delta        delta  `json:"delta"`
			FinishReason *string `json:"finish_reason"`
		}
		var fr *string
		if finish != "" {
			fr = &finish
		}
		payload, _ := json.Marshal(map[string]any{
			"id": "1", "object": "chat.completion.chunk", "model": "gemma",
			"choices": []choice{{Index: 0, Delta: delta{Content: content}, FinishReason: fr}},
		})
		return "data: " + string(payload)
	}
	chunks := []string{
		mk("Here:\n```yaml\n", ""),
		mk("kind: offer\narea: Setubal\nprice_eur: 600\n", ""),
		mk("```\n", ""),
		mk("", "stop"),
		"data: [DONE]",
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %s", r.URL.Path)
		}
		var body openAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode: %v", err)
		}
		if body.Model != "gemma-3-4b-it" || !body.Stream {
			t.Errorf("body model=%q stream=%v", body.Model, body.Stream)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, line := range chunks {
			fmt.Fprintln(w, line)
		}
	}))
	defer srv.Close()

	client := NewOpenWebUiClient(&DSN{URL: srv.URL + "/v1", API: APIOpenAI})
	var assembled strings.Builder
	var blocks []*CodeBlock
	err := client.Query(Request{
		Model:  "gemma-3-4b-it",
		Prompt: "extract listing",
		OnJson: func(res Response) error {
			if res.Response != nil {
				assembled.WriteString(*res.Response)
			}
			return nil
		},
		OnCodeBlock: func(b []*CodeBlock) error {
			blocks = append(blocks, b...)
			return nil
		},
	})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if !strings.Contains(assembled.String(), "kind: offer") {
		t.Fatalf("assembled = %q", assembled.String())
	}
	if len(blocks) != 1 || blocks[0].Type != "yaml" {
		t.Fatalf("blocks = %+v", blocks)
	}
	if !strings.Contains(blocks[0].Code, "price_eur: 600") {
		t.Fatalf("yaml = %q", blocks[0].Code)
	}
}
