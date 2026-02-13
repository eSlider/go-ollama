package ollama

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// simulateStreamBody builds a newline-delimited JSON stream like the Ollama API.
func simulateStreamBody(tokens []string, model string) string {
	var sb strings.Builder
	now := time.Now()
	for _, tok := range tokens {
		r := Response{
			Model:     String(model),
			CreatedAt: &now,
			Response:  String(tok),
			Done:      Bool(false),
		}
		data, _ := json.Marshal(r)
		sb.Write(data)
		sb.WriteString("\n")
	}
	// Final "done" message
	final := Response{
		Model:     String(model),
		CreatedAt: &now,
		Response:  String(""),
		Done:      Bool(true),
	}
	data, _ := json.Marshal(final)
	sb.Write(data)
	sb.WriteString("\n")
	return sb.String()
}

func TestQueryStream_TokenByToken(t *testing.T) {
	tokens := []string{"Hello", ", ", "world", "!"}
	body := simulateStreamBody(tokens, "test-model")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, body)
	}))
	defer srv.Close()

	client := NewOpenWebUiClient(&DSN{URL: srv.URL, Token: "test-token"})

	var collected []string
	err := client.Query(Request{
		Model:  "test-model",
		Prompt: "say hello",
		OnJson: func(res Response) error {
			if res.Response != nil {
				collected = append(collected, *res.Response)
			}
			return nil
		},
	})
	if err != nil {
		t.Fatalf("Query error: %v", err)
	}

	// tokens + 1 final empty token
	wantCount := len(tokens) + 1
	if len(collected) != wantCount {
		t.Errorf("got %d callbacks, want %d", len(collected), wantCount)
	}

	full := strings.Join(collected, "")
	if full != "Hello, world!" {
		t.Errorf("assembled text = %q, want %q", full, "Hello, world!")
	}
}

func TestQueryStream_OnJsonReceivesModel(t *testing.T) {
	body := simulateStreamBody([]string{"hi"}, "llama3.2:3b")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, body)
	}))
	defer srv.Close()

	client := NewOpenWebUiClient(&DSN{URL: srv.URL})

	var models []string
	err := client.Query(Request{
		Model:  "llama3.2:3b",
		Prompt: "test",
		OnJson: func(res Response) error {
			if res.Model != nil {
				models = append(models, *res.Model)
			}
			return nil
		},
	})
	if err != nil {
		t.Fatalf("Query error: %v", err)
	}

	for _, m := range models {
		if m != "llama3.2:3b" {
			t.Errorf("model = %q, want %q", m, "llama3.2:3b")
		}
	}
}

func TestQueryStream_OnJsonDoneFlag(t *testing.T) {
	body := simulateStreamBody([]string{"a", "b"}, "m")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, body)
	}))
	defer srv.Close()

	client := NewOpenWebUiClient(&DSN{URL: srv.URL})

	var doneValues []bool
	err := client.Query(Request{
		Model:  "m",
		Prompt: "test",
		OnJson: func(res Response) error {
			if res.Done != nil {
				doneValues = append(doneValues, *res.Done)
			}
			return nil
		},
	})
	if err != nil {
		t.Fatalf("Query error: %v", err)
	}

	// "a" (false), "b" (false), "" (true)
	if len(doneValues) != 3 {
		t.Fatalf("got %d done values, want 3", len(doneValues))
	}
	if doneValues[0] || doneValues[1] {
		t.Error("intermediate tokens should have done=false")
	}
	if !doneValues[2] {
		t.Error("final token should have done=true")
	}
}

func TestQueryStream_OnJsonError(t *testing.T) {
	body := simulateStreamBody([]string{"a", "b"}, "m")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, body)
	}))
	defer srv.Close()

	client := NewOpenWebUiClient(&DSN{URL: srv.URL})

	sentinel := fmt.Errorf("stop now")
	err := client.Query(Request{
		Model:  "m",
		Prompt: "test",
		OnJson: func(res Response) error {
			return sentinel
		},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "stop now") {
		t.Errorf("error = %q, should contain 'stop now'", err.Error())
	}
}

func TestQueryStream_CodeBlockExtraction(t *testing.T) {
	// Simulate streaming a response that contains a Go code block
	codeResp := []string{
		"Here is a program:\n\n",
		"```go\n",
		"package main\n",
		"\n",
		"import \"fmt\"\n",
		"\n",
		"func main() {\n",
		"\tfmt.Println(\"hello\")\n",
		"}\n",
		"```\n",
		"\nThat's it!",
	}
	body := simulateStreamBody(codeResp, "m")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, body)
	}))
	defer srv.Close()

	client := NewOpenWebUiClient(&DSN{URL: srv.URL})

	var allBlocks []*CodeBlock
	err := client.Query(Request{
		Model:  "m",
		Prompt: "write code",
		OnCodeBlock: func(blocks []*CodeBlock) error {
			allBlocks = append(allBlocks, blocks...)
			return nil
		},
	})
	if err != nil {
		t.Fatalf("Query error: %v", err)
	}

	if len(allBlocks) != 1 {
		t.Fatalf("got %d code blocks, want 1", len(allBlocks))
	}
	if allBlocks[0].Type != "go" {
		t.Errorf("block type = %q, want %q", allBlocks[0].Type, "go")
	}
	if !strings.Contains(allBlocks[0].Code, "fmt.Println") {
		t.Errorf("block code should contain fmt.Println, got: %s", allBlocks[0].Code)
	}
}

func TestQueryStream_MultipleCodeBlocks(t *testing.T) {
	content := "Look:\n\n```python\nprint('hi')\n```\n\nAnd:\n\n```bash\necho hello\n```\n\nDone."
	body := simulateStreamBody([]string{content}, "m")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, body)
	}))
	defer srv.Close()

	client := NewOpenWebUiClient(&DSN{URL: srv.URL})

	var allBlocks []*CodeBlock
	err := client.Query(Request{
		Model:  "m",
		Prompt: "multi",
		OnCodeBlock: func(blocks []*CodeBlock) error {
			allBlocks = append(allBlocks, blocks...)
			return nil
		},
	})
	if err != nil {
		t.Fatalf("Query error: %v", err)
	}

	if len(allBlocks) != 2 {
		t.Fatalf("got %d code blocks, want 2", len(allBlocks))
	}
	if allBlocks[0].Type != "python" {
		t.Errorf("block[0].Type = %q, want python", allBlocks[0].Type)
	}
	if allBlocks[1].Type != "bash" {
		t.Errorf("block[1].Type = %q, want bash", allBlocks[1].Type)
	}
}

func TestQueryStream_OnCodeBlockError(t *testing.T) {
	content := "```go\nfmt.Println()\n```\n"
	body := simulateStreamBody([]string{content}, "m")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, body)
	}))
	defer srv.Close()

	client := NewOpenWebUiClient(&DSN{URL: srv.URL})

	sentinel := fmt.Errorf("block error")
	err := client.Query(Request{
		Model:  "m",
		Prompt: "test",
		OnCodeBlock: func(blocks []*CodeBlock) error {
			return sentinel
		},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "block error") {
		t.Errorf("error = %q, should contain 'block error'", err.Error())
	}
}

func TestQueryStream_BothCallbacks(t *testing.T) {
	content := "Here:\n```sql\nSELECT 1;\n```\nDone"
	body := simulateStreamBody([]string{content}, "m")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, body)
	}))
	defer srv.Close()

	client := NewOpenWebUiClient(&DSN{URL: srv.URL})

	var jsonCalls int
	var blocks []*CodeBlock
	err := client.Query(Request{
		Model:  "m",
		Prompt: "test",
		OnJson: func(res Response) error {
			jsonCalls++
			return nil
		},
		OnCodeBlock: func(b []*CodeBlock) error {
			blocks = append(blocks, b...)
			return nil
		},
	})
	if err != nil {
		t.Fatalf("Query error: %v", err)
	}
	if jsonCalls == 0 {
		t.Error("OnJson was never called")
	}
	if len(blocks) != 1 {
		t.Errorf("got %d code blocks, want 1", len(blocks))
	}
	if blocks[0].Type != "sql" {
		t.Errorf("block type = %q, want sql", blocks[0].Type)
	}
}

func TestQuery_AuthorizationHeader(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		fmt.Fprint(w, simulateStreamBody([]string{"ok"}, "m"))
	}))
	defer srv.Close()

	client := NewOpenWebUiClient(&DSN{URL: srv.URL, Token: "secret-123"})
	_ = client.Query(Request{
		Model:  "m",
		Prompt: "test",
		OnJson: func(res Response) error { return nil },
	})

	if gotAuth != "Bearer secret-123" {
		t.Errorf("Authorization = %q, want %q", gotAuth, "Bearer secret-123")
	}
}

func TestQuery_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, `{"error":"model not found"}`)
	}))
	defer srv.Close()

	client := NewOpenWebUiClient(&DSN{URL: srv.URL})
	err := client.Query(Request{
		Model:  "m",
		Prompt: "test",
		OnJson: func(res Response) error { return nil },
	})
	if err == nil {
		t.Fatal("expected error for 500 response")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error = %q, should mention status 500", err.Error())
	}
}

func TestQuery_RequestJSON(t *testing.T) {
	var gotBody string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := readAll(r.Body)
		gotBody = string(b)
		fmt.Fprint(w, simulateStreamBody([]string{"ok"}, "m"))
	}))
	defer srv.Close()

	client := NewOpenWebUiClient(&DSN{URL: srv.URL})
	_ = client.Query(Request{
		Model:  "llama3.2:3b",
		Prompt: "test prompt",
		Options: &RequestOptions{
			Temperature: Float(0.5),
			NumContext:  Int(4096),
		},
		OnJson: func(res Response) error { return nil },
	})

	if !strings.Contains(gotBody, `"model":"llama3.2:3b"`) {
		t.Errorf("body missing model: %s", gotBody)
	}
	if !strings.Contains(gotBody, `"prompt":"test prompt"`) {
		t.Errorf("body missing prompt: %s", gotBody)
	}
	if !strings.Contains(gotBody, `"temperature":0.5`) {
		t.Errorf("body missing temperature: %s", gotBody)
	}
	if !strings.Contains(gotBody, `"num_ctx":4096`) {
		t.Errorf("body missing num_ctx: %s", gotBody)
	}
}

func readAll(r interface{ Read([]byte) (int, error) }) ([]byte, error) {
	var buf strings.Builder
	b := make([]byte, 1024)
	for {
		n, err := r.Read(b)
		if n > 0 {
			buf.Write(b[:n])
		}
		if err != nil {
			break
		}
	}
	return []byte(buf.String()), nil
}
