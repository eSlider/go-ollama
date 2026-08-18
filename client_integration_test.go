package ollama

import (
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

// Integration tests that hit a live Ollama / Open WebUI instance.
// When OPEN_WEB_API_GENERATE_URL is unset, NewOpenWebUiClient uses local Ollama (DefaultGenerateURL).
// Skipped in CI where no Ollama service is available.

func integrationClient(t *testing.T) *Client {
	t.Helper()
	if os.Getenv("CI") != "" {
		t.Skip("integration tests require a live Ollama instance; skipped in CI")
	}
	url := os.Getenv("OPEN_WEB_API_GENERATE_URL")
	token := os.Getenv("OPEN_WEB_API_TOKEN")
	return NewOpenWebUiClient(&DSN{URL: url, Token: token})
}

func TestIntegration_StreamTokenByToken(t *testing.T) {
	client := integrationClient(t)

	var tokens []string
	var gotDone bool
	start := time.Now()

	err := client.Query(Request{
		Model:  "gemma3:1b",
		Prompt: "Say hello in one word",
		Options: &RequestOptions{
			Temperature: Float(0),
			NumPredict:  Int(10),
		},
		OnJson: func(res Response) error {
			if res.Response != nil {
				tokens = append(tokens, *res.Response)
			}
			if res.Done != nil && *res.Done {
				gotDone = true
			}
			return nil
		},
	})
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Query error: %v", err)
	}
	if !gotDone {
		t.Error("never received done=true")
	}
	if len(tokens) == 0 {
		t.Fatal("received zero tokens")
	}

	full := strings.Join(tokens, "")
	t.Logf("Response: %q (%d tokens in %v)", full, len(tokens), elapsed)

	if full == "" {
		t.Error("assembled response is empty")
	}
}

func TestIntegration_SystemPrompt(t *testing.T) {
	client := integrationClient(t)

	var full strings.Builder

	s := "Your name is TestBot. Always introduce yourself by name."
	err := client.Query(Request{
		Model:  "gemma3:1b",
		Prompt: "What is your name?",
		System: &s,
		Options: &RequestOptions{
			Temperature: Float(0),
			NumPredict:  Int(30),
		},
		OnJson: func(res Response) error {
			if res.Response != nil {
				full.WriteString(*res.Response)
			}
			return nil
		},
	})
	if err != nil {
		t.Fatalf("Query error: %v", err)
	}

	resp := full.String()
	t.Logf("Response: %q", resp)

	if !strings.Contains(strings.ToLower(resp), "testbot") {
		t.Errorf("expected response to contain 'TestBot', got: %s", resp)
	}
}

func TestIntegration_TokensPerSecond(t *testing.T) {
	client := integrationClient(t)

	var tokenCount int
	start := time.Now()

	err := client.Query(Request{
		Model:  "gemma3:1b",
		Prompt: "Count from 1 to 10",
		Options: &RequestOptions{
			Temperature: Float(0),
			NumPredict:  Int(50),
		},
		OnJson: func(res Response) error {
			if res.Response != nil && *res.Response != "" {
				tokenCount++
			}
			return nil
		},
	})
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Query error: %v", err)
	}

	tokSec := float64(tokenCount) / elapsed.Seconds()
	t.Logf("Tokens: %d, Time: %v, Speed: %.1f tok/s", tokenCount, elapsed, tokSec)

	if tokenCount == 0 {
		t.Error("received zero tokens")
	}
	if tokSec <= 0 {
		t.Error("tokens per second should be positive")
	}
}

func TestIntegration_ConversationContext(t *testing.T) {
	client := integrationClient(t)

	// Build a conversation prompt like the TUI does.
	conversation := "User: My name is Alice\n\nAssistant: Nice to meet you, Alice!\n\nUser: What is my name?\n\nAssistant: "

	var full strings.Builder

	err := client.Query(Request{
		Model:  "gemma3:1b",
		Prompt: conversation,
		Options: &RequestOptions{
			Temperature: Float(0),
			NumPredict:  Int(30),
		},
		OnJson: func(res Response) error {
			if res.Response != nil {
				full.WriteString(*res.Response)
			}
			return nil
		},
	})
	if err != nil {
		t.Fatalf("Query error: %v", err)
	}

	resp := full.String()
	t.Logf("Response: %q", resp)

	if !strings.Contains(resp, "Alice") {
		t.Errorf("expected model to recall 'Alice', got: %s", resp)
	}
}

// TestIntegration_CodeBlockExtraction tests that the model can extract code blocks from a prompt.
// Does: Verify that the model can extract multiple code blocks from a prompt.
func TestIntegration_CodeBlockExtraction(t *testing.T) {
	client := integrationClient(t)

	var blocks []*CodeBlock
	var full strings.Builder

	err := client.Query(Request{
		Model:  "gemma3:1b",
		Prompt: "Write a Go hello world program in a code block",
		Options: &RequestOptions{
			Temperature: Float(0),
			NumPredict:  Int(100),
		},
		OnJson: func(res Response) error {
			if res.Response != nil {
				full.WriteString(*res.Response)
			}
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

	t.Logf("Response: %q", full.String())
	t.Logf("Code blocks found: %d", len(blocks))
	for i, b := range blocks {
		t.Logf("  Block %d [%s]: %s", i+1, b.Type, b.Code)
	}

	if len(blocks) == 0 {
		t.Error("expected at least one code block")
	}
}

func TestIntegration_Ps(t *testing.T) {
	client := integrationClient(t)

	status, err := client.Ps()
	if err != nil {
		t.Fatalf("Ps error: %v", err)
	}

	t.Logf("Running models: %d", len(status.Models))
	for _, m := range status.Models {
		vramMB := float64(m.SizeVRAM) / 1024 / 1024
		t.Logf("  %s  params=%s  quant=%s  vram=%.0fMB  ctx=%d",
			m.Name, m.Details.ParameterSize, m.Details.QuantLevel, vramMB, m.ContextLength)
		if m.ExpiresAt != nil {
			t.Logf("    expires: %s (in %s)", m.ExpiresAt.Format(time.RFC3339), time.Until(*m.ExpiresAt).Truncate(time.Second))
		}
	}

	// Verify struct fields are populated for any loaded model.
	for _, m := range status.Models {
		if m.Name == "" {
			t.Error("model name is empty")
		}
		if m.Size <= 0 {
			t.Errorf("model %s has invalid size: %d", m.Name, m.Size)
		}
		if m.Details.Family == "" {
			t.Errorf("model %s has empty family", m.Name)
		}
		if m.Details.ParameterSize == "" {
			t.Errorf("model %s has empty parameter_size", m.Name)
		}
	}
}

func TestIntegration_Embed(t *testing.T) {
	client := integrationClient(t)

	model := os.Getenv("OLLAMA_EMBED_MODEL")
	if model == "" {
		model = "llama3.2:1b"
	}

	resp, err := client.Embed(EmbedRequest{
		Model: model,
		Input: []string{"Hello world"},
	})
	if err != nil {
		if strings.Contains(err.Error(), "not support embeddings") || strings.Contains(err.Error(), "NaN") {
			t.Skipf("model %s does not support embeddings on this server: %v", model, err)
		}
		t.Fatalf("Embed error: %v", err)
	}

	if resp.Model != model {
		t.Errorf("model = %q, want %s", resp.Model, model)
	}
	if len(resp.Embeddings) != 1 {
		t.Fatalf("got %d embeddings, want 1", len(resp.Embeddings))
	}
	dim := len(resp.Embeddings[0])
	t.Logf("Embedding dimension: %d, eval tokens: %d, duration: %dns",
		dim, resp.PromptEvalCount, resp.TotalDuration)

	if dim == 0 {
		t.Error("embedding dimension is 0")
	}

	// Verify the vector is not all zeros.
	var norm float64
	for _, v := range resp.Embeddings[0] {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm < 0.01 {
		t.Error("embedding vector norm is near zero")
	}
}

func TestIntegration_EmbedBatch(t *testing.T) {
	client := integrationClient(t)

	inputs := []string{
		"The cat sat on the mat",
		"The dog lay on the rug",
		"Quantum computing uses qubits",
	}

	model := os.Getenv("OLLAMA_EMBED_MODEL")
	if model == "" {
		model = "llama3.2:1b"
	}

	resp, err := client.Embed(EmbedRequest{
		Model: model,
		Input: inputs,
	})
	if err != nil {
		if strings.Contains(err.Error(), "not support embeddings") || strings.Contains(err.Error(), "NaN") {
			t.Skipf("model %s does not support embeddings on this server: %v", model, err)
		}
		t.Fatalf("Embed error: %v", err)
	}

	if len(resp.Embeddings) != len(inputs) {
		t.Fatalf("got %d embeddings, want %d", len(resp.Embeddings), len(inputs))
	}

	// All embeddings should have the same dimension.
	dim := len(resp.Embeddings[0])
	for i, emb := range resp.Embeddings {
		if len(emb) != dim {
			t.Errorf("embedding[%d] dim = %d, want %d", i, len(emb), dim)
		}
	}

	// Similar sentences should be closer than dissimilar ones.
	simCatDog := cosineSimilarity(resp.Embeddings[0], resp.Embeddings[1])
	simCatQuantum := cosineSimilarity(resp.Embeddings[0], resp.Embeddings[2])
	t.Logf("Similarity cat/dog: %.4f, cat/quantum: %.4f", simCatDog, simCatQuantum)

	if simCatDog <= simCatQuantum {
		t.Errorf("expected cat/dog similarity (%.4f) > cat/quantum similarity (%.4f)",
			simCatDog, simCatQuantum)
	}
}

func cosineSimilarity(a, b []float64) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func TestIntegration_Version(t *testing.T) {
	client := integrationClient(t)

	ver, err := client.Version()
	if err != nil {
		t.Fatalf("Version error: %v", err)
	}
	t.Logf("Ollama version: %s", ver.Version)

	if ver.Version == "" {
		t.Error("version is empty")
	}
}

func TestIntegration_Tags(t *testing.T) {
	client := integrationClient(t)

	tags, err := client.Tags()
	if err != nil {
		t.Fatalf("Tags error: %v", err)
	}
	t.Logf("Available models: %d", len(tags.Models))

	if len(tags.Models) == 0 {
		t.Fatal("no models available")
	}
	for _, m := range tags.Models {
		if m.Name == "" {
			t.Error("model name is empty")
		}
		t.Logf("  %s  size=%d  family=%s", m.Name, m.Size, m.Details.Family)
	}
}

func TestIntegration_Show(t *testing.T) {
	client := integrationClient(t)

	resp, err := client.Show(ShowRequest{Name: "gemma3:1b"})
	if err != nil {
		t.Fatalf("Show error: %v", err)
	}

	t.Logf("Modelfile length: %d", len(resp.Modelfile))
	t.Logf("Parameters: %s", resp.Parameters)
	t.Logf("Details: family=%s params=%s quant=%s",
		resp.Details.Family, resp.Details.ParameterSize, resp.Details.QuantLevel)
	t.Logf("ModelInfo keys: %d", len(resp.ModelInfo))

	if resp.Details.Family == "" {
		t.Error("details.family is empty")
	}
	if resp.Details.ParameterSize == "" {
		t.Error("details.parameter_size is empty")
	}
}

func TestIntegration_Chat(t *testing.T) {
	client := integrationClient(t)

	var tokens []string
	var finalResp ChatResponse

	err := client.Chat(ChatRequest{
		Model: "gemma3:1b",
		Messages: []ChatMessage{
			{Role: "system", Content: "You are a helpful assistant named TestBot."},
			{Role: "user", Content: "What is your name? Reply in one sentence."},
		},
		Options: &RequestOptions{
			Temperature: Float(0),
			NumPredict:  Int(30),
		},
	}, func(res ChatResponse) error {
		if res.Message.Content != "" {
			tokens = append(tokens, res.Message.Content)
		}
		if res.Done {
			finalResp = res
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	full := strings.Join(tokens, "")
	t.Logf("Response: %q (%d chunks)", full, len(tokens))
	t.Logf("Stats: prompt_eval=%d, eval=%d", finalResp.PromptEvalCount, finalResp.EvalCount)

	if full == "" {
		t.Error("empty response")
	}
	if !strings.Contains(strings.ToLower(full), "testbot") {
		t.Errorf("expected 'TestBot' in response, got: %s", full)
	}
	if finalResp.PromptEvalCount == 0 {
		t.Error("prompt_eval_count is 0")
	}
}

func TestIntegration_ChatMultiTurn(t *testing.T) {
	client := integrationClient(t)

	var full strings.Builder

	err := client.Chat(ChatRequest{
		Model: "gemma3:1b",
		Messages: []ChatMessage{
			{Role: "system", Content: "You have perfect memory. Always recall facts from the conversation."},
			{Role: "user", Content: "Remember this: my name is Alice."},
			{Role: "assistant", Content: "Got it! Your name is Alice. I'll remember that."},
			{Role: "user", Content: "Say my name."},
		},
		Options: &RequestOptions{
			Temperature: Float(0),
			NumPredict:  Int(30),
		},
	}, func(res ChatResponse) error {
		full.WriteString(res.Message.Content)
		return nil
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	resp := full.String()
	t.Logf("Response: %q", resp)

	if !strings.Contains(resp, "Alice") {
		t.Errorf("expected model to recall 'Alice', got: %s", resp)
	}
}

func TestIntegration_Copy(t *testing.T) {
	client := integrationClient(t)

	err := client.Copy(CopyRequest{
		Source:      "gemma3:1b",
		Destination: "gemma3:1b-test-copy",
	})
	if err != nil {
		t.Fatalf("Copy error: %v", err)
	}
	t.Log("Copy succeeded")

	// Clean up
	err = client.Delete(DeleteRequest{Name: "gemma3:1b-test-copy"})
	if err != nil {
		t.Fatalf("Delete cleanup error: %v", err)
	}
	t.Log("Delete cleanup succeeded")
}
