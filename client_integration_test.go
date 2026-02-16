package ollama

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

// Integration tests that hit a live Ollama / Open WebUI instance.
// Skipped when OPEN_WEB_API_GENERATE_URL is not set.

func integrationClient(t *testing.T) *Client {
	t.Helper()
	url := os.Getenv("OPEN_WEB_API_GENERATE_URL")
	token := os.Getenv("OPEN_WEB_API_TOKEN")
	if url == "" {
		t.Skip("OPEN_WEB_API_GENERATE_URL not set, skipping integration test")
	}
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

	err := client.Query(Request{
		Model:  "gemma3:1b",
		Prompt: "What is your name?",
		System: String("Your name is TestBot. Always introduce yourself by name."),
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

func formatBytes(b int64) string {
	switch {
	case b >= 1<<30:
		return fmt.Sprintf("%.1fGB", float64(b)/(1<<30))
	case b >= 1<<20:
		return fmt.Sprintf("%.0fMB", float64(b)/(1<<20))
	default:
		return fmt.Sprintf("%dB", b)
	}
}
