package ollama

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// APIBackend selects the wire protocol used by Query.
type APIBackend string

const (
	// APIOllama is the native Ollama /api/generate NDJSON stream.
	APIOllama APIBackend = "ollama"
	// APIOpenAI is OpenAI-compatible chat completions (llama.cpp server, etc.).
	APIOpenAI APIBackend = "openai"
	// APICompletions is OpenAI-compatible /v1/completions (llama.cpp prompt API).
	APICompletions APIBackend = "completions"
)

// DefaultOpenAIChatURL is the local llama-server OpenAI chat endpoint.
const DefaultOpenAIChatURL = "http://127.0.0.1:18434/v1/chat/completions"

// resolveAPI picks openai vs ollama from DSN.API or URL path heuristics.
func (d *DSN) resolveAPI() APIBackend {
	if d == nil {
		return APIOllama
	}
	switch APIBackend(strings.ToLower(strings.TrimSpace(string(d.API)))) {
	case APICompletions:
		return APICompletions
	case APIOpenAI:
		return APIOpenAI
	case APIOllama:
		return APIOllama
	}
	u := strings.ToLower(d.URL)
	if strings.Contains(u, "/v1/completions") && !strings.Contains(u, "chat/completions") {
		return APICompletions
	}
	if strings.Contains(u, "/v1/") || strings.HasSuffix(u, "/v1") || strings.Contains(u, "chat/completions") {
		return APIOpenAI
	}
	return APIOllama
}

// chatCompletionsURL normalizes a DSN URL to .../v1/chat/completions.
func chatCompletionsURL(raw string) string {
	u := strings.TrimSpace(raw)
	if u == "" {
		return DefaultOpenAIChatURL
	}
	u = strings.TrimSuffix(u, "/")
	if strings.HasSuffix(u, "/chat/completions") {
		return u
	}
	if strings.HasSuffix(u, "/v1") {
		return u + "/chat/completions"
	}
	if strings.Contains(u, "/v1/") {
		// already something under /v1 — if not chat, still append chat path from base /v1
		if i := strings.Index(u, "/v1/"); i >= 0 {
			return u[:i+3] + "/chat/completions"
		}
	}
	return u + "/v1/chat/completions"
}

type openAIChatRequest struct {
	Model       string              `json:"model"`
	Messages    []openAIChatMessage `json:"messages"`
	Stream      bool                `json:"stream"`
	Temperature *float64            `json:"temperature,omitempty"`
	TopP        *float64            `json:"top_p,omitempty"`
	MaxTokens   *int                `json:"max_tokens,omitempty"`
	Stop        []string            `json:"stop,omitempty"`
}

type openAIChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIChatChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int    `json:"index"`
		FinishReason string `json:"finish_reason"`
		Delta        struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"delta"`
		Message *struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		Text string `json:"text"`
	} `json:"choices"`
}

func (c *Client) queryOpenAI(request Request) error {
	url := chatCompletionsURL(c.ds.URL)
	msgs := make([]openAIChatMessage, 0, 2)
	if request.System != nil && strings.TrimSpace(*request.System) != "" {
		msgs = append(msgs, openAIChatMessage{Role: "system", Content: *request.System})
	}
	msgs = append(msgs, openAIChatMessage{Role: "user", Content: request.Prompt})

	body := openAIChatRequest{
		Model:    request.Model,
		Messages: msgs,
		Stream:   true,
	}
	if request.Options != nil {
		body.Temperature = request.Options.Temperature
		body.TopP = request.Options.TopP
		body.MaxTokens = request.Options.NumPredict
		body.Stop = request.Options.Stop
	}

	data, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("openai marshal: %w", err)
	}
	req, err := http.NewRequest("POST", url, bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("openai request: %w", err)
	}
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Content-Type", "application/json")
	if c.ds.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.ds.Token)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("openai send: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("openai status %d: %s", resp.StatusCode, b)
	}

	return c.consumeOpenAIStream(resp.Body, request)
}

func (c *Client) consumeOpenAIStream(r io.Reader, request Request) error {
	shouldAnalyse := request.OnCodeBlock != nil
	var text string
	model := request.Model

	scanner := bufio.NewScanner(r)
	// SSE lines can be large for long YAML fences
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			// some servers send raw NDJSON
			if err := c.handleOpenAIPayload([]byte(line), &model, request, &text, shouldAnalyse); err != nil {
				return err
			}
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			done := true
			res := Response{Model: String(model), Response: String(""), Done: &done}
			if request.OnJson != nil {
				if err := request.OnJson(res); err != nil {
					return fmt.Errorf("openai OnJson: %w", err)
				}
			}
			if shouldAnalyse && text != "" {
				blocks := ParseCodeBlock(&text)
				if len(blocks) > 0 {
					if err := request.OnCodeBlock(blocks); err != nil {
						return fmt.Errorf("openai OnCodeBlock: %w", err)
					}
				}
			}
			return nil
		}
		if err := c.handleOpenAIPayload([]byte(payload), &model, request, &text, shouldAnalyse); err != nil {
			return err
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("openai stream: %w", err)
	}
	// flush remaining fence if stream ended without [DONE]
	if shouldAnalyse && text != "" {
		blocks := ParseCodeBlock(&text)
		if len(blocks) > 0 {
			if err := request.OnCodeBlock(blocks); err != nil {
				return fmt.Errorf("openai OnCodeBlock: %w", err)
			}
		}
	}
	return nil
}

func (c *Client) handleOpenAIPayload(raw []byte, model *string, request Request, text *string, shouldAnalyse bool) error {
	var chunk openAIChatChunk
	if err := json.Unmarshal(raw, &chunk); err != nil {
		return fmt.Errorf("openai unmarshal: %w", err)
	}
	if chunk.Model != "" {
		*model = chunk.Model
	}
	if len(chunk.Choices) == 0 {
		return nil
	}
	ch := chunk.Choices[0]
	content := ch.Delta.Content
	if content == "" && ch.Message != nil {
		content = ch.Message.Content
	}
	if content == "" {
		content = ch.Text
	}
	done := ch.FinishReason != ""
	res := Response{
		Model:    String(*model),
		Response: String(content),
		Done:     &done,
	}
	if request.OnJson != nil {
		if err := request.OnJson(res); err != nil {
			return fmt.Errorf("openai OnJson: %w", err)
		}
	}
	if shouldAnalyse && content != "" {
		*text = *text + content
		blocks := ParseCodeBlock(text)
		if len(blocks) > 0 {
			*text = ""
			if err := request.OnCodeBlock(blocks); err != nil {
				return fmt.Errorf("openai OnCodeBlock: %w", err)
			}
		}
	}
	return nil
}
