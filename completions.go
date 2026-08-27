package ollama

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// DefaultOpenAICompletionsURL is llama.cpp /v1/completions on the usual remote-forward port.
const DefaultOpenAICompletionsURL = "http://127.0.0.1:8102/v1/completions"

type openAICompletionsRequest struct {
	Model       string   `json:"model,omitempty"`
	Prompt      string   `json:"prompt"`
	Stream      bool     `json:"stream"`
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	Stop        []string `json:"stop,omitempty"`
	CachePrompt bool     `json:"cache_prompt,omitempty"`
}

func completionsURL(raw string) string {
	u := strings.TrimSpace(raw)
	if u == "" {
		return DefaultOpenAICompletionsURL
	}
	u = strings.TrimSuffix(u, "/")
	if strings.HasSuffix(u, "/completions") && !strings.Contains(u, "chat/completions") {
		return u
	}
	if strings.HasSuffix(u, "/chat/completions") {
		return strings.TrimSuffix(u, "/chat/completions") + "/completions"
	}
	if strings.HasSuffix(u, "/v1") {
		return u + "/completions"
	}
	if i := strings.Index(u, "/v1/"); i >= 0 {
		return u[:i+3] + "/completions"
	}
	return u + "/v1/completions"
}

func (c *Client) queryCompletions(request Request) error {
	url := completionsURL(c.ds.URL)
	prompt := request.Prompt
	if request.System != nil && strings.TrimSpace(*request.System) != "" {
		prompt = strings.TrimSpace(*request.System) + "\n\n" + prompt
	}
	stream := true
	if request.Stream != nil {
		stream = *request.Stream
	}
	body := openAICompletionsRequest{
		Model:  request.Model,
		Prompt: prompt,
		Stream: stream,
	}
	if request.Options != nil {
		body.Temperature = request.Options.Temperature
		body.TopP = request.Options.TopP
		body.MaxTokens = request.Options.NumPredict
		body.Stop = request.Options.Stop
	}

	data, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("completions marshal: %w", err)
	}
	req, err := http.NewRequest("POST", url, bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("completions request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if stream {
		req.Header.Set("Accept", "text/event-stream")
	} else {
		req.Header.Set("Accept", "application/json")
	}
	if c.ds.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.ds.Token)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("completions send: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("completions status %d: %s", resp.StatusCode, b)
	}
	if stream {
		return c.consumeOpenAIStream(resp.Body, request)
	}
	return c.consumeCompletionsJSON(resp.Body, request)
}

func (c *Client) consumeCompletionsJSON(r io.Reader, request Request) error {
	raw, err := io.ReadAll(r)
	if err != nil {
		return fmt.Errorf("completions read: %w", err)
	}
	var chunk openAIChatChunk
	if err := json.Unmarshal(raw, &chunk); err != nil {
		return fmt.Errorf("completions unmarshal: %w", err)
	}
	model := request.Model
	text := ""
	shouldAnalyse := request.OnCodeBlock != nil
	if err := c.handleOpenAIPayload(raw, &model, request, &text, shouldAnalyse); err != nil {
		return err
	}
	if shouldAnalyse && text != "" {
		blocks := ParseCodeBlock(&text)
		if len(blocks) > 0 {
			if err := request.OnCodeBlock(blocks); err != nil {
				return fmt.Errorf("completions OnCodeBlock: %w", err)
			}
		}
	}
	done := true
	if request.OnJson != nil {
		res := Response{Model: String(model), Response: String(""), Done: &done}
		if err := request.OnJson(res); err != nil {
			return fmt.Errorf("completions OnJson: %w", err)
		}
	}
	return nil
}
