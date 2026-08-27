// Package main demonstrates go-ollama against an OpenAI-compatible llama-server
// (e.g. arc-1 Gemma on :18434).
//
//	ssh -L 18434:127.0.0.1:18434 arc-1
//	export OPENAI_BASE_URL=http://127.0.0.1:18434/v1
//	export OPENAI_MODEL=gemma-3-4b-it
//	go run ./examples/openai-llama/
package main

import (
	"fmt"
	"os"

	ollama "github.com/eslider/go-ollama"
)

func main() {
	base := os.Getenv("OPENAI_BASE_URL")
	if base == "" {
		base = "http://127.0.0.1:18434/v1"
	}
	model := os.Getenv("OPENAI_MODEL")
	if model == "" {
		model = "gemma-3-4b-it"
	}

	client := ollama.NewOpenWebUiClient(&ollama.DSN{
		URL: base,
		API: ollama.APIOpenAI,
	})

	var yamlBlocks []*ollama.CodeBlock
	err := client.Query(ollama.Request{
		Model: model,
		System: ollama.String(
			"Extract housing offers as a single YAML fenced block with keys: kind, area, price_eur, rooms, summary.",
		),
		Prompt: "Сдам T1 в Сетубале, 600€, с парковкой, с 1 сентября.",
		Options: &ollama.RequestOptions{
			Temperature: ollama.Float(0),
		},
		OnJson: func(res ollama.Response) error {
			if res.Response != nil {
				fmt.Print(*res.Response)
			}
			return nil
		},
		OnCodeBlock: func(blocks []*ollama.CodeBlock) error {
			yamlBlocks = append(yamlBlocks, blocks...)
			return nil
		},
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nError: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("\n\nYAML blocks: %d\n", len(yamlBlocks))
	for i, b := range yamlBlocks {
		fmt.Printf("--- block %d [%s] ---\n%s\n", i+1, b.Type, b.Code)
	}
}
