// Package main demonstrates basic usage of the ollama client library.
//
// By default it uses local Ollama. Optional overrides:
//
//	export OPEN_WEB_API_GENERATE_URL="https://ai.example.com/ollama/api/generate"
//	export OPEN_WEB_API_TOKEN="sk-..."
//	go run ./examples/basic/
package main

import (
	"fmt"
	"os"

	ollama "github.com/eslider/go-ollama"
)

func main() {
	client := ollama.NewOpenWebUiClient(&ollama.DSN{
		URL:   os.Getenv("OPEN_WEB_API_GENERATE_URL"),
		Token: os.Getenv("OPEN_WEB_API_TOKEN"),
	})

	// Simple text query with streaming output
	fmt.Println("--- Streaming text response ---")
	err := client.Query(ollama.Request{
		Model:  "llama3.2:3b",
		Prompt: "Write a haiku about Go programming",
		Options: &ollama.RequestOptions{
			Temperature: ollama.Float(0.7),
		},
		OnJson: func(res ollama.Response) error {
			if res.Response != nil {
				fmt.Print(*res.Response)
			}
			return nil
		},
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nError: %v\n", err)
		os.Exit(1)
	}
	fmt.Println()

	// Query with code block extraction
	fmt.Println("\n--- Code block extraction ---")
	var codeBlocks []*ollama.CodeBlock

	err = client.Query(ollama.Request{
		Model:  "llama3.2:3b",
		Prompt: "Write a simple Go HTTP server",
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
			codeBlocks = append(codeBlocks, blocks...)
			return nil
		},
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nError: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n\nExtracted %d code block(s):\n", len(codeBlocks))
	for i, block := range codeBlocks {
		fmt.Printf("  Block %d [%s]: %d bytes\n", i+1, block.Type, len(block.Code))
	}
}
