// Package main dials llama.cpp /v1/completions through SSH (go-sshlib).
//
//	export OLLAMA_SSH=naj-mdx-1
//	export OPENAI_BASE_URL=http://127.0.0.1:8102/v1/completions
//	go run ./examples/ssh-completions/
package main

import (
	"fmt"
	"os"

	ollama "github.com/eslider/go-ollama"
)

func main() {
	base := os.Getenv("OPENAI_BASE_URL")
	if base == "" {
		base = ollama.DefaultOpenAICompletionsURL
	}
	sshHost := os.Getenv("OLLAMA_SSH")
	if sshHost == "" {
		sshHost = "naj-mdx-1"
	}

	client := ollama.NewOpenWebUiClient(&ollama.DSN{
		URL: base,
		API: ollama.APICompletions,
		SSH: sshHost,
	})
	defer client.Close()

	prompt := os.Getenv("PROMPT")
	if prompt == "" {
		prompt = "Die heutige Schicht beginnt um"
	}

	err := client.Query(ollama.Request{
		Prompt: prompt,
		Stream: ollama.Bool(false),
		Options: &ollama.RequestOptions{
			Temperature: ollama.Float(0.7),
			Stop:        []string{"\n"},
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
}
