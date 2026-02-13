# go-ollama

Go client library for the [Ollama](https://ollama.com/) / [Open WebUI](https://openwebui.com/) API with streaming support.

## Features

- Authenticated API calls with Bearer token
- Streaming responses (line-by-line JSON processing, similar to WebSockets)
- Code block extraction from AI responses using regex
- Configurable model options (temperature, context size, GPU settings, etc.)
- Image support for multimodal models (base64-encoded)

## Installation

```bash
go get github.com/eslider/go-ollama
```

## Usage

```go
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

    err := client.Query(ollama.Request{
        Model:  "llama3.2:3b",
        Prompt: "Hello, how are you?",
        Options: &ollama.RequestOptions{
            Temperature: ollama.Float(0.7),
        },
        OnJson: func(res ollama.Response) error {
            fmt.Print(*res.Response)
            return nil
        },
    })
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
    }
}
```

## Environment Variables

| Variable | Description |
|---|---|
| `OPEN_WEB_API_GENERATE_URL` | Ollama/Open WebUI API endpoint URL |
| `OPEN_WEB_API_TOKEN` | Bearer token for authentication |

## Code Block Extraction

The client can automatically extract markdown code blocks from streaming responses:

```go
err := client.Query(ollama.Request{
    Model:  "llama3.2:3b",
    Prompt: "Write a Go hello world program",
    OnCodeBlock: func(blocks []*ollama.CodeBlock) error {
        for _, block := range blocks {
            fmt.Printf("Language: %s\nCode:\n%s\n", block.Type, block.Code)
        }
        return nil
    },
})
```

## License

[MIT](LICENSE)
