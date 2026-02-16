// Package ollama This package provides a client for the ollama OpenWeb UI
// to use Authenticated API calls. It also provides a SplitScanner to split
// the response from the ollama API by new line.
// The client sends a request to the ollama API and processes the response line by line in time like websockets
package ollama

import (
	base64 "encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

// Client is a client for the ollama Web UI to use Authenticated API calls
type Client struct {
	client *http.Client // HTTP client
	ds     *DSN         // Data source name
}

// DSN is a data source name for the ollama API
type DSN struct {
	URL   string // URL of the ollama API
	Token string // Token for the ollama API
}

// RequestOptions are options for the ollama API
type RequestOptions struct {
	NumContext       *int     `json:"num_ctx,omitempty"`           // See: https://github.com/ollama/ollama/blob/main/docs/faq.md#how-can-i-specify-the-context-window-size
	NumBatch         *int     `json:"num_batch,omitempty"`         // Number of tokens to generate in a single batch
	NumKeep          *int     `json:"num_keep,omitempty"`          // Number of tokens to keep in the context
	Seed             *int     `json:"seed,omitempty"`              // Random seed - for reproducibility, which means that the same seed will produce the same results
	NumPredict       *int     `json:"num_predict,omitempty"`       // Number of tokens to predict
	TopK             *int     `json:"top_k,omitempty"`             // The number of top tokens to consider
	TopP             *float64 `json:"top_p,omitempty"`             // The cumulative probability of the top tokens
	MinP             *int     `json:"min_p,omitempty"`             // The minimum probability of a token
	TfsZ             *float64 `json:"tfs_z,omitempty"`             // The temperature scaling factor
	TypicalP         *float64 `json:"typical_p,omitempty"`         // The typical probability of a token
	RepeatLastN      *int     `json:"repeat_last_n,omitempty"`     // The number of tokens to consider for the repeat penalty
	Temperature      *float64 `json:"temperature,omitempty"`       // The higher the temperature, the more random the output
	RepeatPenalty    *float64 `json:"repeat_penalty,omitempty"`    // The penalty for repeating tokens
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`  // The penalty for tokens that are already present in the context
	FrequencyPenalty *int     `json:"frequency_penalty,omitempty"` // Frequency penalty which is applied to tokens that are already present in the context
	Mirostat         *int     `json:"mirostat,omitempty"`          // Mirostat is a new sampling method that is more efficient than nucleus sampling
	MirostatTau      *float64 `json:"mirostat_tau,omitempty"`      // Entropy parameter for Mirostat sampling
	MirostatEta      *float64 `json:"mirostat_eta,omitempty"`      // Temperature parameter for Mirostat sampling
	Stop             []string `json:"stop,omitempty"`              // The tokens to stop generation at
	NUMA             *bool    `json:"numa,omitempty"`              // NUMA - Non-Uniform Memory Access
	NumGPU           *int     `json:"num_gpu,omitempty"`           // Number of GPUs to use
	MainGPU          *int     `json:"main_gpu,omitempty"`          // The main GPU to use
	NumThread        *int     `json:"num_thread,omitempty"`        // Number of threads to use
	PadTokens        *int     `json:"pad_tokens,omitempty"`        // The number of padding tokens
	PenalizeNewline  *bool    `json:"penalize_newline,omitempty"`  // Penalize newline tokens
	LowVRAM          *bool    `json:"low_vram,omitempty"`          // Low VRAM mode
	F16Kv            *bool    `json:"f16_kv,omitempty"`            // F16 key-value mode
	VocabOnly        *bool    `json:"vocab_only,omitempty"`        // Vocab only mode
	UseMlock         *bool    `json:"use_mlock,omitempty"`         // Use mlock means that the model will be locked into memory
	UseMmap          *bool    `json:"use_mmap,omitempty"`          // Use mmap means that the model will be memory-mapped
}

// RequestFormat is a format of the request
type RequestFormat string

// Enumerate formats
const (
	FormatJson RequestFormat = "json"
	FormatText RequestFormat = "text"
)

// Request is a request to the ollama API
type Request struct {
	Model       string                   `json:"model"`
	Prompt      string                   `json:"prompt"`               // See: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
	System      *string                  `json:"system,omitempty"`     // (optional) system message to override the model's default system prompt
	Format      *RequestFormat           `json:"format,omitempty"`     // By default is text, but can be json
	Options     *RequestOptions          `json:"options,omitempty"`    // (optional) the options to use for the model
	Suffix      *string                  `json:"suffix,omitempty"`     //  the text after the model response
	Images      []RequestImage           `json:"images,omitempty"`     // (optional) a list of base64-encoded images (for multimodal models such as llava)
	Context     []int                    `json:"-"`                    // (optional) the context to use for the model
	KeepAlive   *string                  `json:"keep_alive,omitempty"` // (optional) controls how long the model will stay loaded into memory following the request (default: 5m)
	Raw         *bool                    `json:"raw,omitempty"`        // (optional) controls how long the model will stay loaded into memory following the request (default: 5m)
	Stream      *bool                    `json:"stream,omitempty"`     // (optional) if true, the response will be streamed line by line
	OnJson      func(Response) error     `json:"-"`
	OnCodeBlock func([]*CodeBlock) error `json:"-"`
}

type RequestImage []byte

// MarshalJSON converts the image to base64
func (i RequestImage) MarshalJSON() ([]byte, error) {
	// See: https://github.com/ollama/ollama/issues/6972
	return json.Marshal(base64.StdEncoding.EncodeToString(i))
}

type Response struct {
	Model     *string    `json:"model,omitempty"`
	CreatedAt *time.Time `json:"created_at,omitempty"`
	Response  *string    `json:"response,omitempty"`
	Done      *bool      `json:"done,omitempty"`
}

// ToJson converts the Request to a JSON string
func (r *Request) ToJson() string {
	data, err := json.Marshal(r)

	if r.Images != nil {
		// Convert images to base64
		r.Stream = Bool(false)
		if r.Model == "" {
			r.Model = "x/llama3.2-vision"
		}
	}

	if err != nil {
		return ""
	}
	return string(data)
}

// NewOpenWebUiClient creates a new Client
func NewOpenWebUiClient(dsn *DSN) *Client {
	return &Client{
		client: &http.Client{
			Timeout: 0,
		},
		ds: dsn,
	}
}

// ProcessModelDetails holds model format metadata from the ps endpoint.
type ProcessModelDetails struct {
	ParentModel   string   `json:"parent_model"`
	Format        string   `json:"format"`
	Family        string   `json:"family"`
	Families      []string `json:"families"`
	ParameterSize string   `json:"parameter_size"`
	QuantLevel    string   `json:"quantization_level"`
}

// ProcessModel describes a currently loaded model returned by /api/ps.
type ProcessModel struct {
	Name          string              `json:"name"`
	Model         string              `json:"model"`
	Size          int64               `json:"size"`
	Digest        string              `json:"digest"`
	Details       ProcessModelDetails `json:"details"`
	ExpiresAt     *time.Time          `json:"expires_at,omitempty"`
	SizeVRAM      int64               `json:"size_vram"`
	ContextLength int                 `json:"context_length"`
}

// ProcessStatus is the response from /api/ps listing running models.
type ProcessStatus struct {
	Models []ProcessModel `json:"models"`
}

// Ps returns the list of models currently loaded in memory.
// The URL is derived from the DSN by replacing the last path segment with "ps".
func (c *Client) Ps() (*ProcessStatus, error) {
	psURL := strings.TrimSuffix(c.ds.URL, "/")
	if i := strings.LastIndex(psURL, "/"); i >= 0 {
		psURL = psURL[:i] + "/ps"
	}

	req, err := http.NewRequest("GET", psURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create ps request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.ds.Token)

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send ps request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ps request failed, status code: %d, body: %s", resp.StatusCode, body)
	}

	var status ProcessStatus
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return nil, fmt.Errorf("failed to decode ps response: %w", err)
	}
	return &status, nil
}

// Query sends a request to the ollama API
func (c *Client) Query(request Request) (err error) {
	js := request.ToJson()
	req, err := http.NewRequest("POST", c.ds.URL, strings.NewReader(js))

	if err != nil {
		return fmt.Errorf("failed to create ollama request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.ds.Token)

	// Response comes line by line
	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send ollama request: %w", err)
	}

	// Check if response code is 200
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("failed to send ollama request, status code: %d, body: %s", resp.StatusCode, body)
	}

	defer resp.Body.Close()

	var (
		scanner = NewSplitScanner(resp.Body, "\n") // Scanner to split response by new line which is JSON terminated by new line
		res     Response                           // Response of the ollama API
	)

	shouldAnalyse := false
	// Collect responseses for code blocks

	var text string
	if request.OnCodeBlock != nil {
		shouldAnalyse = true

	}

	for scanner.Scan() {
		// Check for errors
		if err = scanner.Err(); err != nil {
			return fmt.Errorf("failed to read ollama response: %w", err)
		}

		if err = json.Unmarshal(scanner.Bytes(), &res); err != nil {
			return fmt.Errorf("failed to unmarshal ollama response: %w", err)
		}

		// Unmarshal JSON response and call OnJson handler
		if request.OnJson != nil {
			if err = request.OnJson(res); err != nil {
				return fmt.Errorf("failed to process ollama response: %w", err)
			}
		}

		// Do we need to analyse the response, for blocks of code?
		if shouldAnalyse {
			// Join responses into a single string and find all markdown code blocks by extracting text inside "```" blocks
			text = strings.Join([]string{text, *res.Response}, "")
			blocks := ParseCodeBlock(&text)
			if len(blocks) > 0 {
				// Clear text
				text = ""
				err = request.OnCodeBlock(blocks)
				if err != nil {
					return fmt.Errorf("failed to process ollama response code block: %w", err)
				}
			}

		}
	}
	return
}

// CodeBlock is a code block extracted from the response
type CodeBlock struct {
	Type string
	Code string
}

// CodeBlockRegExp is a regular expression to extract code blocks from the text
var CodeBlockRegExp = regexp.MustCompile("(?s)``+(\\S+)(.+?)\n``+")

// ParseCodeBlock parses the code block from the response
// Use regular expressions to extract code blocks from the text
func ParseCodeBlock(text *string) (blocks []*CodeBlock) {
	for _, match := range CodeBlockRegExp.FindAllStringSubmatch(*text, -1) {
		if len(match) > 2 {
			block := &CodeBlock{
				Type: match[1],
				Code: match[2],
			}
			blocks = append(blocks, block)
			//codeFile, err := oolama.OpenFileDescriptor(fmt.Sprintf("%s_%d.%s", fileName, i+1, lang))
		}
	}

	return blocks
}
