// Package ollama This package provides a client for the ollama OpenWeb UI
// to use Authenticated API calls. It also provides a SplitScanner to split
// the response from the ollama API by new line.
// The client sends a request to the ollama API and processes the response line by line in time like websockets
package ollama

import (
	"context"
	"crypto/tls"
	base64 "encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"regexp"
	"strings"
	"time"
)

// DefaultGenerateURL is the Ollama /api/generate endpoint on the local default port.
// Used when DSN.URL is empty in NewOpenWebUiClient.
const DefaultGenerateURL = "http://localhost:11434/api/generate"

// Client is a client for the ollama Web UI to use Authenticated API calls
type Client struct {
	client *http.Client // HTTP client
	ds     *DSN         // Data source name
	closer io.Closer    // optional SSH pool
}

// DSN is a data source name for the ollama API
type DSN struct {
	URL   string     // URL of the ollama /api/generate, OpenAI /v1, or /v1/completions
	Token string     // Token for the ollama API / Bearer for OpenAI-compatible servers
	API   APIBackend // ollama | openai | completions; empty = auto-detect from URL
	// SSH is an OpenSSH Host alias (e.g. "naj-mdx-1"). When set, HTTP is dialed
	// through go-sshlib to the URL host:port on the remote side (ssh -W).
	SSH string
	// DialContext overrides TCP dial (tests, custom tunnels). Wins over SSH.
	DialContext func(ctx context.Context, network, addr string) (net.Conn, error)
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
	Model           *string    `json:"model,omitempty"`
	CreatedAt       *time.Time `json:"created_at,omitempty"`
	Response        *string    `json:"response,omitempty"`
	Done            *bool      `json:"done,omitempty"`
	PromptEvalCount *int       `json:"prompt_eval_count,omitempty"`
	EvalCount       *int       `json:"eval_count,omitempty"`
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

// NewOpenWebUiClient creates a new Client.
// If dsn is nil or dsn.URL is empty (after trimming space), URL defaults to DefaultGenerateURL (local Ollama).
func NewOpenWebUiClient(dsn *DSN) *Client {
	var resolved DSN
	if dsn != nil {
		resolved = *dsn
	}
	if strings.TrimSpace(resolved.URL) == "" {
		resolved.URL = DefaultGenerateURL
	}
	transport := &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
	}
	var closer io.Closer
	if resolved.DialContext != nil {
		transport.DialContext = resolved.DialContext
	} else if alias := strings.TrimSpace(resolved.SSH); alias != "" {
		pool := newSSHPool(alias)
		transport.DialContext = pool.DialContext
		closer = pool
	}
	return &Client{
		client: &http.Client{Timeout: 0, Transport: transport},
		ds:     &resolved,
		closer: closer,
	}
}

// Close releases an SSH tunnel opened via DSN.SSH.
func (c *Client) Close() error {
	if c == nil || c.closer == nil {
		return nil
	}
	return c.closer.Close()
}

// apiURL replaces the last path segment of the DSN URL with the given segment.
// e.g. "http://host/api/generate" + "tags" → "http://host/api/tags"
func (c *Client) apiURL(segment string) string {
	base := strings.TrimSuffix(c.ds.URL, "/")
	if i := strings.LastIndex(base, "/"); i >= 0 {
		return base[:i] + "/" + segment
	}
	return base + "/" + segment
}

// doJSON performs an HTTP request and decodes the JSON response into dest.
func (c *Client) doJSON(method, url string, body interface{}, dest interface{}) error {
	var reqBody io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("failed to marshal request body: %w", err)
		}
		reqBody = strings.NewReader(string(data))
	}

	req, err := http.NewRequest(method, url, reqBody)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.ds.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.ds.Token)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("request failed, status code: %d, body: %s", resp.StatusCode, respBody)
	}

	if dest != nil {
		if err := json.NewDecoder(resp.Body).Decode(dest); err != nil {
			return fmt.Errorf("failed to decode response: %w", err)
		}
	}
	return nil
}

// doStream performs a POST and reads newline-delimited JSON, calling onJSON for each line.
func (c *Client) doStream(url string, body interface{}, onJSON func(json.RawMessage) error) error {
	data, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequest("POST", url, strings.NewReader(string(data)))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-Type", "application/json")
	if c.ds.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.ds.Token)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("request failed, status code: %d, body: %s", resp.StatusCode, respBody)
	}

	scanner := NewSplitScanner(resp.Body, "\n")
	for scanner.Scan() {
		if err := scanner.Err(); err != nil {
			return fmt.Errorf("failed to read response: %w", err)
		}
		if err := onJSON(scanner.Bytes()); err != nil {
			return err
		}
	}
	return nil
}

// --- /api/version ----------------------------------------------------------

// VersionResponse is the response from GET /api/version.
type VersionResponse struct {
	Version string `json:"version"`
}

// Version returns the Ollama server version.
func (c *Client) Version() (*VersionResponse, error) {
	var result VersionResponse
	if err := c.doJSON("GET", c.apiURL("version"), nil, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// --- /api/tags -------------------------------------------------------------

// TagModel describes a model returned by the /api/tags endpoint.
type TagModel struct {
	Name       string              `json:"name"`
	Model      string              `json:"model"`
	ModifiedAt string              `json:"modified_at"`
	Size       int64               `json:"size"`
	Digest     string              `json:"digest"`
	Details    ProcessModelDetails `json:"details"`
}

// TagsResponse is the response from /api/tags listing available models.
type TagsResponse struct {
	Models []TagModel `json:"models"`
}

// Tags returns the list of locally available models.
func (c *Client) Tags() (*TagsResponse, error) {
	var result TagsResponse
	if err := c.doJSON("GET", c.apiURL("tags"), nil, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// --- /api/show -------------------------------------------------------------

// ShowRequest is the request body for POST /api/show.
type ShowRequest struct {
	Name    string `json:"name"`
	Verbose *bool  `json:"verbose,omitempty"`
}

// ShowResponse is the response from POST /api/show.
type ShowResponse struct {
	Modelfile  string                 `json:"modelfile"`
	Parameters string                 `json:"parameters"`
	Template   string                 `json:"template"`
	Details    ProcessModelDetails    `json:"details"`
	ModelInfo  map[string]interface{} `json:"model_info"`
}

// Show returns detailed information about a model.
func (c *Client) Show(request ShowRequest) (*ShowResponse, error) {
	var result ShowResponse
	if err := c.doJSON("POST", c.apiURL("show"), request, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// --- /api/ps ---------------------------------------------------------------

// ProcessModelDetails holds model format metadata.
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
func (c *Client) Ps() (*ProcessStatus, error) {
	var result ProcessStatus
	if err := c.doJSON("GET", c.apiURL("ps"), nil, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// --- /api/embed ------------------------------------------------------------

// EmbedRequest is a request to the /api/embed endpoint.
type EmbedRequest struct {
	Model     string   `json:"model"`
	Input     []string `json:"input"`
	Truncate  *bool    `json:"truncate,omitempty"`
	KeepAlive *string  `json:"keep_alive,omitempty"`
}

// EmbedResponse is the response from the /api/embed endpoint.
type EmbedResponse struct {
	Model           string      `json:"model"`
	Embeddings      [][]float64 `json:"embeddings"`
	TotalDuration   int64       `json:"total_duration"`
	LoadDuration    int64       `json:"load_duration"`
	PromptEvalCount int         `json:"prompt_eval_count"`
}

// Embed generates embeddings for the given input texts.
func (c *Client) Embed(request EmbedRequest) (*EmbedResponse, error) {
	var result EmbedResponse
	if err := c.doJSON("POST", c.apiURL("embed"), request, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// --- /api/chat -------------------------------------------------------------

// ChatMessage represents a single message in a chat conversation.
type ChatMessage struct {
	Role      string         `json:"role"`
	Content   string         `json:"content"`
	Images    []RequestImage `json:"images,omitempty"`
	ToolCalls []ToolCall     `json:"tool_calls,omitempty"`
}

// ToolCall represents a tool invocation requested by the model.
type ToolCall struct {
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction describes the function name and arguments of a tool call.
type ToolCallFunction struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

// ChatRequest is the request body for POST /api/chat.
type ChatRequest struct {
	Model     string          `json:"model"`
	Messages  []ChatMessage   `json:"messages"`
	Format    *RequestFormat  `json:"format,omitempty"`
	Options   *RequestOptions `json:"options,omitempty"`
	Stream    *bool           `json:"stream,omitempty"`
	KeepAlive *string         `json:"keep_alive,omitempty"`
}

// ChatResponse is a single streamed (or non-streamed) response from POST /api/chat.
type ChatResponse struct {
	Model           string      `json:"model"`
	CreatedAt       *time.Time  `json:"created_at,omitempty"`
	Message         ChatMessage `json:"message"`
	Done            bool        `json:"done"`
	DoneReason      string      `json:"done_reason,omitempty"`
	TotalDuration   int64       `json:"total_duration,omitempty"`
	LoadDuration    int64       `json:"load_duration,omitempty"`
	PromptEvalCount int         `json:"prompt_eval_count,omitempty"`
	EvalCount       int         `json:"eval_count,omitempty"`
}

// Chat sends a chat completion request. It streams newline-delimited JSON and
// calls onResponse for each chunk. For non-streaming, set Stream to Bool(false)
// and the single response will still be delivered via onResponse.
func (c *Client) Chat(request ChatRequest, onResponse func(ChatResponse) error) error {
	return c.doStream(c.apiURL("chat"), request, func(raw json.RawMessage) error {
		var res ChatResponse
		if err := json.Unmarshal(raw, &res); err != nil {
			return fmt.Errorf("failed to unmarshal chat response: %w", err)
		}
		return onResponse(res)
	})
}

// --- /api/copy -------------------------------------------------------------

// CopyRequest is the request body for POST /api/copy.
type CopyRequest struct {
	Source      string `json:"source"`
	Destination string `json:"destination"`
}

// Copy duplicates a model under a new name.
func (c *Client) Copy(request CopyRequest) error {
	return c.doJSON("POST", c.apiURL("copy"), request, nil)
}

// --- /api/delete -----------------------------------------------------------

// DeleteRequest is the request body for DELETE /api/delete.
type DeleteRequest struct {
	Name string `json:"name"`
}

// Delete removes a model and its data.
func (c *Client) Delete(request DeleteRequest) error {
	data, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("failed to marshal delete request: %w", err)
	}
	req, err := http.NewRequest("DELETE", c.apiURL("delete"), strings.NewReader(string(data)))
	if err != nil {
		return fmt.Errorf("failed to create delete request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if c.ds.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.ds.Token)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send delete request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("delete request failed, status code: %d, body: %s", resp.StatusCode, body)
	}
	return nil
}

// --- /api/pull -------------------------------------------------------------

// PullRequest is the request body for POST /api/pull.
type PullRequest struct {
	Name     string `json:"name"`
	Insecure *bool  `json:"insecure,omitempty"`
	Stream   *bool  `json:"stream,omitempty"`
}

// PullResponse is a single streamed status from POST /api/pull.
type PullResponse struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty"`
	Completed int64  `json:"completed,omitempty"`
}

// Pull downloads a model from the Ollama registry. Progress is reported via onStatus.
func (c *Client) Pull(request PullRequest, onStatus func(PullResponse) error) error {
	return c.doStream(c.apiURL("pull"), request, func(raw json.RawMessage) error {
		var res PullResponse
		if err := json.Unmarshal(raw, &res); err != nil {
			return fmt.Errorf("failed to unmarshal pull response: %w", err)
		}
		return onStatus(res)
	})
}

// --- /api/push -------------------------------------------------------------

// PushRequest is the request body for POST /api/push.
type PushRequest struct {
	Name     string `json:"name"`
	Insecure *bool  `json:"insecure,omitempty"`
	Stream   *bool  `json:"stream,omitempty"`
}

// PushResponse is a single streamed status from POST /api/push.
type PushResponse struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty"`
	Completed int64  `json:"completed,omitempty"`
}

// Push uploads a model to the Ollama registry. Progress is reported via onStatus.
func (c *Client) Push(request PushRequest, onStatus func(PushResponse) error) error {
	return c.doStream(c.apiURL("push"), request, func(raw json.RawMessage) error {
		var res PushResponse
		if err := json.Unmarshal(raw, &res); err != nil {
			return fmt.Errorf("failed to unmarshal push response: %w", err)
		}
		return onStatus(res)
	})
}

// --- /api/create -----------------------------------------------------------

// CreateRequest is the request body for POST /api/create.
type CreateRequest struct {
	Model     string `json:"model"`
	From      string `json:"from,omitempty"`
	Modelfile string `json:"modelfile,omitempty"`
	System    string `json:"system,omitempty"`
	Stream    *bool  `json:"stream,omitempty"`
}

// CreateResponse is a single streamed status from POST /api/create.
type CreateResponse struct {
	Status string `json:"status"`
}

// Create creates a model. Progress is reported via onStatus.
func (c *Client) Create(request CreateRequest, onStatus func(CreateResponse) error) error {
	return c.doStream(c.apiURL("create"), request, func(raw json.RawMessage) error {
		var res CreateResponse
		if err := json.Unmarshal(raw, &res); err != nil {
			return fmt.Errorf("failed to unmarshal create response: %w", err)
		}
		return onStatus(res)
	})
}

// --- /api/blobs ------------------------------------------------------------

// BlobExists checks whether a blob with the given digest exists on the server.
func (c *Client) BlobExists(digest string) (bool, error) {
	url := c.apiURL("blobs/" + digest)
	req, err := http.NewRequest("HEAD", url, nil)
	if err != nil {
		return false, fmt.Errorf("failed to create blob HEAD request: %w", err)
	}
	if c.ds.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.ds.Token)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return false, fmt.Errorf("failed to send blob HEAD request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		return true, nil
	}
	if resp.StatusCode == http.StatusNotFound {
		return false, nil
	}
	return false, fmt.Errorf("blob HEAD unexpected status: %d", resp.StatusCode)
}

// BlobCreate uploads a binary blob with the given digest.
func (c *Client) BlobCreate(digest string, body io.Reader) error {
	url := c.apiURL("blobs/" + digest)
	req, err := http.NewRequest("POST", url, body)
	if err != nil {
		return fmt.Errorf("failed to create blob POST request: %w", err)
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	if c.ds.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.ds.Token)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send blob POST request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("blob POST failed, status code: %d, body: %s", resp.StatusCode, respBody)
	}
	return nil
}

// Query sends a request to the ollama API or an OpenAI-compatible chat endpoint.
func (c *Client) Query(request Request) (err error) {
	switch c.ds.resolveAPI() {
	case APICompletions:
		return c.queryCompletions(request)
	case APIOpenAI:
		return c.queryOpenAI(request)
	}
	js := request.ToJson()
	req, err := http.NewRequest("POST", c.ds.URL, strings.NewReader(js))

	if err != nil {
		return fmt.Errorf("failed to create ollama request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-type", "application/json")
	if c.ds.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.ds.Token)
	}

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
