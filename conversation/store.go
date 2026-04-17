// Package conversation provides an Elasticsearch-backed store for chat
// conversation turns generated via the Ollama / Open WebUI client.
//
// Each stored document represents a single turn in a session and carries
// token-accounting fields returned by Ollama (prompt_eval_count, eval_count,
// total_duration) plus a dense_vector embedding used for k-NN similarity
// search.
//
// The package intentionally keeps the mapping small and explicit so the
// storage and query shape can be reviewed / discussed directly — see the
// index mapping constant in this file.
package conversation

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/elastic/go-elasticsearch/v8"
	"github.com/elastic/go-elasticsearch/v8/esapi"
)

// Role identifies who produced a turn.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system"
)

// DefaultIndex is the default Elasticsearch index name for conversation turns.
const DefaultIndex = "ollama-conversations"

// Turn is a single chat turn persisted to Elasticsearch.
//
// The JSON tags define the _source shape; the index mapping in
// indexMappingTemplate must stay aligned with these fields.
type Turn struct {
	ID              string    `json:"id"`
	SessionID       string    `json:"session_id"`
	Sequence        int       `json:"sequence"`
	Role            Role      `json:"role"`
	Text            string    `json:"text"`
	Model           string    `json:"model,omitempty"`
	PromptEvalCount int       `json:"prompt_eval_count,omitempty"`
	EvalCount       int       `json:"eval_count,omitempty"`
	TotalDurationNs int64     `json:"total_duration_ns,omitempty"`
	CreatedAt       time.Time `json:"created_at"`
	Embedding       []float32 `json:"embedding,omitempty"`
}

// Hit is a single similarity search result.
type Hit struct {
	Turn  Turn
	Score float64
}

// Config holds connection and index parameters for a Store.
type Config struct {
	// URL of the Elasticsearch cluster (e.g. http://localhost:9200).
	URL string
	// Index name for conversation turns. Defaults to DefaultIndex when empty.
	Index string
	// Dims is the embedding vector dimension. Must match the embedding model
	// used by the caller (e.g. Ollama /api/embed output length).
	Dims int
	// APIKey for Elastic Cloud / secured clusters. Optional.
	APIKey string
	// Username / Password for basic auth. Optional.
	Username string
	Password string
}

// Store persists and queries conversation turns in Elasticsearch.
type Store struct {
	es    *elasticsearch.Client
	index string
	dims  int
}

// NewStore builds a Store using the official go-elasticsearch v8 client.
func NewStore(cfg Config) (*Store, error) {
	if cfg.URL == "" {
		return nil, errors.New("conversation: Config.URL is required")
	}
	if cfg.Dims <= 0 {
		return nil, errors.New("conversation: Config.Dims must be > 0 (embedding dimension)")
	}

	esCfg := elasticsearch.Config{
		Addresses: []string{cfg.URL},
		APIKey:    cfg.APIKey,
		Username:  cfg.Username,
		Password:  cfg.Password,
	}
	es, err := elasticsearch.NewClient(esCfg)
	if err != nil {
		return nil, fmt.Errorf("conversation: new es client: %w", err)
	}

	index := cfg.Index
	if index == "" {
		index = DefaultIndex
	}
	return &Store{es: es, index: index, dims: cfg.Dims}, nil
}

// Index returns the configured index name.
func (s *Store) Index() string { return s.index }

// Ping verifies connectivity to the cluster.
func (s *Store) Ping(ctx context.Context) error {
	res, err := s.es.Info(s.es.Info.WithContext(ctx))
	if err != nil {
		return fmt.Errorf("conversation: ping: %w", err)
	}
	defer res.Body.Close()
	if res.IsError() {
		return fmt.Errorf("conversation: ping status %s", res.Status())
	}
	return nil
}

// indexMappingTemplate is the JSON mapping used when creating the index.
// The %d placeholder is replaced by Config.Dims at creation time.
//
// Rationale:
//   - keyword fields for id/session/role/model → exact-match filters.
//   - text for the human content → full-text scoring if needed.
//   - date for created_at → range queries, ILM alignment if added later.
//   - dense_vector with index: true + cosine similarity → kNN search.
const indexMappingTemplate = `{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
  "mappings": {
    "properties": {
      "id":                 {"type": "keyword"},
      "session_id":         {"type": "keyword"},
      "sequence":           {"type": "integer"},
      "role":               {"type": "keyword"},
      "text":               {"type": "text"},
      "model":              {"type": "keyword"},
      "prompt_eval_count":  {"type": "integer"},
      "eval_count":         {"type": "integer"},
      "total_duration_ns":  {"type": "long"},
      "created_at":         {"type": "date"},
      "embedding": {
        "type": "dense_vector",
        "dims": %d,
        "index": true,
        "similarity": "cosine"
      }
    }
  }
}`

// EnsureIndex creates the index with the configured mapping if it does not
// exist. It is safe to call on every startup (idempotent).
func (s *Store) EnsureIndex(ctx context.Context) error {
	existsRes, err := s.es.Indices.Exists(
		[]string{s.index},
		s.es.Indices.Exists.WithContext(ctx),
	)
	if err != nil {
		return fmt.Errorf("conversation: index exists: %w", err)
	}
	defer existsRes.Body.Close()

	switch existsRes.StatusCode {
	case http.StatusOK:
		return nil
	case http.StatusNotFound:
		// Fall through to create.
	default:
		return fmt.Errorf("conversation: unexpected exists status %s", existsRes.Status())
	}

	body := fmt.Sprintf(indexMappingTemplate, s.dims)
	createRes, err := s.es.Indices.Create(
		s.index,
		s.es.Indices.Create.WithContext(ctx),
		s.es.Indices.Create.WithBody(bytes.NewReader([]byte(body))),
	)
	if err != nil {
		return fmt.Errorf("conversation: index create: %w", err)
	}
	defer createRes.Body.Close()
	if createRes.IsError() {
		raw, _ := io.ReadAll(createRes.Body)
		return fmt.Errorf("conversation: index create %s: %s", createRes.Status(), string(raw))
	}
	return nil
}

// SaveTurn indexes a single turn. The document id is Turn.ID; when empty,
// Elasticsearch auto-generates one and the caller can ignore it.
// A refresh=wait_for is used so the document is searchable immediately after
// this call returns — intentional for an interactive TUI; callers writing
// at higher throughput should prefer the bulk API.
func (s *Store) SaveTurn(ctx context.Context, turn Turn) error {
	if turn.CreatedAt.IsZero() {
		turn.CreatedAt = time.Now().UTC()
	}
	if len(turn.Embedding) != 0 && len(turn.Embedding) != s.dims {
		return fmt.Errorf("conversation: embedding dim=%d, store dims=%d", len(turn.Embedding), s.dims)
	}

	data, err := json.Marshal(turn)
	if err != nil {
		return fmt.Errorf("conversation: marshal turn: %w", err)
	}

	req := esapi.IndexRequest{
		Index:      s.index,
		DocumentID: turn.ID,
		Body:       bytes.NewReader(data),
		Refresh:    "wait_for",
	}
	res, err := req.Do(ctx, s.es)
	if err != nil {
		return fmt.Errorf("conversation: index turn: %w", err)
	}
	defer res.Body.Close()
	if res.IsError() {
		raw, _ := io.ReadAll(res.Body)
		return fmt.Errorf("conversation: index turn %s: %s", res.Status(), string(raw))
	}
	return nil
}

// similarRequest is the top-level shape Elasticsearch accepts for a kNN
// search. Kept as an unexported struct so the query stays one place.
type similarRequest struct {
	Knn    knnClause `json:"knn"`
	Size   int       `json:"size"`
	Source any       `json:"_source,omitempty"`
}

type knnClause struct {
	Field         string    `json:"field"`
	QueryVector   []float32 `json:"query_vector"`
	K             int       `json:"k"`
	NumCandidates int       `json:"num_candidates"`
	Filter        any       `json:"filter,omitempty"`
}

// Similar runs a k-NN search against the embedding field and returns the
// top-k hits ordered by score (highest first). An optional sessionID filter
// restricts the search to a single conversation.
func (s *Store) Similar(ctx context.Context, vector []float32, k int, sessionID string) ([]Hit, error) {
	if len(vector) != s.dims {
		return nil, fmt.Errorf("conversation: query vector dim=%d, store dims=%d", len(vector), s.dims)
	}
	if k <= 0 {
		k = 5
	}

	clause := knnClause{
		Field:         "embedding",
		QueryVector:   vector,
		K:             k,
		NumCandidates: maxInt(k*10, 50),
	}
	if sessionID != "" {
		clause.Filter = map[string]any{
			"term": map[string]any{"session_id": sessionID},
		}
	}

	body, err := json.Marshal(similarRequest{Knn: clause, Size: k})
	if err != nil {
		return nil, fmt.Errorf("conversation: marshal knn: %w", err)
	}

	res, err := s.es.Search(
		s.es.Search.WithContext(ctx),
		s.es.Search.WithIndex(s.index),
		s.es.Search.WithBody(bytes.NewReader(body)),
	)
	if err != nil {
		return nil, fmt.Errorf("conversation: search: %w", err)
	}
	defer res.Body.Close()
	if res.IsError() {
		raw, _ := io.ReadAll(res.Body)
		return nil, fmt.Errorf("conversation: search %s: %s", res.Status(), string(raw))
	}

	var parsed struct {
		Hits struct {
			Hits []struct {
				Score  float64 `json:"_score"`
				Source Turn    `json:"_source"`
			} `json:"hits"`
		} `json:"hits"`
	}
	if err := json.NewDecoder(res.Body).Decode(&parsed); err != nil {
		return nil, fmt.Errorf("conversation: decode search: %w", err)
	}

	out := make([]Hit, 0, len(parsed.Hits.Hits))
	for _, h := range parsed.Hits.Hits {
		out = append(out, Hit{Turn: h.Source, Score: h.Score})
	}
	return out, nil
}

// DeleteIndex removes the index. Intended for test teardown.
func (s *Store) DeleteIndex(ctx context.Context) error {
	res, err := s.es.Indices.Delete(
		[]string{s.index},
		s.es.Indices.Delete.WithContext(ctx),
		s.es.Indices.Delete.WithIgnoreUnavailable(true),
	)
	if err != nil {
		return fmt.Errorf("conversation: index delete: %w", err)
	}
	defer res.Body.Close()
	if res.IsError() && res.StatusCode != http.StatusNotFound {
		raw, _ := io.ReadAll(res.Body)
		return fmt.Errorf("conversation: index delete %s: %s", res.Status(), string(raw))
	}
	return nil
}

// Float64ToFloat32 converts Ollama's [][]float64 embedding output into the
// []float32 slice Elasticsearch dense_vector expects.
func Float64ToFloat32(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
