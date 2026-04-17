//go:build integration

// Integration tests for the conversation Store.
//
// These tests require BOTH a running Elasticsearch 8 cluster AND a running
// Ollama (or Open WebUI) instance reachable by the official client. They are
// intentionally gated behind the `integration` build tag so the default
// `go test ./...` run stays hermetic.
//
// Run locally:
//
//	docker compose up -d
//	ollama serve   # or point OPEN_WEB_API_GENERATE_URL at a remote instance
//	ELASTICSEARCH_URL=http://localhost:9200 \
//	  go test -tags=integration -race -v ./conversation/...
//
// No vectors are synthesised: embeddings come from Ollama's /api/embed.
package conversation_test

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"

	ollama "github.com/eslider/go-ollama"
	"github.com/eslider/go-ollama/conversation"
)

const defaultEmbedModel = "llama3.2:1b"

// testContext returns a context with a reasonable deadline for ES/Ollama RPCs.
func testContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	t.Cleanup(cancel)
	return ctx
}

// requireElasticURL skips the test when ELASTICSEARCH_URL is unset so
// developers must opt in explicitly by starting `docker compose up -d`.
func requireElasticURL(t *testing.T) string {
	t.Helper()
	url := os.Getenv("ELASTICSEARCH_URL")
	if url == "" {
		t.Skip("ELASTICSEARCH_URL is not set; run `docker compose up -d` and export it")
	}
	return url
}

// requireOllamaClient returns a live Ollama client and skips the test if the
// CI flag is set — matching the existing pattern in client_integration_test.go.
func requireOllamaClient(t *testing.T) *ollama.Client {
	t.Helper()
	if os.Getenv("CI") != "" {
		t.Skip("integration tests require live Ollama; skipped in CI")
	}
	return ollama.NewOpenWebUiClient(&ollama.DSN{
		URL:   os.Getenv("OPEN_WEB_API_GENERATE_URL"),
		Token: os.Getenv("OPEN_WEB_API_TOKEN"),
	})
}

// embedModel returns the model used for embeddings in these tests.
func embedModel() string {
	if m := os.Getenv("OLLAMA_EMBED_MODEL"); m != "" {
		return m
	}
	return defaultEmbedModel
}

// uniqueIndex builds a per-test index name so parallel tests never collide.
func uniqueIndex(t *testing.T) string {
	t.Helper()
	//nolint:gosec // unique-per-run suffix; cryptographic quality not required.
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	return fmt.Sprintf("ollama-conv-test-%d-%d", time.Now().UnixNano(), r.Intn(1_000_000))
}

// newStoreWithDims embeds a probe string via Ollama to discover the real
// embedding dimension, then builds a Store configured for that dim.
// This keeps the mapping aligned with whichever model the operator chose,
// without hard-coding dimensions that drift between Ollama versions.
func newStoreWithDims(t *testing.T, esURL string, ol *ollama.Client) (*conversation.Store, int, func()) {
	t.Helper()

	probe, err := ol.Embed(ollama.EmbedRequest{
		Model: embedModel(),
		Input: []string{"dimension probe"},
	})
	if err != nil {
		t.Skipf("Ollama /api/embed unavailable (model=%s): %v", embedModel(), err)
	}
	if len(probe.Embeddings) == 0 || len(probe.Embeddings[0]) == 0 {
		t.Fatal("empty embedding from Ollama")
	}
	dims := len(probe.Embeddings[0])
	t.Logf("embed model=%s dims=%d", embedModel(), dims)

	idx := uniqueIndex(t)
	store, err := conversation.NewStore(conversation.Config{
		URL:   esURL,
		Index: idx,
		Dims:  dims,
	})
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}

	ctx := testContext(t)
	if err := store.Ping(ctx); err != nil {
		t.Skipf("Elasticsearch not reachable at %s: %v", esURL, err)
	}
	if err := store.EnsureIndex(ctx); err != nil {
		t.Fatalf("EnsureIndex: %v", err)
	}

	cleanup := func() {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if err := store.DeleteIndex(ctx); err != nil {
			t.Logf("cleanup DeleteIndex: %v", err)
		}
	}
	return store, dims, cleanup
}

// embedOne is a tiny helper that embeds a single text and returns the vector
// already converted to []float32 ready for the Elasticsearch dense_vector.
func embedOne(t *testing.T, ol *ollama.Client, text string) []float32 {
	t.Helper()
	resp, err := ol.Embed(ollama.EmbedRequest{
		Model: embedModel(),
		Input: []string{text},
	})
	if err != nil {
		t.Fatalf("Embed(%q): %v", text, err)
	}
	if len(resp.Embeddings) != 1 {
		t.Fatalf("Embed(%q): got %d vectors, want 1", text, len(resp.Embeddings))
	}
	return conversation.Float64ToFloat32(resp.Embeddings[0])
}

// ---------- tests ----------

func TestIntegration_EnsureIndex_Idempotent(t *testing.T) {
	esURL := requireElasticURL(t)
	ol := requireOllamaClient(t)

	store, _, cleanup := newStoreWithDims(t, esURL, ol)
	defer cleanup()

	ctx := testContext(t)
	if err := store.EnsureIndex(ctx); err != nil {
		t.Fatalf("second EnsureIndex should be a no-op, got: %v", err)
	}
}

func TestIntegration_SaveTurn_RoundTripsTokens(t *testing.T) {
	esURL := requireElasticURL(t)
	ol := requireOllamaClient(t)

	store, _, cleanup := newStoreWithDims(t, esURL, ol)
	defer cleanup()

	ctx := testContext(t)
	turn := conversation.Turn{
		ID:              "turn-1",
		SessionID:       "session-1",
		Sequence:        1,
		Role:            conversation.RoleUser,
		Text:            "What is the capital of France?",
		Model:           "gemma3:1b",
		PromptEvalCount: 42,
		EvalCount:       17,
		TotalDurationNs: 123_456_789,
		Embedding:       embedOne(t, ol, "What is the capital of France?"),
	}
	if err := store.SaveTurn(ctx, turn); err != nil {
		t.Fatalf("SaveTurn: %v", err)
	}

	// Verify the document is searchable via k-NN and the stored fields round
	// trip (including the Ollama token counters).
	hits, err := store.Similar(ctx, turn.Embedding, 1, "session-1")
	if err != nil {
		t.Fatalf("Similar: %v", err)
	}
	if len(hits) != 1 {
		t.Fatalf("got %d hits, want 1", len(hits))
	}
	got := hits[0].Turn
	if got.ID != turn.ID ||
		got.SessionID != turn.SessionID ||
		got.Sequence != turn.Sequence ||
		got.Role != turn.Role ||
		got.Text != turn.Text ||
		got.Model != turn.Model ||
		got.PromptEvalCount != turn.PromptEvalCount ||
		got.EvalCount != turn.EvalCount ||
		got.TotalDurationNs != turn.TotalDurationNs {
		t.Fatalf("round-trip mismatch:\n got=%+v\nwant=%+v", got, turn)
	}
}

func TestIntegration_Similar_RanksClosestFirst(t *testing.T) {
	esURL := requireElasticURL(t)
	ol := requireOllamaClient(t)

	store, _, cleanup := newStoreWithDims(t, esURL, ol)
	defer cleanup()

	ctx := testContext(t)

	corpus := []struct {
		id   string
		text string
	}{
		{"doc-cat", "The cat sat on the mat"},
		{"doc-dog", "The dog lay on the rug"},
		{"doc-qc", "Quantum computing uses qubits to perform calculations"},
	}
	for i, c := range corpus {
		turn := conversation.Turn{
			ID:        c.id,
			SessionID: "rank-session",
			Sequence:  i,
			Role:      conversation.RoleUser,
			Text:      c.text,
			Embedding: embedOne(t, ol, c.text),
		}
		if err := store.SaveTurn(ctx, turn); err != nil {
			t.Fatalf("SaveTurn %s: %v", c.id, err)
		}
	}

	query := embedOne(t, ol, "a feline resting on a carpet")
	hits, err := store.Similar(ctx, query, 3, "")
	if err != nil {
		t.Fatalf("Similar: %v", err)
	}
	if len(hits) != len(corpus) {
		t.Fatalf("got %d hits, want %d", len(hits), len(corpus))
	}

	var ids []string
	for _, h := range hits {
		ids = append(ids, h.Turn.ID)
		t.Logf("hit id=%s score=%.4f text=%q", h.Turn.ID, h.Score, h.Turn.Text)
	}

	// The cat sentence must outrank the quantum computing sentence; exact
	// ordering of cat vs dog depends on the embedding model so only the
	// "closer-than-irrelevant" invariant is asserted.
	catIdx, qcIdx := indexOf(ids, "doc-cat"), indexOf(ids, "doc-qc")
	if catIdx == -1 || qcIdx == -1 {
		t.Fatalf("missing expected docs; got ids=%v", ids)
	}
	if catIdx >= qcIdx {
		t.Fatalf("cat (%d) should rank above quantum (%d); ids=%v", catIdx, qcIdx, ids)
	}

	// Descending score sanity: ES must return hits sorted by _score desc.
	for i := 1; i < len(hits); i++ {
		if hits[i].Score > hits[i-1].Score {
			t.Fatalf("hits not sorted by score desc: %v", hits)
		}
	}
}

func TestIntegration_Similar_SessionFilter(t *testing.T) {
	esURL := requireElasticURL(t)
	ol := requireOllamaClient(t)

	store, _, cleanup := newStoreWithDims(t, esURL, ol)
	defer cleanup()

	ctx := testContext(t)

	// Two sessions, same text — the filter must return only the requested one.
	text := "A short note about the Go programming language"
	vec := embedOne(t, ol, text)

	for _, sess := range []string{"alice", "bob"} {
		if err := store.SaveTurn(ctx, conversation.Turn{
			ID:        "note-" + sess,
			SessionID: sess,
			Role:      conversation.RoleUser,
			Text:      text,
			Embedding: vec,
		}); err != nil {
			t.Fatalf("SaveTurn %s: %v", sess, err)
		}
	}

	hits, err := store.Similar(ctx, vec, 10, "alice")
	if err != nil {
		t.Fatalf("Similar: %v", err)
	}
	if len(hits) != 1 {
		t.Fatalf("got %d hits, want 1 (session filter)", len(hits))
	}
	if hits[0].Turn.SessionID != "alice" {
		t.Fatalf("session=%q, want alice", hits[0].Turn.SessionID)
	}
	if !strings.Contains(hits[0].Turn.Text, "Go programming") {
		t.Fatalf("unexpected text: %q", hits[0].Turn.Text)
	}
}

// ---------- helpers ----------

func indexOf(s []string, v string) int {
	for i, x := range s {
		if x == v {
			return i
		}
	}
	return -1
}
