package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDSN_resolveAPI_completions(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		dsn  DSN
		want APIBackend
	}{
		{"explicit", DSN{API: APICompletions, URL: "http://x/v1/chat/completions"}, APICompletions},
		{"url path", DSN{URL: "http://127.0.0.1:8102/v1/completions"}, APICompletions},
		{"chat still openai", DSN{URL: "http://h/v1/chat/completions"}, APIOpenAI},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := tc.dsn.resolveAPI(); got != tc.want {
				t.Fatalf("resolveAPI = %q want %q", got, tc.want)
			}
		})
	}
}

func TestCompletionsURL(t *testing.T) {
	t.Parallel()
	cases := []struct {
		in, want string
	}{
		{"", DefaultOpenAICompletionsURL},
		{"http://h:8102/v1", "http://h:8102/v1/completions"},
		{"http://h:8102/v1/", "http://h:8102/v1/completions"},
		{"http://h/v1/completions", "http://h/v1/completions"},
		{"http://h/v1/chat/completions", "http://h/v1/completions"},
	}
	for _, tc := range cases {
		if got := completionsURL(tc.in); got != tc.want {
			t.Fatalf("completionsURL(%q)=%q want %q", tc.in, got, tc.want)
		}
	}
}

func TestQueryCompletions_NonStream(t *testing.T) {
	t.Parallel()
	var got openAICompletionsRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/completions" {
			t.Errorf("path=%s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decode: %v", err)
		}
		fmt.Fprint(w, `{"choices":[{"text":" 06:00 Uhr.","index":0,"finish_reason":"stop"}],"model":"gemma"}`)
	}))
	defer srv.Close()

	client := NewOpenWebUiClient(&DSN{URL: srv.URL + "/v1/completions"})
	defer client.Close()
	var assembled strings.Builder
	err := client.Query(Request{
		Prompt: "Die heutige Schicht beginnt um",
		Stream: Bool(false),
		Options: &RequestOptions{
			Temperature: Float(0.7),
			Stop:        []string{"\n"},
			NumPredict:  Int(32),
		},
		OnJson: func(res Response) error {
			if res.Response != nil {
				assembled.WriteString(*res.Response)
			}
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.Prompt != "Die heutige Schicht beginnt um" || got.Stream {
		t.Fatalf("body=%+v", got)
	}
	if len(got.Stop) != 1 || got.Stop[0] != "\n" {
		t.Fatalf("stop=%v", got.Stop)
	}
	if assembled.String() != " 06:00 Uhr." {
		t.Fatalf("text=%q", assembled.String())
	}
}

func TestQueryCompletions_DialContextToRemoteLocalhost(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"choices":[{"text":"ok","finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	dialed := ""
	client := NewOpenWebUiClient(&DSN{
		URL: "http://127.0.0.1:8102/v1/completions",
		API: APICompletions,
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			dialed = addr
			return net.Dial(network, srv.Listener.Addr().String())
		},
	})
	defer client.Close()

	err := client.Query(Request{
		Prompt: "hi",
		Stream: Bool(false),
		OnJson: func(Response) error { return nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	if dialed != "127.0.0.1:8102" {
		t.Fatalf("dialed=%q want 127.0.0.1:8102 (remote localhost via SSH)", dialed)
	}
}

func TestParseOpenSSHHost_OverlayStar(t *testing.T) {
	t.Parallel()
	cfg := `
Host *
  User ano
  Port 22
Host naj-mdx-1
  User devops
  HostName medex.freeddns.org
  Port 22334
  IdentityFile ~/.ssh/id_ed25519
`
	h := parseOpenSSHHost(cfg, "naj-mdx-1")
	if h.User != "devops" || h.HostName != "medex.freeddns.org" || h.Port != "22334" {
		t.Fatalf("%+v", h)
	}
	if len(h.IdentityFiles) != 1 {
		t.Fatalf("identity=%v", h.IdentityFiles)
	}
	miss := parseOpenSSHHost(cfg, "unknown-host")
	if miss.User != "ano" || miss.Port != "22" || miss.HostName != "" {
		t.Fatalf("star-only %+v", miss)
	}
}
