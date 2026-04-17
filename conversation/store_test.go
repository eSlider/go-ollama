package conversation

import (
	"testing"
)

func TestFloat64ToFloat32(t *testing.T) {
	t.Parallel()

	in := []float64{0, 1, -1, 0.5, 3.1415926535}
	out := Float64ToFloat32(in)

	if len(out) != len(in) {
		t.Fatalf("len=%d, want %d", len(out), len(in))
	}
	for i, v := range in {
		if float64(out[i]) != float64(float32(v)) {
			t.Errorf("out[%d] = %v, want %v", i, out[i], float32(v))
		}
	}
}

func TestNewStore_Validation(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name string
		cfg  Config
	}{
		{"missing URL", Config{Dims: 4}},
		{"missing Dims", Config{URL: "http://localhost:9200"}},
		{"zero Dims", Config{URL: "http://localhost:9200", Dims: 0}},
	} {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if _, err := NewStore(tc.cfg); err == nil {
				t.Fatalf("expected error for %+v, got nil", tc.cfg)
			}
		})
	}
}

func TestNewStore_DefaultIndex(t *testing.T) {
	t.Parallel()

	s, err := NewStore(Config{URL: "http://localhost:9200", Dims: 4})
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	if s.Index() != DefaultIndex {
		t.Errorf("Index() = %q, want %q", s.Index(), DefaultIndex)
	}
}
