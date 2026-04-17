package ollama

import (
	"testing"
)

func TestParseCodeBlock_SingleBlock(t *testing.T) {
	blocks := ParseCodeBlock(String("Some text\n```go\nfmt.Println(\"hello\")\n```\nMore text"))

	if len(blocks) != 1 {
		t.Fatalf("got %d blocks, want 1", len(blocks))
	}
	if blocks[0].Type != "go" {
		t.Errorf("type = %q, want %q", blocks[0].Type, "go")
	}
	if blocks[0].Code != "\nfmt.Println(\"hello\")" {
		t.Errorf("code = %q", blocks[0].Code)
	}
}

func TestParseCodeBlock_MultipleBlocks(t *testing.T) {
	blocks := ParseCodeBlock(String("```python\nprint('a')\n```\nmiddle\n```bash\necho b\n```\n"))

	if len(blocks) != 2 {
		t.Fatalf("got %d blocks, want 2", len(blocks))
	}
	if blocks[0].Type != "python" {
		t.Errorf("block[0].Type = %q, want python", blocks[0].Type)
	}
	if blocks[1].Type != "bash" {
		t.Errorf("block[1].Type = %q, want bash", blocks[1].Type)
	}
}

func TestParseCodeBlock_NoBlocks(t *testing.T) {
	blocks := ParseCodeBlock(String("Just plain text with no code"))

	if len(blocks) != 0 {
		t.Errorf("got %d blocks from plain text, want 0", len(blocks))
	}
}

func TestParseCodeBlock_MultilineCode(t *testing.T) {
	blocks := ParseCodeBlock(String("```javascript\nconst x = 1;\nconst y = 2;\nconsole.log(x + y);\n```\n"))

	if len(blocks) != 1 {
		t.Fatalf("got %d blocks, want 1", len(blocks))
	}
	if blocks[0].Type != "javascript" {
		t.Errorf("type = %q, want javascript", blocks[0].Type)
	}
}

func TestParseCodeBlock_SQL(t *testing.T) {
	blocks := ParseCodeBlock(String("Run this:\n```sql\nSELECT * FROM users\nWHERE active = true;\n```\nDone."))

	if len(blocks) != 1 {
		t.Fatalf("got %d blocks, want 1", len(blocks))
	}
	if blocks[0].Type != "sql" {
		t.Errorf("type = %q, want sql", blocks[0].Type)
	}
}

func TestConvertHelpers(t *testing.T) {
	if *Int(42) != 42 {
		t.Error("Int(42) failed")
	}
	if *String("hello") != "hello" {
		t.Error("String failed")
	}
	if *Bool(true) != true {
		t.Error("Bool failed")
	}
	if *Float(3.14) != 3.14 {
		t.Error("Float failed")
	}
}

func TestRequestToJson(t *testing.T) {
	r := Request{
		Model:  "llama3.2:3b",
		Prompt: "test",
		Options: &RequestOptions{
			Temperature: Float(0.7),
		},
	}
	js := r.ToJson()

	if js == "" {
		t.Fatal("ToJson returned empty string")
	}
}

func TestRequestToJson_WithFormat(t *testing.T) {
	format := FormatJson
	r := Request{
		Model:  "m",
		Prompt: "return json",
		Format: &format,
	}
	js := r.ToJson()

	if js == "" {
		t.Fatal("ToJson returned empty string")
	}
}
