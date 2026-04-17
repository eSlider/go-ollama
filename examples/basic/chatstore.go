package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

const chatsDirName = "chats"

var conversFileRe = regexp.MustCompile(`^convers-(\d+)\.yml$`)

// Message is one turn in a multi-turn chat (role: user | assistant).
type Message struct {
	Role    string `yaml:"role"`
	Content string `yaml:"content"`
}

// Conversation is persisted as chats/convers-<id>.yml
type Conversation struct {
	ID       int       `yaml:"id"`
	Title    string    `yaml:"title"`
	Model    string    `yaml:"model"`
	Messages []Message `yaml:"messages,omitempty"`
	// Legacy single-turn fields (migrated into Messages on load)
	Prompt   string    `yaml:"prompt,omitempty"`
	Response string    `yaml:"response,omitempty"`
	Updated  time.Time `yaml:"updated"`
}

func chatsDir() string {
	return chatsDirName
}

func ensureChatsDir() error {
	return os.MkdirAll(chatsDir(), 0o755)
}

func conversationPath(id int) string {
	return filepath.Join(chatsDir(), "convers-"+strconv.Itoa(id)+".yml")
}

func loadConversation(path string) (*Conversation, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c Conversation
	if err := yaml.Unmarshal(data, &c); err != nil {
		return nil, err
	}
	c.MigrateLegacy()
	return &c, nil
}

func (c *Conversation) MigrateLegacy() {
	if len(c.Messages) > 0 {
		return
	}
	if strings.TrimSpace(c.Prompt) != "" {
		c.Messages = append(c.Messages, Message{Role: "user", Content: c.Prompt})
	}
	if strings.TrimSpace(c.Response) != "" {
		c.Messages = append(c.Messages, Message{Role: "assistant", Content: c.Response})
	}
}

func (c *Conversation) Save(path string) error {
	if err := ensureChatsDir(); err != nil {
		return err
	}
	c.Updated = time.Now().UTC()
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := yaml.NewEncoder(f)
	enc.SetIndent(2)
	if err := enc.Encode(c); err != nil {
		return err
	}
	if err := enc.Close(); err != nil {
		return err
	}
	return nil
}

// deriveTitle picks a short label for the list from the prompt.
func deriveTitle(prompt string) string {
	s := strings.TrimSpace(prompt)
	if s == "" {
		return "Untitled"
	}
	line := strings.Split(s, "\n")[0]
	line = strings.TrimSpace(line)
	if len(line) > 48 {
		return line[:45] + "…"
	}
	return line
}

type chatListEntry struct {
	Path string
	ID   int
}

func scanConversations() ([]chatListEntry, error) {
	if err := ensureChatsDir(); err != nil {
		return nil, err
	}
	matches, err := filepath.Glob(filepath.Join(chatsDir(), "convers-*.yml"))
	if err != nil {
		return nil, err
	}
	var out []chatListEntry
	for _, m := range matches {
		base := filepath.Base(m)
		sm := conversFileRe.FindStringSubmatch(base)
		if len(sm) < 2 {
			continue
		}
		id, err := strconv.Atoi(sm[1])
		if err != nil {
			continue
		}
		out = append(out, chatListEntry{Path: m, ID: id})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

func nextConversationID() (int, error) {
	entries, err := scanConversations()
	if err != nil {
		return 0, err
	}
	max := 0
	for _, e := range entries {
		if e.ID > max {
			max = e.ID
		}
	}
	return max + 1, nil
}

func deriveTitleFromMessages(msgs []Message) string {
	for _, m := range msgs {
		if m.Role == "user" && strings.TrimSpace(m.Content) != "" {
			return deriveTitle(m.Content)
		}
	}
	return "New chat"
}

func listTitle(path string, id int) string {
	c, err := loadConversation(path)
	if err != nil {
		return fmt.Sprintf("convers-%d", id)
	}
	if strings.TrimSpace(c.Title) != "" && c.Title != "New chat" {
		return c.Title
	}
	t := deriveTitleFromMessages(c.Messages)
	if t != "Untitled" && t != "New chat" {
		return t
	}
	t2 := deriveTitle(c.Prompt)
	if t2 != "Untitled" {
		return t2
	}
	return fmt.Sprintf("Chat %d", id)
}
