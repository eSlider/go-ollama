// Package main provides a TUI chat client for Ollama / Open WebUI.
//
// By default it talks to local Ollama (http://localhost:11434). To use Open WebUI or another host:
//
//	export OPEN_WEB_API_GENERATE_URL="https://ai.example.com/ollama/api/generate"
//	export OPEN_WEB_API_TOKEN="sk-..."
//	go run ./examples/tui/
package main

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"

	ollama "github.com/eslider/go-ollama"
)

var fallbackModels = []string{
	"gemma3:1b",
	"llama3.2:1b",
	"deepseek-r1:8b",
}

// --- Styles ----------------------------------------------------------------


var (
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#FAFAFA")).
			Background(lipgloss.Color("#7D56F4")).
			Padding(0, 1)

	selectedModelStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#7D56F4")).
				Bold(true)

	modelItemStyle = lipgloss.NewStyle().
			PaddingLeft(2)

	modelCursorStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#7D56F4")).
				Bold(true)

	userStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#25A065")).
			Bold(true)

	assistantStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#7D56F4")).
			Bold(true)

	systemStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#D4A017")).
			Bold(true)

	dimStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#626262"))

	errorStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FF0000")).
			Bold(true)

	statusBarStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FFFDF5")).
			Background(lipgloss.Color("#353533")).
			Padding(0, 1)

	statsStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#A0D0FF"))
)

// --- Tea messages ----------------------------------------------------------

type (
	tokenMsg string
	doneMsg  struct {
		promptEvalCount int
		evalCount       int
	}
)

type (
	errMsg struct{ err error }
	psMsg  struct {
		models map[string]ollama.ProcessModel
	}
)
type tagsMsg struct{ models []string }

// --- Screen state ----------------------------------------------------------

type screen int

const (
	screenModelSelect screen = iota
	screenSystemPrompt
	screenChat
)

type chatEntry struct {
	role string // "user" or "assistant"
	text string
}

// --- Model -----------------------------------------------------------------

type model struct {
	screen screen
	width  int
	height int

	// Model picker
	models        []string
	cursor        int
	runningModels map[string]ollama.ProcessModel

	// System prompt
	systemPrompt string
	sysTA        textarea.Model

	// Chat
	selectedModel string
	history       []chatEntry
	streaming     bool
	streamBuf     *strings.Builder
	textarea      textarea.Model
	viewport      viewport.Model
	vpReady       bool
	err           error

	// Token stats
	tokenCount  int
	streamStart time.Time
	lastTokSec  float64 // final tok/s preserved after stream ends

	// Context window tracking
	ctxSize int // total context window size (from /api/ps)
	ctxUsed int // tokens used in context (prompt + eval from last response)

	// Markdown renderer
	mdRenderer *glamour.TermRenderer

	// Reference to the running program for sending messages from goroutines.
	prog **tea.Program

	client *ollama.Client
}

func initialModel(client *ollama.Client, prog **tea.Program) model {
	ta := textarea.New()
	ta.Placeholder = "Ask something... (enter to send)"
	ta.CharLimit = 4096
	ta.SetHeight(3)
	ta.ShowLineNumbers = false
	ta.FocusedStyle.CursorLine = lipgloss.NewStyle()

	sysTA := textarea.New()
	sysTA.Placeholder = "Optional system prompt..."
	sysTA.CharLimit = 4096
	sysTA.SetHeight(6)
	sysTA.ShowLineNumbers = false
	sysTA.FocusedStyle.CursorLine = lipgloss.NewStyle()

	renderer, _ := glamour.NewTermRenderer(
		glamour.WithAutoStyle(),
		glamour.WithWordWrap(80),
	)

	return model{
		screen:     screenModelSelect,
		models:     fallbackModels,
		client:     client,
		textarea:   ta,
		sysTA:      sysTA,
		streamBuf:  &strings.Builder{},
		mdRenderer: renderer,
		prog:       prog,
	}
}

func (m model) Init() tea.Cmd {
	return tea.Batch(m.fetchTags(), m.fetchPs())
}

func (m model) fetchTags() tea.Cmd {
	return func() tea.Msg {
		tags, err := m.client.Tags()
		if err != nil || len(tags.Models) == 0 {
			return tagsMsg{models: nil}
		}
		names := make([]string, len(tags.Models))
		for i, t := range tags.Models {
			names[i] = t.Name
		}
		return tagsMsg{models: names}
	}
}

func (m model) fetchPs() tea.Cmd {
	return func() tea.Msg {
		status, err := m.client.Ps()
		if err != nil {
			return psMsg{models: nil}
		}
		rm := make(map[string]ollama.ProcessModel, len(status.Models))
		for _, pm := range status.Models {
			rm[pm.Name] = pm
		}
		return psMsg{models: rm}
	}
}

// --- Update ----------------------------------------------------------------

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		if m.screen == screenChat {
			m.resizeChat()
		}
		return m, nil

	case tagsMsg:
		if len(msg.models) > 0 {
			m.models = msg.models
			if m.cursor >= len(m.models) {
				m.cursor = 0
			}
		}
		return m, nil

	case psMsg:
		m.runningModels = msg.models
		if m.selectedModel != "" {
			if pm, ok := msg.models[m.selectedModel]; ok {
				m.ctxSize = pm.ContextLength
			}
		}
		return m, nil

	case tea.KeyMsg:
		if msg.String() == "ctrl+c" {
			return m, tea.Quit
		}
	}

	switch m.screen {
	case screenModelSelect:
		return m.updateModelSelect(msg)
	case screenSystemPrompt:
		return m.updateSystemPrompt(msg)
	case screenChat:
		return m.updateChat(msg)
	}
	return m, nil
}

// --- Model select screen ---------------------------------------------------

func (m model) updateModelSelect(msg tea.Msg) (tea.Model, tea.Cmd) {
	if key, ok := msg.(tea.KeyMsg); ok {
		switch key.String() {
		case "q", "esc":
			return m, tea.Quit
		case "up", "k":
			if m.cursor > 0 {
				m.cursor--
			}
		case "down", "j":
			if m.cursor < len(m.models)-1 {
				m.cursor++
			}
		case "s":
			m.screen = screenSystemPrompt
			m.sysTA.SetValue(m.systemPrompt)
			cmd := m.sysTA.Focus()
			return m, cmd
		case "enter":
			m.selectedModel = m.models[m.cursor]
			m.screen = screenChat
			if pm, ok := m.runningModels[m.selectedModel]; ok {
				m.ctxSize = pm.ContextLength
			}
			m.resizeChat()
			focusCmd := m.textarea.Focus()
			psCmd := m.fetchPs()
			return m, tea.Batch(focusCmd, psCmd)
		}
	}
	return m, nil
}

// --- System prompt screen --------------------------------------------------

func (m model) updateSystemPrompt(msg tea.Msg) (tea.Model, tea.Cmd) {
	if key, ok := msg.(tea.KeyMsg); ok {
		switch key.String() {
		case "esc":
			m.systemPrompt = strings.TrimSpace(m.sysTA.Value())
			m.sysTA.Blur()
			m.screen = screenModelSelect
			return m, nil
		}
	}

	var cmd tea.Cmd
	m.sysTA, cmd = m.sysTA.Update(msg)
	return m, cmd
}

// --- Chat screen -----------------------------------------------------------

func (m model) innerWidth() int {
	if m.width < 20 {
		return 20
	}
	return m.width
}

func (m *model) resizeChat() {
	iw := m.innerWidth()

	const titleH = 1
	const gapH = 1
	const statusH = 1
	taH := m.textarea.Height() + 2

	vpH := m.height - titleH - gapH - taH - statusH
	if vpH < 3 {
		vpH = 3
	}

	if !m.vpReady {
		m.viewport = viewport.New(iw, vpH)
		m.viewport.SetContent(dimStyle.Render("Start a conversation below."))
		m.vpReady = true
	} else {
		m.viewport.Width = iw
		m.viewport.Height = vpH
	}
	m.textarea.SetWidth(iw)

	mdWidth := iw - 6
	if mdWidth < 20 {
		mdWidth = 20
	}
	m.mdRenderer, _ = glamour.NewTermRenderer(
		glamour.WithAutoStyle(),
		glamour.WithWordWrap(mdWidth),
	)
}

func (m model) updateChat(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "esc":
			if m.streaming {
				return m, nil
			}
			m.textarea.Blur()
			m.screen = screenModelSelect
			return m, m.fetchPs()
		case "ctrl+m":
			if m.streaming {
				return m, nil
			}
			m.textarea.Blur()
			m.screen = screenModelSelect
			return m, m.fetchPs()
		case "enter":
			if m.streaming {
				return m, nil
			}
			prompt := strings.TrimSpace(m.textarea.Value())
			if prompt == "" {
				return m, nil
			}
			m.textarea.Reset()
			m.err = nil
			m.history = append(m.history, chatEntry{role: "user", text: prompt})
			m.history = append(m.history, chatEntry{role: "assistant", text: ""})
			m.streaming = true
			m.tokenCount = 0
			m.lastTokSec = 0
			m.streamStart = time.Now()
			m.streamBuf.Reset()
			m.refreshViewport()

			go m.runQuery()
			return m, nil
		}

	case tokenMsg:
		if m.streaming && len(m.history) > 0 {
			m.tokenCount++
			m.streamBuf.WriteString(string(msg))
			m.history[len(m.history)-1].text = m.streamBuf.String()
			m.refreshViewport()
		}
		return m, nil

	case doneMsg:
		elapsed := time.Since(m.streamStart).Seconds()
		if elapsed > 0 {
			m.lastTokSec = float64(m.tokenCount) / elapsed
		}
		m.ctxUsed = msg.promptEvalCount + msg.evalCount
		m.streaming = false
		m.refreshViewport()
		focusCmd := m.textarea.Focus()
		psCmd := m.fetchPs()
		return m, tea.Batch(focusCmd, psCmd)

	case errMsg:
		elapsed := time.Since(m.streamStart).Seconds()
		if elapsed > 0 {
			m.lastTokSec = float64(m.tokenCount) / elapsed
		}
		m.streaming = false
		m.err = msg.err
		if len(m.history) > 0 {
			last := &m.history[len(m.history)-1]
			if last.role == "assistant" && last.text == "" {
				m.history = m.history[:len(m.history)-1]
			}
		}
		m.refreshViewport()
		cmd := m.textarea.Focus()
		return m, cmd
	}

	if !m.streaming {
		var taCmd tea.Cmd
		m.textarea, taCmd = m.textarea.Update(msg)
		cmds = append(cmds, taCmd)
	}

	var vpCmd tea.Cmd
	m.viewport, vpCmd = m.viewport.Update(msg)
	cmds = append(cmds, vpCmd)

	return m, tea.Batch(cmds...)
}

// buildChatMessages converts conversation history into ChatMessage slice
// with a system message prepended.
func (m model) buildChatMessages() []ollama.ChatMessage {
	sysParts := []string{
		fmt.Sprintf("Current time: %s", time.Now().Format("2006-01-02 15:04:05 MST")),
		fmt.Sprintf("Model: %s", m.selectedModel),
	}
	if m.systemPrompt != "" {
		sysParts = append(sysParts, m.systemPrompt)
	}

	msgs := []ollama.ChatMessage{
		{Role: "system", Content: strings.Join(sysParts, "\n")},
	}
	for _, entry := range m.history {
		if entry.role == "assistant" && entry.text == "" {
			continue
		}
		msgs = append(msgs, ollama.ChatMessage{
			Role:    entry.role,
			Content: entry.text,
		})
	}
	return msgs
}

func (m model) runQuery() {
	p := *m.prog

	var finalPromptEval, finalEval int
	err := m.client.Chat(ollama.ChatRequest{
		Model:    m.selectedModel,
		Messages: m.buildChatMessages(),
		Options: &ollama.RequestOptions{
			Temperature: ollama.Float(0.7),
		},
	}, func(res ollama.ChatResponse) error {
		if res.Message.Content != "" {
			p.Send(tokenMsg(res.Message.Content))
		}
		if res.Done {
			finalPromptEval = res.PromptEvalCount
			finalEval = res.EvalCount
		}
		return nil
	})
	if err != nil {
		p.Send(errMsg{err: err})
		return
	}
	p.Send(doneMsg{promptEvalCount: finalPromptEval, evalCount: finalEval})
}

func (m *model) refreshViewport() {
	if !m.vpReady {
		return
	}
	var sb strings.Builder
	for i := range m.history {
		entry := &m.history[i]
		switch entry.role {
		case "user":
			sb.WriteString(userStyle.Render("You: "))
			sb.WriteString(entry.text)
			sb.WriteString("\n\n")
		case "assistant":
			sb.WriteString(assistantStyle.Render("AI:"))
			sb.WriteString("\n")
			isActive := m.streaming && i == len(m.history)-1
			if isActive {
				sb.WriteString(entry.text)
				sb.WriteString(dimStyle.Render("▊"))
			} else {
				sb.WriteString(m.renderMarkdown(entry.text))
			}
			sb.WriteString("\n")
		}
	}
	if m.err != nil {
		sb.WriteString(errorStyle.Render("Error: " + m.err.Error()))
		sb.WriteString("\n")
	}
	m.viewport.SetContent(sb.String())
	if m.streaming {
		m.viewport.GotoBottom()
	}
}

func (m *model) renderMarkdown(text string) string {
	if m.mdRenderer == nil || text == "" {
		return text
	}
	rendered, err := m.mdRenderer.Render(text)
	if err != nil {
		return text
	}
	return strings.TrimRight(rendered, "\n")
}

// --- View ------------------------------------------------------------------

func (m model) View() string {
	switch m.screen {
	case screenModelSelect:
		return m.viewModelSelect()
	case screenSystemPrompt:
		return m.viewSystemPrompt()
	case screenChat:
		return m.viewChat()
	}
	return ""
}

func (m model) viewModelSelect() string {
	iw := m.innerWidth()

	var sb strings.Builder
	sb.WriteString(titleStyle.Width(iw).Render("Select a Model"))
	sb.WriteString("\n\n")

	for i, name := range m.models {
		cursor := "  "
		style := modelItemStyle
		if i == m.cursor {
			cursor = modelCursorStyle.Render("▸ ")
			style = selectedModelStyle
		}
		line := cursor + style.Render(name)
		if pm, ok := m.runningModels[name]; ok {
			line += "  " + statsStyle.Render(
				fmt.Sprintf("● %s  vram=%s  ctx=%d",
					pm.Details.ParameterSize,
					formatBytes(pm.SizeVRAM),
					pm.ContextLength,
				),
			)
		}
		sb.WriteString(line + "\n")
	}

	sb.WriteString("\n")

	if m.systemPrompt != "" {
		sb.WriteString(systemStyle.Render("System: "))
		prompt := m.systemPrompt
		if len(prompt) > 60 {
			prompt = prompt[:57] + "..."
		}
		sb.WriteString(dimStyle.Render(prompt))
		sb.WriteString("\n")
	}

	sb.WriteString("\n")
	sb.WriteString(dimStyle.Render("↑/↓ navigate  •  enter select  •  s system prompt  •  q quit"))

	content := sb.String()

	lines := strings.Count(content, "\n") + 1
	if remaining := m.height - lines; remaining > 0 {
		content += strings.Repeat("\n", remaining)
	}

	return content
}

func (m model) viewSystemPrompt() string {
	iw := m.innerWidth()

	var sb strings.Builder
	sb.WriteString(titleStyle.Width(iw).Render("System Prompt"))
	sb.WriteString("\n\n")
	sb.WriteString(dimStyle.Render("Set instructions for the AI. Leave empty for default behavior."))
	sb.WriteString("\n\n")
	sb.WriteString(m.sysTA.View())
	sb.WriteString("\n\n")
	sb.WriteString(dimStyle.Render("esc save & back"))

	content := sb.String()

	lines := strings.Count(content, "\n") + 1
	if remaining := m.height - lines; remaining > 0 {
		content += strings.Repeat("\n", remaining)
	}

	return content
}

func (m model) viewChat() string {
	iw := m.innerWidth()

	var sb strings.Builder
	sb.WriteString(titleStyle.Width(iw).Render("Chat · " + m.selectedModel))
	sb.WriteString("\n")

	if m.vpReady {
		sb.WriteString(m.viewport.View())
		sb.WriteString("\n")
	}

	sb.WriteString(m.textarea.View())
	sb.WriteString("\n")

	sb.WriteString(statusBarStyle.Width(iw).Render(m.statusLine()))

	return sb.String()
}

func (m model) ctxInfo() string {
	if m.ctxSize <= 0 && m.ctxUsed <= 0 {
		return ""
	}
	avail := m.ctxSize - m.ctxUsed
	if avail < 0 {
		avail = 0
	}
	pct := float64(0)
	if m.ctxSize > 0 {
		pct = float64(m.ctxUsed) / float64(m.ctxSize) * 100
	}
	return fmt.Sprintf("ctx %d/%d (%.0f%% used, %d free)", m.ctxUsed, m.ctxSize, pct, avail)
}

func (m model) statusLine() string {
	ctx := m.ctxInfo()

	if m.streaming {
		elapsed := time.Since(m.streamStart).Seconds()
		tokSec := float64(0)
		if elapsed > 0 {
			tokSec = float64(m.tokenCount) / elapsed
		}
		stats := statsStyle.Render(
			fmt.Sprintf("%d tok  •  %.1f tok/s", m.tokenCount, tokSec),
		)
		line := fmt.Sprintf("⏳ %s", stats)
		if ctx != "" {
			line += "  •  " + statsStyle.Render(ctx)
		}
		return line + "  •  ctrl+c quit"
	}

	parts := []string{"enter send", "ctrl+m model", "esc back", "ctrl+c quit"}
	if m.tokenCount > 0 {
		stats := statsStyle.Render(
			fmt.Sprintf("%d tok  •  %.1f tok/s", m.tokenCount, m.lastTokSec),
		)
		line := fmt.Sprintf("✓ %s", stats)
		if ctx != "" {
			line += "  •  " + statsStyle.Render(ctx)
		}
		return line + "  •  " + strings.Join(parts, "  •  ")
	}
	if ctx != "" {
		return statsStyle.Render(ctx) + "  •  " + strings.Join(parts, "  •  ")
	}
	return strings.Join(parts, "  •  ")
}

func formatBytes(b int64) string {
	switch {
	case b >= 1<<30:
		return fmt.Sprintf("%.1fGB", float64(b)/(1<<30))
	case b >= 1<<20:
		return fmt.Sprintf("%.0fMB", float64(b)/(1<<20))
	default:
		return fmt.Sprintf("%dB", b)
	}
}

// --- Main ------------------------------------------------------------------

func main() {
	url := os.Getenv("OPEN_WEB_API_GENERATE_URL")
	token := os.Getenv("OPEN_WEB_API_TOKEN")

	if url == "" {
		// url = "http://191.168.1.134:11434/api/generate"
		url = "http://172.17.0.1:11434/api/generate"
	}

	client := ollama.NewOpenWebUiClient(&ollama.DSN{
		URL:   url,
		Token: token,
	})

	var p *tea.Program
	m := initialModel(client, &p)
	p = tea.NewProgram(m, tea.WithAltScreen(), tea.WithMouseCellMotion())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
