// Package main demonstrates basic usage of the ollama client library.
//
// By default it uses local Ollama. Optional overrides:
//
//	export OPEN_WEB_API_GENERATE_URL="https://ai.example.com/ollama/api/generate"
//	export OPEN_WEB_API_TOKEN="sk-..."
//	go run ./examples/basic/
//
// Chats are stored as chats/convers-<n>.yml with multi-turn messages. Enter in
// Prompt sends; history is included in the next request. Sending again while
// streaming bumps the request generation, clears the partial reply, and uses the
// new prompt for that turn. Tab / Ctrl+arrow cycles focus.
package main

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	markdown "github.com/MichaelMure/go-term-markdown"
	ollama "github.com/eslider/go-ollama"
	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
)

const defaultModel = "gemma3:1b"

// contentPane wraps TextView to add mouse-drag line selection (capture) and copy on release.
type contentPane struct {
	*tview.TextView
	dragging             bool
	dragMoved            bool
	dragStart, dragEnd   int
	innerMouse           func(action tview.MouseAction, event *tcell.EventMouse, setFocus func(p tview.Primitive)) (consumed bool, capture tview.Primitive)
	onExitKeyboardVisual func()
}

func newContentPane(tv *tview.TextView) *contentPane {
	return &contentPane{TextView: tv}
}

func (p *contentPane) MouseHandler() func(action tview.MouseAction, event *tcell.EventMouse, setFocus func(p tview.Primitive)) (consumed bool, capture tview.Primitive) {
	if p.innerMouse == nil {
		p.innerMouse = p.TextView.MouseHandler()
	}
	inner := p.innerMouse
	return func(action tview.MouseAction, event *tcell.EventMouse, setFocus func(p tview.Primitive)) (consumed bool, capture tview.Primitive) {
		x, y := event.Position()
		switch action {
		case tview.MouseLeftDown:
			if p.InInnerRect(x, y) {
				setFocus(p)
				if p.onExitKeyboardVisual != nil {
					p.onExitKeyboardVisual()
				}
				p.dragging = true
				p.dragMoved = false
				line := globalLineFromMouse(p.TextView, x, y)
				p.dragStart, p.dragEnd = line, line
				highlightLineRange(p.TextView, line, line)
				return true, p
			}
		case tview.MouseMove:
			if p.dragging {
				p.dragMoved = true
				line := globalLineFromMouse(p.TextView, x, y)
				if line >= 0 {
					p.dragEnd = line
					highlightLineRange(p.TextView, p.dragStart, p.dragEnd)
				}
				return true, p
			}
		case tview.MouseLeftUp:
			if p.dragging {
				p.dragging = false
				shouldCopy := p.dragStart != p.dragEnd || p.dragMoved
				if shouldCopy {
					lo, hi := min(p.dragStart, p.dragEnd), max(p.dragStart, p.dragEnd)
					lines := plainLines(p.TextView)
					if len(lines) > 0 {
						if hi >= len(lines) {
							hi = len(lines) - 1
						}
						if lo < 0 {
							lo = 0
						}
						_ = copyToClipboard(strings.Join(lines[lo:hi+1], "\n"))
					}
				}
				p.dragMoved = false
				p.TextView.Highlight()
				return true, nil
			}
		}
		if inner != nil {
			return inner(action, event, setFocus)
		}
		return false, nil
	}
}

func plainLines(tv *tview.TextView) []string {
	t := tv.GetText(true)
	if t == "" {
		return nil
	}
	return strings.Split(t, "\n")
}

func globalLineFromMouse(tv *tview.TextView, x, y int) int {
	if !tv.InInnerRect(x, y) {
		return -1
	}
	_, rectY, _, _ := tv.GetInnerRect()
	row, _ := tv.GetScrollOffset()
	innerY := y - rectY
	lineIdx := innerY + row
	lines := plainLines(tv)
	n := len(lines)
	if n == 0 {
		return -1
	}
	if lineIdx < 0 {
		lineIdx = 0
	}
	if lineIdx >= n {
		lineIdx = n - 1
	}
	return lineIdx
}

// annotateLineRegions wraps each logical line in a tview region so Highlight() inverts that line.
func annotateLineRegions(translated string) string {
	lines := strings.Split(translated, "\n")
	var b strings.Builder
	for i, line := range lines {
		fmt.Fprintf(&b, `["L%d"]%s[""]`, i, line)
		if i < len(lines)-1 {
			b.WriteByte('\n')
		}
	}
	return b.String()
}

func highlightLineRange(tv *tview.TextView, a, b int) {
	lines := plainLines(tv)
	n := len(lines)
	if n == 0 {
		tv.Highlight()
		return
	}
	lo, hi := min(a, b), max(a, b)
	if lo < 0 {
		lo = 0
	}
	if hi >= n {
		hi = n - 1
	}
	ids := make([]string, 0, hi-lo+1)
	for i := lo; i <= hi; i++ {
		ids = append(ids, fmt.Sprintf("L%d", i))
	}
	tv.Highlight(ids...)
}

func copyToClipboard(s string) error {
	if s == "" {
		return nil
	}
	try := []struct {
		name string
		args []string
	}{
		{"wl-copy", nil},
		{"xclip", []string{"-selection", "clipboard"}},
		{"xsel", []string{"--clipboard", "--input"}},
	}
	var lastErr error
	for _, c := range try {
		path, err := exec.LookPath(c.name)
		if err != nil {
			lastErr = err
			continue
		}
		cmd := exec.Command(path, c.args...)
		cmd.Stdin = strings.NewReader(s)
		if err := cmd.Run(); err == nil {
			return nil
		}
		lastErr = err
	}
	if lastErr != nil {
		return lastErr
	}
	return fmt.Errorf("no clipboard helper (install wl-copy, xclip, or xsel)")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// buildGeneratePrompt turns prior turns + the latest user message into one prompt for /api/generate.
func buildGeneratePrompt(history []Message, lastUser string) string {
	var b strings.Builder
	for _, m := range history {
		switch m.Role {
		case "user":
			b.WriteString("User: ")
			b.WriteString(m.Content)
			b.WriteString("\n\n")
		case "assistant":
			b.WriteString("Assistant: ")
			b.WriteString(m.Content)
			b.WriteString("\n\n")
		}
	}
	b.WriteString("User: ")
	b.WriteString(lastUser)
	b.WriteString("\n\nAssistant:")
	return b.String()
}

// formatTranscriptMarkdown renders the full thread for the markdown pane (history + in-flight turn).
func formatTranscriptMarkdown(history []Message, pending string, assistantDraft string) string {
	var b strings.Builder
	for _, m := range history {
		switch m.Role {
		case "user":
			b.WriteString("### User\n\n")
			b.WriteString(m.Content)
			b.WriteString("\n\n")
		case "assistant":
			b.WriteString("### Assistant\n\n")
			b.WriteString(m.Content)
			b.WriteString("\n\n")
		}
	}
	if pending != "" {
		b.WriteString("### User\n\n")
		b.WriteString(pending)
		b.WriteString("\n\n")
	}
	if assistantDraft != "" {
		b.WriteString("### Assistant\n\n")
		b.WriteString(assistantDraft)
	}
	return strings.TrimSpace(b.String())
}

// brailleSpinner returns a rotating Braille frame for activity indication.
func brailleSpinner(frame int) string {
	s := []rune("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
	return string(s[frame%len(s)])
}

// indeterminateBar is a sliding highlight over a light track (no total size known).
func indeterminateBar(width, frame int) string {
	if width <= 0 {
		return ""
	}
	r := make([]rune, width)
	for i := range r {
		r[i] = '░'
	}
	for k := 0; k < min(4, width); k++ {
		r[(frame+k)%width] = '█'
	}
	return string(r)
}

// tokenFillBar grows with output tokens (asymptotic — no max token count known).
func tokenFillBar(width, tokens, frame int) string {
	if width <= 0 {
		return ""
	}
	fill := int(float64(width) * (1.0 - math.Exp(-float64(tokens)/200.0)))
	if fill > width {
		fill = width
	}
	r := make([]rune, width)
	for i := range r {
		if i < fill {
			r[i] = '█'
		} else {
			r[i] = '░'
		}
	}
	// subtle pulse on the frontier
	if fill > 0 && fill < width {
		r[(frame+fill)%width] = '▓'
	}
	return string(r)
}

func progressBar(width, tokens, frame int) string {
	if tokens == 0 {
		return indeterminateBar(width, frame)
	}
	return tokenFillBar(width, tokens, frame)
}

func main() {
	client := ollama.NewOpenWebUiClient(&ollama.DSN{
		URL:   os.Getenv("OPEN_WEB_API_GENERATE_URL"),
		Token: os.Getenv("OPEN_WEB_API_TOKEN"),
	})

	app := tview.NewApplication()

	var (
		visualMode             bool
		lineAnchor, lineCursor int
		mu                     sync.Mutex
	)

	var (
		currentPath   string
		currentID     int
		chatMessages  []Message
		pendingUser   string // in-flight user message (not yet committed to chatMessages)
		responseBuf   strings.Builder
		tokenCount    int
		startTime     time.Time
		duration      time.Duration
		queryGen      uint64
		chatEntries   []chatListEntry
		streamRunning bool
		// Throttle streaming redraws — full markdown render every token blocks the UI thread.
		streamDrawMu     sync.Mutex
		lastStreamRedraw time.Time
	)

	tv := tview.NewTextView().
		SetDynamicColors(true).
		SetScrollable(true).
		SetWrap(false).
		SetRegions(true).
		SetChangedFunc(func() {
			app.Draw()
		})
	tv.SetBorder(true).SetTitle("Response · hjkl · V · y · drag · Tab / Ctrl+arrows")

	pane := newContentPane(tv)

	promptInput := tview.NewInputField().
		SetLabel("Prompt ").
		SetFieldWidth(0)
	promptInput.SetBorder(true).SetTitle("Enter = send")

	statusView := tview.NewTextView().
		SetDynamicColors(true).
		SetTextAlign(tview.AlignLeft)
	statusView.SetBorder(true).SetTitle("Status")

	navList := tview.NewList().
		ShowSecondaryText(true).
		SetWrapAround(true)

	navList.SetBorder(true).SetTitle("Chats · ↑↓ j/k · Enter · n new")

	// j/k act like vim for vertical movement (List defaults use arrows/Tab, not j/k).
	navList.SetInputCapture(func(ev *tcell.EventKey) *tcell.EventKey {
		if ev.Key() == tcell.KeyRune {
			switch ev.Rune() {
			case 'j':
				return tcell.NewEventKey(tcell.KeyDown, 0, tcell.ModNone)
			case 'k':
				return tcell.NewEventKey(tcell.KeyUp, 0, tcell.ModNone)
			case 'g':
				return tcell.NewEventKey(tcell.KeyHome, 0, tcell.ModNone)
			case 'G':
				return tcell.NewEventKey(tcell.KeyEnd, 0, tcell.ModNone)
			}
		}
		return ev
	})

	mainCol := tview.NewFlex().
		SetDirection(tview.FlexRow).
		AddItem(pane, 0, 1, true).
		AddItem(promptInput, 3, 0, false).
		AddItem(statusView, 4, 0, false)

	// FlexColumn lays out children left-to-right: nav strip on the left, main column on the right.
	root := tview.NewFlex().
		SetDirection(tview.FlexColumn).
		AddItem(navList, 30, 0, true).
		AddItem(mainCol, 0, 1, false)

	updateStatus := func() {
		mu.Lock()
		vm := visualMode
		a, c := lineAnchor, lineCursor
		mu.Unlock()

		tokensPerSecond := 0.0
		if duration.Seconds() > 0 {
			tokensPerSecond = float64(tokenCount) / duration.Seconds()
		}
		statusView.Clear()
		base := ""
		if currentPath != "" {
			base = filepath.Base(currentPath)
		}
		if vm {
			lo, hi := min(a, c), max(a, c)
			fmt.Fprintf(statusView,
				"[yellow]%s[white] | [yellow]VISUAL[white] %d–%d | j/k · y · Esc\n[yellow]Tokens:[white] %d | [yellow]Time:[white] %v | [yellow]tok/s:[white] %.2f",
				base, lo+1, hi+1, tokenCount, duration, tokensPerSecond)
			return
		}
		if streamRunning {
			fr := 0
			if !startTime.IsZero() {
				fr = int(time.Since(startTime).Milliseconds() / 80)
			}
			spin := brailleSpinner(fr)
			bar := progressBar(22, tokenCount, fr)
			phase := "Requesting"
			icon := "[yellow]◌[white]"
			if tokenCount > 0 {
				phase = "Generating"
				icon = "[cyan]▶[white]"
			}
			fmt.Fprintf(statusView,
				"%s %s [yellow]%s[white]  [gray]%s[white]\n[cyan]%s[white]  tok [yellow]%d[white]  ·  [gray]%v[white]  ·  [green]%.1f[white] tok/s",
				icon, spin, phase, base, bar, tokenCount, duration, tokensPerSecond)
			return
		}
		fmt.Fprintf(statusView,
			"[gray]●[white] [green]Ready[white]  |  [gray]%s[white]\n[yellow]tok[white] %d  ·  [gray]%v[white]  ·  [green]%.1f[white] tok/s",
			base, tokenCount, duration, tokensPerSecond)
	}

	saveCurrentChat := func() {
		if currentPath == "" {
			return
		}
		c := &Conversation{
			ID:       currentID,
			Model:    defaultModel,
			Messages: chatMessages,
			Title:    deriveTitleFromMessages(chatMessages),
		}
		if existing, err := loadConversation(currentPath); err == nil && strings.TrimSpace(existing.Title) != "" && existing.Title != "New chat" {
			c.Title = existing.Title
		}
		if c.Title == "" || c.Title == "Untitled" {
			c.Title = deriveTitleFromMessages(chatMessages)
		}
		if c.Title == "" {
			c.Title = "New chat"
		}
		_ = c.Save(currentPath)
	}

	pushContent := func(text string) {
		pane.Clear()
		rendered := string(markdown.Render(text, 80, 6))
		translated := tview.TranslateANSI(rendered)
		fmt.Fprint(pane, annotateLineRegions(translated))
		mu.Lock()
		visualMode = false
		lineAnchor, lineCursor = 0, 0
		mu.Unlock()
		pane.Highlight()
		updateStatus()
	}

	pane.onExitKeyboardVisual = func() {
		mu.Lock()
		visualMode = false
		mu.Unlock()
		pane.SetBorderColor(tview.Styles.BorderColor)
		updateStatus()
	}

	refreshNavList := func() {
		navList.Clear()
		navList.AddItem("New conversation", "Save as chats/convers-<n>.yml", 'n', nil)
		var err error
		chatEntries, err = scanConversations()
		if err != nil {
			navList.AddItem(fmt.Sprintf("Error: %v", err), "", 0, nil)
			return
		}
		for _, e := range chatEntries {
			title := listTitle(e.Path, e.ID)
			navList.AddItem(title, filepath.Base(e.Path), 0, nil)
		}
	}

	selectNavIndex := func(targetPath string) {
		if targetPath == "" {
			return
		}
		for i, e := range chatEntries {
			if e.Path == targetPath {
				navList.SetCurrentItem(i + 1)
				return
			}
		}
	}

	loadChat := func(path string) {
		if path == currentPath {
			return
		}
		atomic.AddUint64(&queryGen, 1)
		saveCurrentChat()
		c, err := loadConversation(path)
		if err != nil {
			pane.Clear()
			fmt.Fprintf(pane, "[red]load %s: %v[white]", path, err)
			updateStatus()
			return
		}
		currentPath = path
		currentID = c.ID
		chatMessages = append([]Message(nil), c.Messages...)
		pendingUser = ""
		responseBuf.Reset()
		promptInput.SetText("")
		pushContent(formatTranscriptMarkdown(chatMessages, pendingUser, responseBuf.String()))
		selectNavIndex(path)
		updateStatus()
	}

	ensureChatExists := func() {
		if currentPath != "" {
			return
		}
		if err := ensureChatsDir(); err != nil {
			return
		}
		id, err := nextConversationID()
		if err != nil {
			return
		}
		path := conversationPath(id)
		c := &Conversation{
			ID:    id,
			Title: "New chat",
			Model: defaultModel,
		}
		if err := c.Save(path); err != nil {
			return
		}
		currentPath = path
		currentID = id
		chatMessages = nil
		refreshNavList()
		selectNavIndex(path)
	}

	startNewChat := func() {
		atomic.AddUint64(&queryGen, 1)
		saveCurrentChat()
		if err := ensureChatsDir(); err != nil {
			return
		}
		id, err := nextConversationID()
		if err != nil {
			return
		}
		path := conversationPath(id)
		c := &Conversation{ID: id, Title: "New chat", Model: defaultModel}
		if err := c.Save(path); err != nil {
			return
		}
		currentPath = path
		currentID = id
		chatMessages = nil
		pendingUser = ""
		responseBuf.Reset()
		promptInput.SetText("")
		pushContent("")
		refreshNavList()
		selectNavIndex(path)
		updateStatus()
	}

	startQuery := func() {
		userText := strings.TrimSpace(promptInput.GetText())
		if userText == "" {
			return
		}
		ensureChatExists()
		if currentPath == "" {
			return
		}

		// Invalidate any in-flight stream; new prompt replaces the in-progress assistant output.
		atomic.AddUint64(&queryGen, 1)
		gen := atomic.LoadUint64(&queryGen)

		pendingUser = userText
		promptInput.SetText("")
		responseBuf.Reset()
		tokenCount = 0
		startTime = time.Now()
		duration = 0
		streamRunning = true
		apiPrompt := buildGeneratePrompt(chatMessages, pendingUser)
		streamDrawMu.Lock()
		lastStreamRedraw = time.Time{}
		streamDrawMu.Unlock()

		// Must not use QueueUpdateDraw here: startQuery runs on the UI event loop, and
		// QueueUpdate blocks until that loop processes the update — deadlock.
		pushContent(formatTranscriptMarkdown(chatMessages, pendingUser, responseBuf.String()))

		go func() {
			err := client.Query(ollama.Request{
				Model:  defaultModel,
				Prompt: apiPrompt,
				Stream: new(true),
				Options: &ollama.RequestOptions{
					Temperature: new(0.7),
				},
				OnJson: func(res ollama.Response) error {
					if atomic.LoadUint64(&queryGen) != gen {
						return nil
					}

					done := res.Done != nil && *res.Done

					if res.Response != nil {
						responseBuf.WriteString(*res.Response)
						tokenCount++
						duration = time.Since(startTime)
					}

					if done {
						app.QueueUpdateDraw(func() {
							streamRunning = false
							assistant := responseBuf.String()
							u := pendingUser
							chatMessages = append(chatMessages,
								Message{Role: "user", Content: u},
								Message{Role: "assistant", Content: assistant},
							)
							pendingUser = ""
							responseBuf.Reset()
							pushContent(formatTranscriptMarkdown(chatMessages, pendingUser, responseBuf.String()))
							saveCurrentChat()
							refreshNavList()
							selectNavIndex(currentPath)
							updateStatus()
						})
						return nil
					}

					// Interim redraw: at most ~12–15 full renders per second so the UI stays responsive.
					if res.Response == nil {
						return nil
					}
					streamDrawMu.Lock()
					allow := time.Since(lastStreamRedraw) >= 75*time.Millisecond
					if allow {
						lastStreamRedraw = time.Now()
					}
					streamDrawMu.Unlock()
					if !allow {
						return nil
					}
					app.QueueUpdateDraw(func() {
						pushContent(formatTranscriptMarkdown(chatMessages, pendingUser, responseBuf.String()))
						updateStatus()
					})
					return nil
				},
			})
			app.QueueUpdateDraw(func() {
				streamRunning = false
				if err != nil {
					if atomic.LoadUint64(&queryGen) != gen {
						return
					}
					if responseBuf.Len() > 0 && pendingUser != "" {
						chatMessages = append(chatMessages,
							Message{Role: "user", Content: pendingUser},
							Message{Role: "assistant", Content: responseBuf.String()},
						)
						pendingUser = ""
						responseBuf.Reset()
					}
					pushContent(formatTranscriptMarkdown(chatMessages, pendingUser, responseBuf.String()) +
						"\n\n[red]**Error:** " + tview.Escape(err.Error()) + "[white]")
					saveCurrentChat()
				}
				updateStatus()
			})
		}()
	}

	navList.SetSelectedFunc(func(idx int, main, secondary string, shortcut rune) {
		if idx == 0 {
			startNewChat()
			return
		}
		if idx-1 < len(chatEntries) {
			loadChat(chatEntries[idx-1].Path)
		}
	})

	promptInput.SetInputCapture(func(ev *tcell.EventKey) *tcell.EventKey {
		if ev.Key() == tcell.KeyEnter {
			startQuery()
			return nil
		}
		return ev
	})

	pane.SetInputCapture(func(ev *tcell.EventKey) *tcell.EventKey {
		mu.Lock()
		vm := visualMode
		mu.Unlock()

		if vm {
			lines := plainLines(pane.TextView)
			n := len(lines)
			if n == 0 {
				n = 1
			}

			moveCursor := func(delta int) {
				mu.Lock()
				lineCursor += delta
				if lineCursor < 0 {
					lineCursor = 0
				}
				if lineCursor > n-1 {
					lineCursor = n - 1
				}
				a, c := lineAnchor, lineCursor
				mu.Unlock()
				pane.ScrollTo(lineCursor, 0)
				highlightLineRange(pane.TextView, a, c)
				updateStatus()
			}

			switch ev.Key() {
			case tcell.KeyEsc:
				mu.Lock()
				visualMode = false
				mu.Unlock()
				pane.Highlight()
				updateStatus()
				pane.SetBorderColor(tview.Styles.BorderColor)
				return nil
			case tcell.KeyUp:
				moveCursor(-1)
				return nil
			case tcell.KeyDown:
				moveCursor(1)
				return nil
			case tcell.KeyLeft, tcell.KeyRight, tcell.KeyPgUp, tcell.KeyPgDn,
				tcell.KeyCtrlF, tcell.KeyCtrlB:
				return ev
			case tcell.KeyRune:
				switch ev.Rune() {
				case 'y':
					mu.Lock()
					a, c := lineAnchor, lineCursor
					mu.Unlock()
					lo, hi := min(a, c), max(a, c)
					if len(lines) > 0 {
						if hi >= len(lines) {
							hi = len(lines) - 1
						}
						if lo < 0 {
							lo = 0
						}
						_ = copyToClipboard(strings.Join(lines[lo:hi+1], "\n"))
					}
					mu.Lock()
					visualMode = false
					mu.Unlock()
					pane.Highlight()
					updateStatus()
					pane.SetBorderColor(tview.Styles.BorderColor)
					return nil
				case 'j':
					moveCursor(1)
					return nil
				case 'k':
					moveCursor(-1)
					return nil
				case 'h', 'l':
					return ev
				default:
					return nil
				}
			default:
				return nil
			}
		}

		if ev.Key() == tcell.KeyRune && ev.Rune() == 'V' {
			row, _ := pane.GetScrollOffset()
			lines := plainLines(pane.TextView)
			n := len(lines)
			if n == 0 {
				return nil
			}
			if row >= n {
				row = n - 1
			}
			mu.Lock()
			visualMode = true
			lineAnchor, lineCursor = row, row
			mu.Unlock()
			highlightLineRange(pane.TextView, lineAnchor, lineCursor)
			pane.SetBorderColor(tcell.ColorYellow)
			updateStatus()
			return nil
		}

		if ev.Key() == tcell.KeyRune && ev.Rune() == 'y' {
			plain := pane.GetText(true)
			_ = copyToClipboard(plain)
			return nil
		}

		return ev
	})

	_ = ensureChatsDir()
	refreshNavList()
	if len(chatEntries) > 0 {
		loadChat(chatEntries[0].Path)
	}

	focusOrder := []tview.Primitive{navList, promptInput, pane, statusView}
	focusStep := func(delta int) {
		cur := app.GetFocus()
		idx := 0
		found := false
		for i, p := range focusOrder {
			if p == cur {
				idx = i
				found = true
				break
			}
		}
		if !found {
			idx = 0
		}
		idx = (idx + delta + len(focusOrder)) % len(focusOrder)
		app.SetFocus(focusOrder[idx])
	}

	app.SetInputCapture(func(event *tcell.EventKey) *tcell.EventKey {
		switch event.Key() {
		case tcell.KeyCtrlC:
			saveCurrentChat()
			app.Stop()
			return nil
		case tcell.KeyTab:
			switch app.GetFocus() {
			case navList:
				app.SetFocus(promptInput)
			case promptInput:
				app.SetFocus(pane)
			case pane:
				app.SetFocus(statusView)
			case statusView:
				app.SetFocus(navList)
			default:
				app.SetFocus(navList)
			}
			return nil
		default:
			if event.Modifiers()&tcell.ModCtrl != 0 {
				switch event.Key() {
				case tcell.KeyLeft, tcell.KeyUp:
					focusStep(-1)
					return nil
				case tcell.KeyRight, tcell.KeyDown:
					focusStep(1)
					return nil
				}
			}
			return event
		}
	})

	if err := app.SetRoot(root, true).EnableMouse(true).Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error running TUI: %v\n", err)
		os.Exit(1)
	}
}
