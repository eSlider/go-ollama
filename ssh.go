package ollama

import (
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"

	sshlib "github.com/blacknon/go-sshlib"
	"golang.org/x/crypto/ssh"
)

type sshPool struct {
	alias string
	once  sync.Once
	con   *sshlib.Connect
	err   error
}

func newSSHPool(alias string) *sshPool {
	return &sshPool{alias: alias}
}

func (p *sshPool) DialContext(ctx context.Context, network, addr string) (net.Conn, error) {
	p.once.Do(func() {
		p.con, p.err = dialSSH(p.alias)
	})
	if p.err != nil {
		return nil, p.err
	}
	type result struct {
		c   net.Conn
		err error
	}
	ch := make(chan result, 1)
	go func() {
		c, err := p.con.Dial(network, addr)
		ch <- result{c: c, err: err}
	}()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case r := <-ch:
		return r.c, r.err
	}
}

func (p *sshPool) Close() error {
	if p == nil || p.con == nil {
		return nil
	}
	return p.con.Close()
}

func dialSSH(alias string) (*sshlib.Connect, error) {
	cfgPath := filepath.Join(os.Getenv("HOME"), ".ssh", "config")
	raw, err := os.ReadFile(cfgPath)
	if err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("ssh config: %w", err)
	}
	host := parseOpenSSHHost(string(raw), alias)
	if host.HostName == "" {
		host.HostName = alias
	}
	if host.Port == "" {
		host.Port = "22"
	}
	if host.User == "" {
		host.User = os.Getenv("USER")
	}

	auths := sshAuthMethods(host.IdentityFiles)
	if len(auths) == 0 {
		return nil, fmt.Errorf("ssh %s: no agent keys or identity files", alias)
	}

	con := &sshlib.Connect{
		AutoReconnect: true,
		Agent:         sshlib.ConnectSshAgent(),
	}

	if err := con.CreateClient(host.HostName, host.Port, host.User, auths); err != nil {
		return nil, fmt.Errorf("ssh %s (%s@%s:%s): %w", alias, host.User, host.HostName, host.Port, err)
	}
	return con, nil
}

func sshAuthMethods(identityFiles []string) []ssh.AuthMethod {
	auths := []ssh.AuthMethod{}
	ag := sshlib.ConnectSshAgent()
	if signers, err := sshlib.CreateSignerAgent(ag); err == nil && len(signers) > 0 {
		auths = append(auths, ssh.PublicKeys(signers...))
	}
	seen := map[string]struct{}{}
	for _, f := range identityFiles {
		f = expandHome(f)
		if f == "" {
			continue
		}
		if _, ok := seen[f]; ok {
			continue
		}
		seen[f] = struct{}{}
		if _, err := os.Stat(f); err != nil {
			continue
		}
		auth, err := sshlib.CreateAuthMethodPublicKey(f, "")
		if err != nil {
			continue
		}
		auths = append(auths, auth)
	}
	return auths
}

func expandHome(p string) string {
	p = strings.TrimSpace(p)
	if p == "~" {
		return os.Getenv("HOME")
	}
	if strings.HasPrefix(p, "~/") {
		return filepath.Join(os.Getenv("HOME"), p[2:])
	}
	return p
}

type openSSHHost struct {
	User          string
	HostName      string
	Port          string
	IdentityFiles []string
}

func parseOpenSSHHost(configText, alias string) openSSHHost {
	var out openSSHHost
	var apply bool
	for _, line := range strings.Split(configText, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		key := strings.ToLower(fields[0])
		if key == "host" {
			apply = hostPatternsMatch(fields[1:], alias)
			continue
		}
		if !apply {
			continue
		}
		val := strings.Join(fields[1:], " ")
		switch key {
		case "user":
			out.User = val
		case "hostname":
			out.HostName = val
		case "port":
			out.Port = val
		case "identityfile":
			out.IdentityFiles = append(out.IdentityFiles, val)
		}
	}
	return out
}

func hostPatternsMatch(patterns []string, alias string) bool {
	matched := false
	for _, p := range patterns {
		neg := strings.HasPrefix(p, "!")
		if neg {
			p = p[1:]
		}
		ok := sshHostMatch(p, alias)
		if neg && ok {
			return false
		}
		if ok {
			matched = true
		}
	}
	return matched
}

func sshHostMatch(pattern, name string) bool {
	if pattern == "*" {
		return true
	}
	if !strings.ContainsAny(pattern, "*?") {
		return pattern == name
	}
	ok, err := filepath.Match(pattern, name)
	return err == nil && ok
}
