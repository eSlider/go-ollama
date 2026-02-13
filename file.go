package ollama

import (
	"fmt"
	"os"
	"path/filepath"
)

// OpenFileDescriptor opens a file descriptor at the given path
//   - If the file does not exist, it will be created
//   - If the directory does not exist, it will be created
//   - Returns the file descriptor and an error if any
//   - The path can be absolute or relative to the current working directory
func OpenFileDescriptor(path string) (*os.File, error) {
	// Resolve to absolute path if relative
	if !filepath.IsAbs(path) {
		cwd, err := os.Getwd()
		if err != nil {
			return nil, fmt.Errorf("failed to get working directory: %w", err)
		}
		path = filepath.Join(cwd, path)
	}

	// Ensure parent directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	return os.OpenFile(path, os.O_CREATE|os.O_WRONLY, 0644)
}
