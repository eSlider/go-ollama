package ollama

import (
	"bufio"
	"bytes"
	"io"
)

// NewSplitScanner returns a new SplitScanner to split the data at the given substring
func NewSplitScanner(body io.ReadCloser, splitChar string) *bufio.Scanner {
	sc := bufio.NewScanner(body)
	sc.Split(SplitAt(splitChar))
	return sc
}

// SplitAt Custom split function. This will split the data at the given substring.
func SplitAt(substring string) func(data []byte, atEOF bool) (advance int, token []byte, err error) {
	searchBytes := []byte(substring)
	searchLength := len(substring)
	return func(data []byte, atEOF bool) (advance int, token []byte, err error) {
		dataLen := len(data)

		// Return Nothing if at the end of file or no data passed.
		if atEOF && dataLen == 0 {
			return 0, nil, nil
		}

		// Find next separator and return token.
		if i := bytes.Index(data, searchBytes); i >= 0 {
			return i + searchLength, data[0:i], nil
		}

		// If we're at EOF, we have a final, non-terminated line. Return it.
		if atEOF {
			return dataLen, data, nil
		}

		// Request more data.
		return 0, nil, nil
	}
}
