package ollama

// Int returns a pointer to an int
func Int(i int) *int {
	return &i
}

// String returns a pointer to a string
func String(s string) *string {
	return &s
}

// Bool returns a pointer to a bool
func Bool(b bool) *bool {
	return &b
}

// Float returns a pointer to a float64
func Float(i float64) *float64 {
	return &i
}
