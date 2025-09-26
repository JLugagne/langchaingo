package vertex

import (
	"strings"
	"testing"

	"google.golang.org/genai"
	"github.com/tmc/langchaingo/llms"
)

func TestConvertToolSchemaType(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected genai.Type
	}{
		{"object type", "object", genai.TypeObject},
		{"string type", "string", genai.TypeString},
		{"number type", "number", genai.TypeNumber},
		{"integer type", "integer", genai.TypeInteger},
		{"boolean type", "boolean", genai.TypeBoolean},
		{"array type", "array", genai.TypeArray},
		{"unknown type", "unknown", genai.TypeUnspecified},
		{"empty type", "", genai.TypeUnspecified},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertToolSchemaType(tt.input)
			if result != tt.expected {
				t.Errorf("convertToolSchemaType(%q) = %v, want %v", tt.input, result, tt.expected)
			}
		})
	}
}

func TestConvertParts(t *testing.T) {
	// Skip this test since convertParts is deprecated in the new API
	t.Skip("convertParts function is deprecated in favor of convertMessages")
}

func TestConvertContent(t *testing.T) {
	// Skip this test since convertContent is deprecated in the new API
	t.Skip("convertContent function is deprecated in favor of convertMessages")
}

func TestConvertCandidates(t *testing.T) {
	// Skip this test since convertCandidates is deprecated in the new API
	t.Skip("convertCandidates function is deprecated in favor of convertResponse")
}
func TestConvertTools(t *testing.T) { //nolint:funlen // comprehensive test //nolint:funlen // comprehensive test
	tests := []struct {
		name    string
		tools   []llms.Tool
		wantErr bool
		check   func(t *testing.T, result []*genai.Tool)
	}{
		{
			name: "single function tool",
			tools: []llms.Tool{
				{
					Type: "function",
					Function: &llms.FunctionDefinition{
						Name:        "get_weather",
						Description: "Get weather information",
						Parameters: map[string]any{
							"type": "object",
							"properties": map[string]any{
								"location": map[string]any{
									"type":        "string",
									"description": "The city name",
								},
								"unit": map[string]any{
									"type":        "string",
									"description": "Temperature unit",
								},
							},
							"required": []string{"location"},
						},
					},
				},
			},
			check: func(t *testing.T, result []*genai.Tool) {
				if len(result) != 1 {
					t.Fatalf("expected 1 tool, got %d", len(result))
				}
				if len(result[0].FunctionDeclarations) != 1 {
					t.Fatalf("expected 1 function declaration, got %d", len(result[0].FunctionDeclarations))
				}
				fd := result[0].FunctionDeclarations[0]
				if fd.Name != "get_weather" {
					t.Errorf("expected function name 'get_weather', got %q", fd.Name)
				}
				if fd.Description != "Get weather information" {
					t.Errorf("expected description, got %q", fd.Description)
				}
				if fd.Parameters.Type != genai.TypeObject {
					t.Errorf("expected object type, got %v", fd.Parameters.Type)
				}
				if len(fd.Parameters.Properties) != 2 {
					t.Errorf("expected 2 properties, got %d", len(fd.Parameters.Properties))
				}
				if fd.Parameters.Properties["location"].Type != genai.TypeString {
					t.Errorf("expected location to be string type")
				}
				if len(fd.Parameters.Required) != 1 || fd.Parameters.Required[0] != "location" {
					t.Errorf("expected required=['location'], got %v", fd.Parameters.Required)
				}
			},
		},
		{
			name: "unsupported tool type",
			tools: []llms.Tool{
				{
					Type: "unsupported",
					Function: &llms.FunctionDefinition{
						Name: "test",
					},
				},
			},
			wantErr: true,
		},
		{
			name: "invalid parameters type",
			tools: []llms.Tool{
				{
					Type: "function",
					Function: &llms.FunctionDefinition{
						Name:       "test",
						Parameters: "invalid",
					},
				},
			},
			wantErr: true,
		},
		{
			name: "invalid properties type",
			tools: []llms.Tool{
				{
					Type: "function",
					Function: &llms.FunctionDefinition{
						Name: "test",
						Parameters: map[string]any{
							"type":       "object",
							"properties": "invalid",
						},
					},
				},
			},
			wantErr: true,
		},
		{
			name: "required as interface slice",
			tools: []llms.Tool{
				{
					Type: "function",
					Function: &llms.FunctionDefinition{
						Name: "test",
						Parameters: map[string]any{
							"type": "object",
							"properties": map[string]any{
								"field": map[string]any{"type": "string"},
							},
							"required": []interface{}{"field"},
						},
					},
				},
			},
			check: func(t *testing.T, result []*genai.Tool) {
				fd := result[0].FunctionDeclarations[0]
				if len(fd.Parameters.Required) != 1 || fd.Parameters.Required[0] != "field" {
					t.Errorf("expected required=['field'], got %v", fd.Parameters.Required)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := convertTools(tt.tools)
			if (err != nil) != tt.wantErr {
				t.Errorf("convertTools() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && tt.check != nil {
				tt.check(t, result)
			}
		})
	}
}

func TestShowContent(t *testing.T) {
	// This is mainly for coverage - just ensure it doesn't panic with new API structure
	var buf strings.Builder
	contents := []*genai.Content{
		{
			Role: "user",
			Parts: []*genai.Part{
				{Text: "Hello"},
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1, 2, 3}}},
				{FunctionCall: &genai.FunctionCall{Name: "test", Args: map[string]any{"x": 1}}},
				{FunctionResponse: &genai.FunctionResponse{Name: "test", Response: map[string]any{"result": "ok"}}},
			},
		},
	}

	// Should not panic
	showContent(&buf, contents)

	output := buf.String()
	if !strings.Contains(output, "Role=user") {
		t.Error("expected output to contain role")
	}
	if !strings.Contains(output, "Text \"Hello\"") {
		t.Error("expected output to contain text")
	}
	if !strings.Contains(output, "Blob MIME=\"image/png\"") {
		t.Error("expected output to contain blob info")
	}
	if !strings.Contains(output, "FunctionCall Name=test") {
		t.Error("expected output to contain function call")
	}
	if !strings.Contains(output, "FunctionResponse Name=test") {
		t.Error("expected output to contain function response")
	}
}

func TestErrorValues(t *testing.T) {
	// Test that error variables are properly defined
	if ErrNoContentInResponse == nil {
		t.Error("ErrNoContentInResponse should not be nil")
	}
	if ErrUnknownPartInResponse == nil {
		t.Error("ErrUnknownPartInResponse should not be nil")
	}
	if ErrInvalidMimeType == nil {
		t.Error("ErrInvalidMimeType should not be nil")
	}

	// Test error messages
	if !strings.Contains(ErrNoContentInResponse.Error(), "no content") {
		t.Error("ErrNoContentInResponse should mention 'no content'")
	}
	if !strings.Contains(ErrUnknownPartInResponse.Error(), "unknown part") {
		t.Error("ErrUnknownPartInResponse should mention 'unknown part'")
	}
	if !strings.Contains(ErrInvalidMimeType.Error(), "invalid mime") {
		t.Error("ErrInvalidMimeType should mention 'invalid mime'")
	}
}

func TestConstants(t *testing.T) {
	// Test that constants have expected values
	tests := []struct {
		name     string
		constant string
		expected string
	}{
		{"CITATIONS", CITATIONS, "citations"},
		{"SAFETY", SAFETY, "safety"},
		{"RoleSystem", RoleSystem, "system"},
		{"RoleModel", RoleModel, "model"},
		{"RoleUser", RoleUser, "user"},
		{"RoleTool", RoleTool, "tool"},
		{"ResponseMIMETypeJson", ResponseMIMETypeJson, "application/json"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.constant != tt.expected {
				t.Errorf("expected %s to be %q, got %q", tt.name, tt.expected, tt.constant)
			}
		})
	}
}

func TestConvertAndStreamFromIterator(t *testing.T) {
	// Skip actual implementation tests since we can't create a real iterator
	// These tests would need integration with the actual genai package
	t.Skip("Skipping iterator tests - requires real genai.GenerateContentResponseIterator")
}

// Test that Vertex implements llms.Model interface
func TestVertexImplementsModel(t *testing.T) {
	var _ llms.Model = &Vertex{}
}
