//nolint:all
package googleai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/tmc/langchaingo/internal/imageutil"
	"github.com/tmc/langchaingo/llms"
	"google.golang.org/genai"
)

var (
	ErrNoContentInResponse   = errors.New("no content in generation response")
	ErrUnknownPartInResponse = errors.New("unknown part type in generation response")
	ErrInvalidMimeType       = errors.New("invalid mime type on content")
)

const (
	CITATIONS            = "citations"
	SAFETY               = "safety"
	RoleSystem           = "system"
	RoleModel            = "model"
	RoleUser             = "user"
	RoleTool             = "tool"
	ResponseMIMETypeJson = "application/json"
)

// Call implements the [llms.Model] interface.
func (g *GoogleAI) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	return llms.GenerateFromSinglePrompt(ctx, g, prompt, options...)
}

// GenerateContent implements the [llms.Model] interface.
func (g *GoogleAI) GenerateContent(
	ctx context.Context,
	messages []llms.MessageContent,
	options ...llms.CallOption,
) (*llms.ContentResponse, error) {
	if g.CallbacksHandler != nil {
		g.CallbacksHandler.HandleLLMGenerateContentStart(ctx, messages)
	}

	opts := llms.CallOptions{
		Model:          g.opts.DefaultModel,
		CandidateCount: g.opts.DefaultCandidateCount,
		MaxTokens:      g.opts.DefaultMaxTokens,
		Temperature:    g.opts.DefaultTemperature,
		TopP:           g.opts.DefaultTopP,
		TopK:           g.opts.DefaultTopK,
	}
	for _, opt := range options {
		opt(&opts)
	}

	// Update the tracked model if it was overridden
	effectiveModel := opts.Model
	if effectiveModel != "" && effectiveModel != g.model {
		g.model = effectiveModel
	}

	// Convert messages to the new API format
	contents, systemInstruction, err := convertMessages(messages)
	if err != nil {
		return nil, err
	}

	// Create generation config
	tempFloat := float32(opts.Temperature)
	maxTokensInt := int32(opts.MaxTokens)
	topPFloat := float32(opts.TopP)
	topKFloat := float32(opts.TopK)

	config := &genai.GenerateContentConfig{
		Temperature:     &tempFloat,
		MaxOutputTokens: maxTokensInt,
		TopP:            &topPFloat,
		TopK:            &topKFloat,
	}

	// Set stop sequences if provided
	if len(opts.StopWords) > 0 {
		config.StopSequences = opts.StopWords
	}

	// Set JSON mode if requested
	if opts.JSONMode {
		config.ResponseMIMEType = ResponseMIMETypeJson
	}
	if opts.ResponseMIMEType != "" {
		if opts.JSONMode {
			return nil, fmt.Errorf("conflicting options, can't use JSONMode and ResponseMIMEType together")
		}
		config.ResponseMIMEType = opts.ResponseMIMEType
	}

	if g.opts.ResponseFormat != nil {
		config.ResponseJsonSchema = g.opts.ResponseFormat
		config.ResponseMIMEType = ResponseMIMETypeJson
	}

	// Set tools if provided
	if len(opts.Tools) > 0 {
		tools, err := convertTools(opts.Tools)
		if err != nil {
			return nil, err
		}
		config.Tools = tools
	}

	// Add system instruction to the first content if present
	allContents := contents
	if systemInstruction != nil {
		allContents = append([]*genai.Content{systemInstruction}, contents...)
	}

	// Generate content using the correct API
	resp, err := g.client.Models.GenerateContent(ctx, opts.Model, allContents, config)
	if err != nil {
		return nil, err
	}

	// Convert response back to langchain format
	response, err := convertResponse(resp)
	if err != nil {
		return nil, err
	}

	if g.CallbacksHandler != nil {
		g.CallbacksHandler.HandleLLMGenerateContentEnd(ctx, response)
	}

	return response, nil
}

// convertMessages converts langchain messages to genai format
func convertMessages(messages []llms.MessageContent) ([]*genai.Content, *genai.Content, error) {
	contents := []*genai.Content{}
	var systemInstruction *genai.Content

	for _, msg := range messages {
		parts := []*genai.Part{}
		for _, part := range msg.Parts {
			switch p := part.(type) {
			case llms.TextContent:
				parts = append(parts, &genai.Part{
					Text: p.Text,
				})
			case llms.BinaryContent:
				parts = append(parts, &genai.Part{
					InlineData: &genai.Blob{
						MIMEType: p.MIMEType,
						Data:     p.Data,
					},
				})
			case llms.ImageURLContent:
				// Download image and convert to binary
				typ, data, err := imageutil.DownloadImageData(p.URL)
				if err != nil {
					return nil, nil, err
				}
				parts = append(parts, &genai.Part{
					InlineData: &genai.Blob{
						MIMEType: typ,
						Data:     data,
					},
				})
			case llms.ToolCall:
				// Handle tool calls
				fc := p.FunctionCall
				var argsMap map[string]any
				if err := json.Unmarshal([]byte(fc.Arguments), &argsMap); err != nil {
					return nil, nil, err
				}
				parts = append(parts, &genai.Part{
					FunctionCall: &genai.FunctionCall{
						Name: fc.Name,
						Args: argsMap,
					},
				})
			case llms.ToolCallResponse:
				// Handle tool responses
				parts = append(parts, &genai.Part{
					FunctionResponse: &genai.FunctionResponse{
						Name: p.Name,
						Response: map[string]any{
							"response": p.Content,
						},
					},
				})
			}
		}

		content := &genai.Content{
			Parts: parts,
		}

		// Set role
		switch msg.Role {
		case llms.ChatMessageTypeSystem:
			content.Role = "user"
			systemInstruction = content
		case llms.ChatMessageTypeHuman:
			content.Role = "user"
			contents = append(contents, content)
		case llms.ChatMessageTypeAI:
			content.Role = "model"
			contents = append(contents, content)
		case llms.ChatMessageTypeGeneric:
			content.Role = "user"
			contents = append(contents, content)
		case llms.ChatMessageTypeTool:
			content.Role = "user"
			contents = append(contents, content)
		default:
			return nil, nil, fmt.Errorf("role %v not supported", msg.Role)
		}
	}

	return contents, systemInstruction, nil
}

// convertResponse converts genai response to langchain format
func convertResponse(resp *genai.GenerateContentResponse) (*llms.ContentResponse, error) {
	if resp == nil || len(resp.Candidates) == 0 {
		return nil, ErrNoContentInResponse
	}

	var contentResponse llms.ContentResponse

	for _, candidate := range resp.Candidates {
		buf := strings.Builder{}
		var toolCalls []llms.ToolCall

		if candidate.Content != nil {
			for _, part := range candidate.Content.Parts {
				if part != nil {
					// Handle text content
					if part.Text != "" {
						buf.WriteString(part.Text)
					}
					// Handle function calls
					if part.FunctionCall != nil {
						b, err := json.Marshal(part.FunctionCall.Args)
						if err != nil {
							return nil, err
						}
						toolCall := llms.ToolCall{
							FunctionCall: &llms.FunctionCall{
								Name:      part.FunctionCall.Name,
								Arguments: string(b),
							},
						}
						toolCalls = append(toolCalls, toolCall)
					}
				}
			}
		}

		metadata := make(map[string]any)
		metadata[CITATIONS] = candidate.CitationMetadata
		metadata[SAFETY] = candidate.SafetyRatings

		// Add usage information if available
		if resp.UsageMetadata != nil {
			metadata["input_tokens"] = resp.UsageMetadata.PromptTokenCount
			metadata["output_tokens"] = resp.UsageMetadata.CandidatesTokenCount
			metadata["total_tokens"] = resp.UsageMetadata.TotalTokenCount
			// Standardized field names
			metadata["PromptTokens"] = resp.UsageMetadata.PromptTokenCount
			metadata["CompletionTokens"] = resp.UsageMetadata.CandidatesTokenCount
			metadata["TotalTokens"] = resp.UsageMetadata.TotalTokenCount
		}

		// Standard fields
		metadata["ThinkingContent"] = ""
		metadata["ThinkingTokens"] = 0

		finishReason := "stop"
		finishReason = fmt.Sprintf("%v", candidate.FinishReason)

		contentResponse.Choices = append(contentResponse.Choices,
			&llms.ContentChoice{
				Content:        buf.String(),
				StopReason:     finishReason,
				GenerationInfo: metadata,
				ToolCalls:      toolCalls,
			})
	}

	return &contentResponse, nil
}

// convertCandidates converts a sequence of genai.Candidate to a response.
// This is a legacy function that may not be used with the new SDK structure
func convertCandidates(candidates []*genai.Candidate, usage *genai.UsageMetadata) (*llms.ContentResponse, error) {
	// For now, use the new convertResponse approach
	// This function is kept for compatibility but redirects to the new implementation
	return nil, fmt.Errorf("convertCandidates is deprecated, use convertResponse instead")
}

// convertParts is deprecated - use convertMessages instead
// This function is kept for compatibility but should not be used
func convertParts(parts []llms.ContentPart) ([]genai.Part, error) {
	return nil, fmt.Errorf("convertParts is deprecated, use convertMessages instead")
}

// convertContent is deprecated - use convertMessages instead
func convertContent(content llms.MessageContent) (*genai.Content, error) {
	return nil, fmt.Errorf("convertContent is deprecated, use convertMessages instead")
}

// generateFromSingleMessage generates content from the parts of a single
// message. (Legacy function - needs reimplementation for new SDK)
func generateFromSingleMessage(
	ctx context.Context,
	model interface{}, // Changed from *genai.GenerativeModel
	parts []llms.ContentPart,
	opts *llms.CallOptions,
) (*llms.ContentResponse, error) {
	return nil, fmt.Errorf("generateFromSingleMessage not implemented for new SDK")
}

func generateFromMessages(
	ctx context.Context,
	model interface{}, // Changed from *genai.GenerativeModel
	messages []llms.MessageContent,
	opts *llms.CallOptions,
) (*llms.ContentResponse, error) {
	return nil, fmt.Errorf("generateFromMessages not implemented for new SDK")
}

// convertAndStreamFromIterator takes an iterator of GenerateContentResponse
// and produces a llms.ContentResponse reply from it, while streaming the
// resulting text into the opts-provided streaming function.
// (Legacy function - needs reimplementation for new SDK)
func convertAndStreamFromIterator(
	ctx context.Context,
	iter interface{}, // Changed from *genai.GenerateContentResponseIterator
	opts *llms.CallOptions,
) (*llms.ContentResponse, error) {
	return nil, fmt.Errorf("convertAndStreamFromIterator not implemented for new SDK")
}

// convertSchemaRecursive recursively converts a schema map to a genai.Schema
func convertSchemaRecursive(schemaMap map[string]any, toolIndex int, propertyPath string) (*genai.Schema, error) {
	schema := &genai.Schema{}

	if ty, ok := schemaMap["type"]; ok {
		tyString, ok := ty.(string)
		if !ok {
			return nil, fmt.Errorf("tool [%d], property [%s]: expected string for type", toolIndex, propertyPath)
		}
		schema.Type = convertToolSchemaType(tyString)
	}

	if desc, ok := schemaMap["description"]; ok {
		descString, ok := desc.(string)
		if !ok {
			return nil, fmt.Errorf("tool [%d], property [%s]: expected string for description", toolIndex, propertyPath)
		}
		schema.Description = descString
	}

	// Handle object properties recursively
	if properties, ok := schemaMap["properties"]; ok {
		propMap, ok := properties.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("tool [%d], property [%s]: expected map for properties", toolIndex, propertyPath)
		}

		schema.Properties = make(map[string]*genai.Schema)
		for propName, propValue := range propMap {
			valueMap, ok := propValue.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("tool [%d], property [%s.%s]: expect to find a value map", toolIndex, propertyPath, propName)
			}

			nestedPath := propName
			if propertyPath != "" {
				nestedPath = propertyPath + "." + propName
			}

			nestedSchema, err := convertSchemaRecursive(valueMap, toolIndex, nestedPath)
			if err != nil {
				return nil, err
			}
			schema.Properties[propName] = nestedSchema
		}
	} else if schema.Type == genai.TypeObject && propertyPath == "" {
		// For top-level object schemas without properties, this is an error
		return nil, fmt.Errorf("tool [%d]: expected to find a map of properties", toolIndex)
	}

	// Handle array items recursively
	if items, ok := schemaMap["items"]; ok && schema.Type == genai.TypeArray {
		itemMap, ok := items.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("tool [%d], property [%s]: expect to find a map for array items", toolIndex, propertyPath)
		}

		itemsPath := propertyPath + "[]"
		itemsSchema, err := convertSchemaRecursive(itemMap, toolIndex, itemsPath)
		if err != nil {
			return nil, err
		}
		schema.Items = itemsSchema
	}

	// Handle required fields
	if required, ok := schemaMap["required"]; ok {
		if rs, ok := required.([]string); ok {
			schema.Required = rs
		} else if ri, ok := required.([]interface{}); ok {
			rs := make([]string, 0, len(ri))
			for _, r := range ri {
				rString, ok := r.(string)
				if !ok {
					return nil, fmt.Errorf("tool [%d], property [%s]: expected string for required", toolIndex, propertyPath)
				}
				rs = append(rs, rString)
			}
			schema.Required = rs
		} else {
			return nil, fmt.Errorf("tool [%d], property [%s]: expected array for required", toolIndex, propertyPath)
		}
	}

	return schema, nil
}

// convertTools converts from a list of langchaingo tools to a list of genai
// tools.
func convertTools(tools []llms.Tool) ([]*genai.Tool, error) {
	genaiFuncDecls := make([]*genai.FunctionDeclaration, 0, len(tools))
	for i, tool := range tools {
		if tool.Type != "function" {
			return nil, fmt.Errorf("tool [%d]: unsupported type %q, want 'function'", i, tool.Type)
		}

		// We have a llms.FunctionDefinition in tool.Function, and we have to
		// convert it to genai.FunctionDeclaration
		genaiFuncDecl := &genai.FunctionDeclaration{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
		}

		// Expect the Parameters field to be a map[string]any, from which we will
		// extract properties to populate the schema.
		params, ok := tool.Function.Parameters.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("tool [%d]: unsupported type %T of Parameters", i, tool.Function.Parameters)
		}

		schema, err := convertSchemaRecursive(params, i, "")
		if err != nil {
			return nil, err
		}
		genaiFuncDecl.Parameters = schema

		// google genai only support one tool, multiple tools must be embedded into function declarations:
		// https://github.com/GoogleCloudPlatform/generative-ai/issues/636
		// https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling#chat-samples
		genaiFuncDecls = append(genaiFuncDecls, genaiFuncDecl)
	}

	// Return nil if no tools are provided
	if len(genaiFuncDecls) == 0 {
		return nil, nil
	}

	genaiTools := []*genai.Tool{{FunctionDeclarations: genaiFuncDecls}}

	return genaiTools, nil
}

// convertToolSchemaType converts a tool's schema type from its langchaingo
// representation (string) to a genai enum.
func convertToolSchemaType(ty string) genai.Type {
	switch ty {
	case "object":
		return genai.TypeObject
	case "string":
		return genai.TypeString
	case "number":
		return genai.TypeNumber
	case "integer":
		return genai.TypeInteger
	case "boolean":
		return genai.TypeBoolean
	case "array":
		return genai.TypeArray
	default:
		return genai.TypeUnspecified
	}
}

// showContent is a debugging helper for genai.Content.
func showContent(w io.Writer, cs []*genai.Content) {
	fmt.Fprintf(w, "Content (len=%v)\n", len(cs))
	for i, c := range cs {
		fmt.Fprintf(w, "[%d]: Role=%s\n", i, c.Role)
		for j, p := range c.Parts {
			fmt.Fprintf(w, "  Parts[%v]: ", j)
			if p != nil {
				if p.Text != "" {
					fmt.Fprintf(w, "Text %q\n", p.Text)
				} else if p.InlineData != nil {
					fmt.Fprintf(w, "Blob MIME=%q, size=%d\n", p.InlineData.MIMEType, len(p.InlineData.Data))
				} else if p.FunctionCall != nil {
					fmt.Fprintf(w, "FunctionCall Name=%v, Args=%v\n", p.FunctionCall.Name, p.FunctionCall.Args)
				} else if p.FunctionResponse != nil {
					fmt.Fprintf(w, "FunctionResponse Name=%v Response=%v\n", p.FunctionResponse.Name, p.FunctionResponse.Response)
				} else {
					fmt.Fprintf(w, "Part type: %T\n", p)
				}
			} else {
				fmt.Fprintf(w, "nil part\n")
			}
		}
	}
}
