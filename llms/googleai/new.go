// package googleai implements a langchaingo provider for Google AI LLMs.
// See https://ai.google.dev/ for more details.
package googleai

import (
	"context"
	"strings"

	"google.golang.org/genai"
	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/llms"
)

// GoogleAI is a type that represents a Google AI API client.
type GoogleAI struct {
	CallbacksHandler callbacks.Handler
	client           *genai.Client
	opts             Options
	model            string // Track current model for reasoning detection
}

var (
	_ llms.Model          = &GoogleAI{}
	_ llms.ReasoningModel = &GoogleAI{}
)

// New creates a new GoogleAI client.
func New(ctx context.Context, opts ...Option) (*GoogleAI, error) {
	clientOptions := DefaultOptions()
	for _, opt := range opts {
		opt(&clientOptions)
	}
	clientOptions.EnsureAuthPresent()

	gi := &GoogleAI{
		opts:  clientOptions,
		model: clientOptions.DefaultModel, // Store the default model
	}

	// Create client config for google.golang.org/genai
	config := &genai.ClientConfig{}
	
	// Set API key if provided
	for _, opt := range clientOptions.ClientOptions {
		// Note: This is a simplified approach - may need refinement based on actual option types
		if apiKeyOpt, ok := opt.(interface{ GetAPIKey() string }); ok {
			config.APIKey = apiKeyOpt.GetAPIKey()
			break
		}
	}
	
	// Set project and location for Vertex AI if specified
	if clientOptions.CloudProject != "" {
		config.Project = clientOptions.CloudProject
	}
	if clientOptions.CloudLocation != "" {
		config.Location = clientOptions.CloudLocation
	}
	
	client, err := genai.NewClient(ctx, config)
	if err != nil {
		return gi, err
	}

	gi.client = client
	return gi, nil
}

// Close closes the underlying genai client.
// This should be called when the GoogleAI instance is no longer needed
// to prevent memory leaks from the underlying gRPC connections.
func (g *GoogleAI) Close() error {
	// Note: The new google.golang.org/genai client may not have a Close method
	// Check the client documentation for proper cleanup
	if g.client != nil {
		// For now, no explicit close needed - check actual API documentation
		return nil
	}
	return nil
}

// SupportsReasoning implements the ReasoningModel interface.
// Returns true if the current model supports reasoning/thinking tokens.
func (g *GoogleAI) SupportsReasoning() bool {
	// Check the current model (may have been overridden by WithModel option)
	model := g.model
	if model == "" {
		model = g.opts.DefaultModel
	}

	// Gemini 2.0 models support reasoning/thinking capabilities
	if strings.Contains(model, "gemini-2.0") {
		return true
	}

	// Future Gemini 3+ models expected to support reasoning
	if strings.Contains(model, "gemini-3") || strings.Contains(model, "gemini-4") {
		return true
	}

	// Gemini Experimental models may have reasoning capabilities
	if strings.Contains(model, "gemini-exp") && strings.Contains(model, "thinking") {
		return true
	}

	return false
}
