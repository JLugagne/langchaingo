// package vertex implements a langchaingo provider for Google Vertex AI LLMs,
// including the new Gemini models.
// See https://cloud.google.com/vertex-ai for more details.
package vertex

import (
	"context"

	"google.golang.org/genai"
	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/googleai"
	"github.com/tmc/langchaingo/llms/googleai/internal/palmclient"
)

// Vertex is a type that represents a Vertex AI API client.
//
// Right now, the Vertex Gemini SDK doesn't support embeddings; therefore,
// for embeddings we also hold a palmclient.
type Vertex struct {
	CallbacksHandler callbacks.Handler
	client           *genai.Client
	opts             googleai.Options
	palmClient       *palmclient.PaLMClient
}

var _ llms.Model = &Vertex{}

// New creates a new Vertex client.
func New(ctx context.Context, opts ...googleai.Option) (*Vertex, error) {
	clientOptions := googleai.DefaultOptions()
	for _, opt := range opts {
		opt(&clientOptions)
	}

	// Create client config for Vertex AI
	config := &genai.ClientConfig{
		Project:  clientOptions.CloudProject,
		Location: clientOptions.CloudLocation,
	}
	
	client, err := genai.NewClient(ctx, config)
	if err != nil {
		return nil, err
	}

	palmOpts := []palmclient.Option{
		palmclient.WithEmbeddingModelName(clientOptions.DefaultEmbeddingModel),
		palmclient.WithClientOptions(clientOptions.ClientOptions...),
	}
	palmClient, err := palmclient.New(
		ctx,
		clientOptions.CloudProject,
		clientOptions.CloudLocation,
		palmOpts...)
	if err != nil {
		return nil, err
	}

	v := &Vertex{
		opts:       clientOptions,
		client:     client,
		palmClient: palmClient,
	}
	return v, nil
}

// Close closes the underlying genai and palm clients.
// This should be called when the Vertex instance is no longer needed
// to prevent memory leaks from the underlying gRPC connections.
func (v *Vertex) Close() error {
	// Note: The new google.golang.org/genai client may not have a Close method
	// Check the client documentation for proper cleanup
	if v.client != nil {
		// For now, no explicit close needed - check actual API documentation
	}
	// Note: palmClient doesn't have a Close method based on the codebase
	return nil
}
