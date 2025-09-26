package googleai

import (
	"context"
	"fmt"

	"google.golang.org/genai"
)

// CreateEmbedding creates embeddings from texts.
func (g *GoogleAI) CreateEmbedding(ctx context.Context, texts []string) ([][]float32, error) {
	results := make([][]float32, 0, len(texts))

	// Convert texts to Content format
	contents := make([]*genai.Content, 0, len(texts))
	for _, text := range texts {
		contents = append(contents, &genai.Content{
			Parts: []*genai.Part{{
				Text: text,
			}},
		})
	}

	// Use EmbedContent method from the genai client
	resp, err := g.client.Models.EmbedContent(ctx, g.opts.DefaultEmbeddingModel, contents, &genai.EmbedContentConfig{})
	if err != nil {
		return nil, fmt.Errorf("failed to create embeddings: %w", err)
	}

	// Extract embeddings from response
	for _, embedding := range resp.Embeddings {
		results = append(results, embedding.Values)
	}

	return results, nil
}
