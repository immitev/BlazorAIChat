using Azure;
using Azure.AI.OpenAI;
using Azure.Search.Documents;
using Azure.Search.Documents.Indexes;
using Azure.Search.Documents.Indexes.Models;
using Azure.Search.Documents.Models;
using BlazorAIChat.Models;
using Microsoft.Extensions.Options;

namespace BlazorAIChat.Services
{
    public class AISearchService
    {
        private readonly AppSettings settings;
        private readonly ILogger<AISearchService> logger;

        private AzureOpenAIClient? azureOpenAIClient;
        private SearchIndexClient? searchIndexClient;
        private SearchIndexerClient? searchIndexerClient;
        private SearchClient? searchClient;
        public bool IsReady { get; private set; } = false;

        public AISearchService(IOptions<AppSettings> appSettings,ILogger<AISearchService> logger)
        {

            settings = appSettings.Value;
            this.logger = logger;

            //Check to see if we are configured to use AI Search
            if (settings.UsesAzureAISearchSharedKnowledge==false)
            {
                logger.LogInformation("Application not configured to use Azure AI Search for shared index");
                return;
            }

            //configure clients
            azureOpenAIClient = new AzureOpenAIClient(new Uri(settings.AzureOpenAIChatCompletion.Endpoint), new AzureKeyCredential(settings.AzureOpenAIChatCompletion.ApiKey));
            searchIndexClient = new SearchIndexClient(new Uri(settings.AzureAISearch.Endpoint), new AzureKeyCredential(settings.AzureAISearch.ApiKey));
            searchIndexerClient = new SearchIndexerClient(new Uri(settings.AzureAISearch.Endpoint), new AzureKeyCredential(settings.AzureAISearch.ApiKey));
            searchClient = searchIndexClient.GetSearchClient(settings.AzureAISearch.SharedIndex);

            //Setup Azure AI Search only if we are using a shared index
            if (settings.AzureAISearch.SharedIndex != null)
            {
                // Fire and forget async call, log any exceptions
                _ = Task.Run(async () =>
                {
                    try
                    {
                        await SetupAndRunIndexer();
                    }
                    catch (Exception ex)
                    {
                        logger.LogError(ex, "Error running SetupAndRunIndexer in AISearchService constructor.");
                    }
                });
            }
        }

        public async Task SetupAndRunIndexer()
        {
            IsReady = false;

            //Setup index
            logger.LogInformation("Setting up Azure AI Search index");
            logger.LogInformation($"Creating / Updating index {settings.AzureAISearch.SharedIndex}");
            var index = GetSampleIndex();
            await searchIndexClient.CreateOrUpdateIndexAsync(index);
            logger.LogInformation("Index created / updated");

            //Setup data source
            logger.LogInformation("Creating / updating data source connection for Azure AI Search");
            var dataSource = new SearchIndexerDataSourceConnection(
                $"{settings.AzureAISearch.SharedIndex}-blob",
                SearchIndexerDataSourceType.AzureBlob,
                connectionString: settings.AzureAISearch.SharedIndexAzureBlobStorageConnection,
                container: new SearchIndexerDataContainer(settings.AzureAISearch.SharedIndexAzureBlobStorageContainer));

            await searchIndexerClient.CreateOrUpdateDataSourceConnectionAsync(dataSource);
            logger.LogInformation($"Data source created / updated for Azure AI Search");

            //Setup Skillset
            var skillset = new SearchIndexerSkillset($"{settings.AzureAISearch.SharedIndex}-skillset", new List<SearchIndexerSkill>
            {  
                // Add required skills here    
                new SplitSkill(
                    new List<InputFieldMappingEntry>
                    {
                        new InputFieldMappingEntry("text") { Source = "/document/content" }
                    },
                    new List<OutputFieldMappingEntry>
                    {
                        new OutputFieldMappingEntry("textItems") { TargetName = "pages" }
                    })
                {
                    Context = "/document",
                    TextSplitMode = TextSplitMode.Pages,
                    MaximumPageLength = 2000,
                    PageOverlapLength = 500,
                },
                new AzureOpenAIEmbeddingSkill(
                    new List<InputFieldMappingEntry>
                    {
                        new InputFieldMappingEntry("text") { Source = "/document/pages/*" }
                    },
                    new List<OutputFieldMappingEntry>
                    {
                        new OutputFieldMappingEntry("embedding") { TargetName = "vector" }
                    }
                )
                {
                    Context = "/document/pages/*",
                    ResourceUri = new Uri(settings.AzureOpenAIChatCompletion.Endpoint),
                    ApiKey = settings.AzureOpenAIChatCompletion.ApiKey,
                    DeploymentName = settings.AzureOpenAIEmbedding.DeploymentName,
                    ModelName = settings.AzureOpenAIEmbedding.DeploymentName
                }
            })
            {

                IndexProjection = new SearchIndexerIndexProjection(new[]
                {
                    new SearchIndexerIndexProjectionSelector(settings.AzureAISearch.SharedIndex, parentKeyFieldName: "parent_id", sourceContext: "/document/pages/*", mappings: new[]
                    {
                        new InputFieldMappingEntry("chunk")
                        {
                            Source = "/document/pages/*"
                        },
                        new InputFieldMappingEntry("vector")
                        {
                            Source = "/document/pages/*/vector"
                        },
                        new InputFieldMappingEntry("title")
                        {
                            Source = "/document/metadata_storage_name"
                        }
                    })
                })
                {
                    Parameters = new SearchIndexerIndexProjectionsParameters
                    {
                        ProjectionMode = IndexProjectionMode.SkipIndexingParentDocuments
                    }
                }
            };
            await searchIndexerClient.CreateOrUpdateSkillsetAsync(skillset).ConfigureAwait(false);
            logger.LogInformation("Skillset created / updated for Azure AI Search.");

            //create an indexer
            var indexer = new SearchIndexer($"{settings.AzureAISearch.SharedIndex}-indexer", dataSource.Name, settings.AzureAISearch.SharedIndex)
            {
                Description = "Indexer to chunk documents, generate embeddings, and add to the index",
                Schedule = new IndexingSchedule(TimeSpan.FromDays(1))
                {
                    StartTime = DateTimeOffset.Now
                },
                Parameters = new IndexingParameters()
                {
                    BatchSize = 1,
                    MaxFailedItems = 0,
                    MaxFailedItemsPerBatch = 0,
                },
                SkillsetName = skillset.Name,
            };
            await searchIndexerClient.CreateOrUpdateIndexerAsync(indexer).ConfigureAwait(false);
            logger.LogInformation("Indexer created for Azure AI Search");

            //Run Indexer
            logger.LogInformation("Starting the Azure AI Search indexer");
            await searchIndexerClient.RunIndexerAsync(indexer.Name).ConfigureAwait(false);
            logger.LogInformation("Azure AI Search indexer is running");

            IsReady = true;
        }


        private SearchIndex GetSampleIndex()
        {
            const string vectorSearchHnswProfile = "my-vector-profile";
            const string vectorSearchExhaustiveKnnProfile = "myExhaustiveKnnProfile";
            const string vectorSearchHnswConfig = "myHnsw";
            const string vectorSearchExhaustiveKnnConfig = "myExhaustiveKnn";
            const string vectorSearchVectorizer = "myOpenAIVectorizer";
            const string semanticSearchConfig = "my-semantic-config";
            const int modelDimensions = 1536;

            SearchIndex searchIndex = new(settings.AzureAISearch.SharedIndex)
            {
                VectorSearch = new()
                {
                    Profiles =
                    {
                        new VectorSearchProfile(vectorSearchHnswProfile, vectorSearchHnswConfig)
                        {
                            VectorizerName = vectorSearchVectorizer
                        },
                        new VectorSearchProfile(vectorSearchExhaustiveKnnProfile, vectorSearchExhaustiveKnnConfig)
                    },
                    Algorithms =
                    {
                        new HnswAlgorithmConfiguration(vectorSearchHnswConfig),
                        new ExhaustiveKnnAlgorithmConfiguration(vectorSearchExhaustiveKnnConfig)
                    },
                    Vectorizers =
                    {
                        new AzureOpenAIVectorizer(vectorSearchVectorizer)
                        {

                            Parameters = new AzureOpenAIVectorizerParameters()
                            {
                                ResourceUri = new Uri(settings.AzureOpenAIChatCompletion.Endpoint),
                                ApiKey = settings.AzureOpenAIChatCompletion.ApiKey,
                                DeploymentName = settings.AzureOpenAIEmbedding.DeploymentName,
                                ModelName = settings.AzureOpenAIEmbedding.DeploymentName
                            }
                        }
                    }
                },
                SemanticSearch = new()
                {
                    Configurations =
                    {
                        new SemanticConfiguration(semanticSearchConfig, new()
                        {
                            TitleField = new SemanticField(fieldName: "title"),
                            ContentFields =
                            {
                                new SemanticField(fieldName: "chunk")
                            },
                        })
                    },
                },
                Fields =
                {
                    new SearchableField("parent_id") { IsFilterable = true, IsSortable = true, IsFacetable = true },
                    new SearchableField("chunk_id") { IsKey = true, IsFilterable = true, IsSortable = true, IsFacetable = true, AnalyzerName = LexicalAnalyzerName.Keyword },
                    new SearchableField("title"),
                    new SearchableField("chunk"),
                    new SearchField("vector", SearchFieldDataType.Collection(SearchFieldDataType.Single))
                    {
                        IsSearchable = true,
                        VectorSearchDimensions = modelDimensions,
                        VectorSearchProfileName = vectorSearchHnswProfile
                    },
                    new SearchableField("category") { IsFilterable = true, IsSortable = true, IsFacetable = true },
                },
            };

            return searchIndex;
        }

        public async Task<List<(double? score, string? title, string? chunk)>> Search(string query, int resultsCount = 3, string? filter = null, bool textOnly = false, bool exhaustive = false, bool hybrid = false, bool semantic = false)
        {
            if (!IsReady)
            {
                logger.LogWarning("Azure AI Search Indexer is not configured or ready.");
                return new List<(double? score, string? title, string? chunk)>();
            }

            logger.LogInformation($"Starting search for {query}");
            // Perform the vector similarity search  
            var searchOptions = new Azure.Search.Documents.SearchOptions
            {
                Filter = filter,
                Size = resultsCount,
                Select = { "title", "chunk_id", "chunk", },
                IncludeTotalCount = true
            };
            if (!textOnly)
            {
                searchOptions.VectorSearch = new()
                {
                    Queries = {
                        new VectorizableTextQuery(text: query)
                        {
                            KNearestNeighborsCount = resultsCount,
                            Fields = { "vector" },
                            Exhaustive = exhaustive
                        }
                    },

                };
            }
            if (semantic)
            {
                searchOptions.QueryType = SearchQueryType.Semantic;
                searchOptions.SemanticSearch = new SemanticSearchOptions
                {
                    SemanticConfigurationName = "my-semantic-config",
                    QueryCaption = new QueryCaption(QueryCaptionType.Extractive),
                    QueryAnswer = new QueryAnswer(QueryAnswerType.Extractive)
                };
            }
            string? queryText = (textOnly || hybrid || semantic) ? query : null;
            SearchResults<SearchDocument> response = await searchClient.SearchAsync<SearchDocument>(queryText, searchOptions);
            List<(double? score, string? title, string? chunk)> results = new List<(double? score, string? title, string? chunk)>();

            //If we have semantic search results, we return the answers. 
            if (response.SemanticSearch?.Answers?.Count > 0)
            {
                foreach (QueryAnswerResult answer in response.SemanticSearch.Answers)
                {
                    results.Add((null, answer.Highlights, answer.Text));
                }
                return results;
            }

            //If we don't use semantic search, we return the text chunks 
            await foreach (SearchResult<SearchDocument> result in response.GetResultsAsync())
            {
                //add score, title and chunk to results list
                results.Add((result.Score, result.Document["title"].ToString(), result.Document["chunk"].ToString()));
            }

            logger.LogInformation($"Total Search Results: {response.TotalCount}");
            return results;
        }

    }
}
