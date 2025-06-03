using BlazorAIChat.Models;
using BlazorAIChat.Utils;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Options;
using Microsoft.KernelMemory;
using Microsoft.KernelMemory.AI;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.AzureOpenAI;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using System.Text;
using System.Text.RegularExpressions;

namespace BlazorAIChat.Services
{
    /// <summary>
    /// The AIService class provides various AI-related functionalities such as chat completion, document processing, and session summarization.
    /// </summary>
    public class AIService
    {
#pragma warning disable SKEXP0010, SKEXP0001, SKEXP0020, KMEXP00
        private readonly AppSettings settings;
        private readonly MemoryServerless? kernelMemory;
        private readonly HttpClient? httpClient;
        private readonly ITextTokenizer? chatCompletionTokenizer;
        private readonly ITextTokenizer? embeddingTokenizer;
        private readonly IChatCompletionService? chatCompletionService;
        private readonly IEmbeddingGenerator? textEmbeddingGenerationService;
        public ChatHistory history { get; private set; } = new ChatHistory();
        private readonly AIChatDBContext dbContext;
        private readonly ChatCompletionAgent sessionSummaryAgent;
        private readonly ChatCompletionAgent promptRewriteAgent;
        private readonly ChatCompletionAgent ragDecisionAgent;
        private readonly ChatHistoryService chatHistoryService;
        private readonly ILogger<AIService> logger;
        private readonly Kernel kernel;

        /// <summary>
        /// Initializes a new instance of the <see cref="AIService"/> class.
        /// </summary>
        /// <param name="appSettings">The application settings.</param>
        /// <param name="chatHistoryService">The chat history service.</param>
        /// <param name="httpClientFactory">The HTTP client factory.</param>
        /// <param name="dbContext">The database context.</param>
        /// <param name="kernel">The Semantic Kernel instance.</param>
        public AIService(IOptions<AppSettings> appSettings, ChatHistoryService chatHistoryService, IHttpClientFactory httpClientFactory, AIChatDBContext dbContext, ILogger<AIService> logger, Kernel kernel)
        {
            this.dbContext = dbContext;
            this.chatHistoryService = chatHistoryService;
            this.logger = logger;
            this.kernel = kernel;
            settings = appSettings.Value;
            httpClient = httpClientFactory.CreateClient("retryHttpClient");
            chatCompletionTokenizer = TokenizerFactory.GetTokenizerForModel(settings.AzureOpenAIChatCompletion.Tokenizer);
            embeddingTokenizer = TokenizerFactory.GetTokenizerForModel(settings.AzureOpenAIEmbedding.Tokenizer);

            logger.LogInformation("Initializing AIService with settings");

            chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();
            textEmbeddingGenerationService = kernel.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();

            // Initialize agents
            (sessionSummaryAgent, promptRewriteAgent, ragDecisionAgent) = InitializeAgents();

            // Initialize Kernel Memory
            kernelMemory = InitializeKernelMemory();
        }

        private (ChatCompletionAgent sessionSummaryAgent, ChatCompletionAgent promptRewriteAgent, ChatCompletionAgent ragDecisionAgent) InitializeAgents()
        {
            var sessionSummaryAgent = new ChatCompletionAgent
            {
                Name = "SessionSummaryAgent",
                Kernel = kernel,
                Instructions = """
                    Summarize this text. One to three words maximum length.
                    Plain text only. No punctuation, markup or tags.
                """,
                Arguments = new KernelArguments(
                    new OpenAIPromptExecutionSettings()
                    {
                        Temperature = 0
                    })
            };

            var promptRewriteAgent = new ChatCompletionAgent
            {
                Name = "PromptRewriteAgent",
                Kernel = kernel,
                Instructions = """
                    Rewrite the user prompt to remove all URLs but still make the question or request understandable.
                """,
                Arguments = new KernelArguments(
                    new OpenAIPromptExecutionSettings()
                    {
                        Temperature = 0.3f
                    })
            };

            var ragDecisionAgent = new ChatCompletionAgent
            {
                Name = "RAGDecisionAgent",
                Kernel = kernel,
                Instructions = $"""
                    You are an AI assistant that determines whether a query requires knowledge retrieval.
                    
                    Analyze the user's query and respond with 'true' if the query likely needs to retrieve information from stored knowledge, or 'false' if the query can be answered using the available tools.
                    If the query contains URLs, it should always return 'true'.
                    If the query can be answered with both the tools and knowledge retrieval, it should return 'true'.
                    If you are unsure, return 'true'.
                                      
                    Respond only with 'true' or 'false'.
                """,
                Arguments = new KernelArguments(
                    new OpenAIPromptExecutionSettings()
                    {
                        Temperature = 0,
                        MaxTokens = 10,
                        ToolCallBehavior = ToolCallBehavior.EnableKernelFunctions
                    })
            };

            return (sessionSummaryAgent, promptRewriteAgent, ragDecisionAgent);
        }

        private MemoryServerless? InitializeKernelMemory()
        {
            ArgumentNullException.ThrowIfNull(textEmbeddingGenerationService);

            var kernelMemoryBuilder = new KernelMemoryBuilder()
                .WithAzureOpenAITextEmbeddingGeneration(new AzureOpenAIConfig() { Auth = AzureOpenAIConfig.AuthTypes.APIKey, APIKey = settings.AzureOpenAIChatCompletion.ApiKey, Deployment = settings.AzureOpenAIEmbedding.DeploymentName, Endpoint = settings.AzureOpenAIChatCompletion.Endpoint }, embeddingTokenizer, httpClient: httpClient)
                .WithAzureOpenAITextGeneration(new AzureOpenAIConfig() { Auth = AzureOpenAIConfig.AuthTypes.APIKey, APIKey = settings.AzureOpenAIChatCompletion.ApiKey, Deployment = settings.AzureOpenAIChatCompletion.DeploymentName, Endpoint = settings.AzureOpenAIChatCompletion.Endpoint }, chatCompletionTokenizer, httpClient);

            if (settings.UsesAzureAISearch)
            {
                kernelMemoryBuilder = kernelMemoryBuilder.WithAzureAISearchMemoryDb(new AzureAISearchConfig()
                {
                    Endpoint = settings.AzureAISearch.Endpoint,
                    APIKey = settings.AzureAISearch.ApiKey,
                    Auth = AzureAISearchConfig.AuthTypes.APIKey,
                    UseHybridSearch = false
                });
            }
            else if (settings.UsesPostgreSQL)
            {
                kernelMemoryBuilder = kernelMemoryBuilder.WithPostgresMemoryDb(new PostgresConfig()
                {
                    ConnectionString = settings.ConnectionStrings.PostgreSQL
                });
            }
            else
            {
                kernelMemoryBuilder = kernelMemoryBuilder
                    .WithSimpleVectorDb(new Microsoft.KernelMemory.MemoryStorage.DevTools.SimpleVectorDbConfig { StorageType = Microsoft.KernelMemory.FileSystem.DevTools.FileSystemTypes.Disk, Directory = "KNN" })
                    .WithSimpleFileStorage(new Microsoft.KernelMemory.DocumentStorage.DevTools.SimpleFileStorageConfig() { StorageType = Microsoft.KernelMemory.FileSystem.DevTools.FileSystemTypes.Disk, Directory = "SFS" });
            }

            if (!settings.AzureOpenAIChatCompletion.SupportsImages && settings.UsesAzureDocIntelligence)
            {
                kernelMemoryBuilder = kernelMemoryBuilder.WithAzureAIDocIntel(new AzureAIDocIntelConfig()
                {
                    Endpoint = settings.DocumentIntelligence.Endpoint,
                    APIKey = settings.DocumentIntelligence.ApiKey,
                    Auth = AzureAIDocIntelConfig.AuthTypes.APIKey
                });
            }

            logger.LogInformation("Kernel Memory initialized successfully.");
            return kernelMemoryBuilder.Build<MemoryServerless>();
        }

        /// <summary>
        /// Processes a list of URLs with kernel memory.
        /// </summary>
        /// <param name="urls">The list of URLs to process.</param>
        /// <param name="currentSession">The current session.</param>
        /// <param name="currentUser">The current user.</param>
        private async Task ProcessURLsWithKernelMemory(List<string> urls, Session? currentSession, User currentUser)
        {
            if (kernelMemory != null && currentSession != null)
            {
                string index = currentSession.SessionId;
                TagCollection tags = new TagCollection { { "user", currentUser.Id } };
                foreach (var url in urls)
                {
                    // Check if the document already exists in the database
                    var doc = dbContext.SessionDocuments.FirstOrDefault(d => d.FileNameOrUrl == url && d.SessionId == index);
                    if (doc == null)
                    {
                        var docId = Guid.NewGuid().ToString();
                        // Import the web page into kernel memory
                        await kernelMemory.ImportWebPageAsync(url, docId, tags, index).ConfigureAwait(false);
                        // Wait until the document is ready
                        while (!await kernelMemory.IsDocumentReadyAsync(docId, index).ConfigureAwait(false))
                        {
                            await Task.Delay(500).ConfigureAwait(false);
                        }
                        // Add the document to the database
                        dbContext.SessionDocuments.Add(new SessionDocument() { DocId = docId, FileNameOrUrl = url, SessionId = index });
                        await dbContext.SaveChangesAsync().ConfigureAwait(false);
                    }
                }
            }
        }

        /// <summary>
        /// Gets the chat response asynchronously.
        /// </summary>
        /// <param name="prompt">The prompt for the chat.</param>
        /// <param name="message">The message object.</param>
        /// <param name="currentSession">The current session.</param>
        /// <param name="currentUser">The current user.</param>
        /// <returns>An asynchronous enumerable of streaming chat message content.</returns>
        public async Task<IAsyncEnumerable<List<StreamingChatMessageContent>>> GetChatResponseAsync(string prompt, Message message, Session currentSession, User currentUser)
        {
            logger.LogInformation("Getting chat response for session: {SessionId}, user: {UserId}", currentSession.SessionId, currentUser.Id);
            ArgumentNullException.ThrowIfNull(chatCompletionTokenizer);
            ArgumentNullException.ThrowIfNull(chatCompletionService);

            // Conditionally perform RAG only if needed
            bool ragNeeded = await ShouldUseRAGForPrompt(prompt, currentSession.SessionId);
            if (ragNeeded)
            {
                logger.LogInformation("RAG determined to be needed, performing document retrieval");
                await DoRAG(prompt, message, currentSession, currentUser).ConfigureAwait(false);
            }
            else
            {
                logger.LogInformation("RAG determined not needed, proceeding with direct completion");
                // Just add the prompt to history without RAG
                history.AddUserMessage(prompt);
            }

            // Clean up the chat history to fit within the token limit
            history = AIUtils.CleanUpHistory(history, chatCompletionTokenizer, settings.AzureOpenAIChatCompletion.MaxInputTokens);

            IAsyncEnumerable<StreamingChatMessageContent> streamingMessages;

            // Get the chat response as a stream of messages
            AzureOpenAIPromptExecutionSettings executionSettings = new()
            {
                Temperature = 0,
                FunctionChoiceBehavior = FunctionChoiceBehavior.Auto(options: new() { RetainArgumentTypes = true })
            };

            streamingMessages = chatCompletionService.GetStreamingChatMessageContentsAsync(history, executionSettings, kernel);

            // Buffer and yield messages in chunks
            logger.LogInformation("Chat response generation completed for session: {SessionId}", currentSession.SessionId);
            return BufferMessagesInChunks(streamingMessages, settings.AzureOpenAIChatCompletion.ResponseChunkSize);
        }

        /// <summary>
        /// Determines whether RAG should be used for the given prompt
        /// </summary>
        /// <param name="prompt">The user's prompt</param>
        /// <param name="sessionId">The current session ID</param>
        /// <returns>True if RAG should be used, false otherwise</returns>
        private async Task<bool> ShouldUseRAGForPrompt(string prompt, string sessionId)
        {
            // Always use RAG if URLs are present in the prompt
            if (StringUtils.GetURLsFromString(prompt).Count > 0)
            {
                return true;
            }

            // Check if there are any documents in this session
            bool sessionHasDocuments = dbContext.SessionDocuments.Any(d => d.SessionId == sessionId);
            if (!sessionHasDocuments)
            {
                // If no documents in session, don't bother with RAG
                return false;
            }

            // Use ChatCompletionAgent to analyze if retrieval is needed
            ChatHistory analysisHistory = new();
            analysisHistory.AddUserMessage(prompt);
            StringBuilder output = new();
            Microsoft.SemanticKernel.Agents.AgentThread? agentThread = null;
            await foreach (var response in ragDecisionAgent.InvokeAsync(analysisHistory, agentThread).ConfigureAwait(false))
            {
                output.Append(response.Message.ToString());
                agentThread = response.Thread;
            }
            var completionText = output.ToString().Trim().ToLower();
            if (agentThread is not null)
            {
                await agentThread.DeleteAsync();
            }
            return completionText.Contains("true");
        }

        private async IAsyncEnumerable<List<StreamingChatMessageContent>> BufferMessagesInChunks(IAsyncEnumerable<StreamingChatMessageContent> streamingMessages, int chunkSize)
        {
            List<StreamingChatMessageContent> buffer = new List<StreamingChatMessageContent>(chunkSize);

            await foreach (var message in streamingMessages)
            {
                buffer.Add(message);

                if (buffer.Count >= chunkSize)
                {
                    yield return new List<StreamingChatMessageContent>(buffer);
                    buffer.Clear();
                }
            }

            // Yield any remaining messages
            if (buffer.Count > 0)
            {
                yield return new List<StreamingChatMessageContent>(buffer);
            }
        }

        /// <summary>
        /// Performs Retrieval-Augmented Generation (RAG) on the given prompt.
        /// </summary>
        /// <param name="prompt">The prompt for the chat.</param>
        /// <param name="message">The message object.</param>
        /// <param name="currentSession">The current session.</param>
        /// <param name="currentUser">The current user.</param>
        private async Task DoRAG(string prompt, Message message, Session currentSession, User currentUser)
        {
            logger.LogInformation("Performing RAG for session: {SessionId}, user: {UserId}", currentSession.SessionId, currentUser.Id);
            var urls = StringUtils.GetURLsFromString(prompt);
            SearchResult? searchData = null;
            if (urls.Count > 0)
            {
                // Process URLs with kernel memory
                await ProcessURLsWithKernelMemory(urls, currentSession, currentUser).ConfigureAwait(false);

                // Generate a new prompt without URLs
                string messageToProcessNoURLS = await GenerateNewPromptForMessagesWithUrl(prompt).ConfigureAwait(false);

                // Search the kernel memory with the new prompt
                searchData = await kernelMemory!.SearchAsync(messageToProcessNoURLS, currentSession.Id, minRelevance: 0.5, limit: 5).ConfigureAwait(false);
            }
            else
            {
                // Search the kernel memory with the original prompt
                searchData = await kernelMemory!.SearchAsync(prompt, currentSession.Id, minRelevance: 0.5, limit: 5).ConfigureAwait(false);
            }

            if (searchData != null && !searchData.NoResult)
            {
                if (searchData.Results.Count > 0)
                {
                    history.AddUserMessage("Below are facts related to the question. Use supplied tools if required to fully answer the request without asking them for more information.");
                    foreach (var result in searchData.Results)
                    {
                        foreach (var p in result.Partitions)
                        {
                            // Add the search result text to the chat history
                            history.AddUserMessage(p.Text);
                        }

                        // Add citations to the message
                        if (result.SourceName.ToLower() == "content.url")
                            message.Citations.Add($"{result.SourceUrl}");
                        else
                            message.Citations.Add($"{result.SourceName}");
                    }
                    history.AddUserMessage("----------------------------------");
                }
            }

            // Add the original prompt to the chat history
            history.AddUserMessage(prompt);
            logger.LogInformation("RAG completed for session: {SessionId}", currentSession.SessionId);
        }

        /// <summary>
        /// Generates a new prompt by removing URLs from the given message.
        /// </summary>
        /// <param name="theMessage">The message containing URLs.</param>
        /// <returns>The new prompt without URLs.</returns>
        private async Task<string> GenerateNewPromptForMessagesWithUrl(string theMessage)
        {

            // Create a chat history with the entire conversation
            ChatHistory sessionSummary = new() { new ChatMessageContent(AuthorRole.User, theMessage) };
            StringBuilder output = new();

            Microsoft.SemanticKernel.Agents.AgentThread? agentThread = null;
            // Get the updated prompt from the prompt rewrite agent
            await foreach (var response in promptRewriteAgent.InvokeAsync(sessionSummary, agentThread).ConfigureAwait(false))
            {
                output.Append(response.Message.ToString());
                agentThread = response.Thread;
            }
            var completionText = output.ToString();

            // Delete the thread if required.
            if (agentThread is not null)
            {
                await agentThread.DeleteAsync();
            }

            return completionText;
        }

        /// <summary>
        /// Summarizes the chat session name asynchronously.
        /// </summary>
        /// <param name="sessionId">The session identifier.</param>
        /// <returns>The summarized session name.</returns>
        public async Task<string> SummarizeChatSessionNameAsync(string? sessionId)
        {
            logger.LogInformation("Summarizing chat session name for session: {SessionId}", sessionId);
            ArgumentNullException.ThrowIfNull(sessionId);
            ArgumentNullException.ThrowIfNull(chatCompletionService);

            // Get all messages from the session
            var messages = await chatHistoryService.GetSessionMessagesAsync(sessionId).ConfigureAwait(false);
            var conversationText = string.Join(" ", messages.Select(m => m.Prompt + " " + m.Completion));

            //Strip base64 encoded content from the conversation text. We don't need to send all of that to the agent.
            string pattern = @"data:image\/[a-zA-Z]+;base64,[^\s]+";
            conversationText = Regex.Replace(conversationText, pattern, string.Empty);

            // Create a chat history with the entire conversation
            ChatHistory sessionSummary = new() { new ChatMessageContent(AuthorRole.User, conversationText) };
            StringBuilder output = new();
            // Get the summary from the session summary agent
            Microsoft.SemanticKernel.Agents.AgentThread? agentThread = null;
            await foreach (var response in sessionSummaryAgent.InvokeAsync(sessionSummary, agentThread).ConfigureAwait(false))
            {
                output.Append(response.Message.ToString());
                agentThread = response.Thread;
            }
            var completionText = output.ToString();

            // Delete the thread if required.
            if (agentThread is not null)
            {
                await agentThread.DeleteAsync();
            }

            // Update the session name with the summary
            var session = await chatHistoryService.GetSessionAsync(sessionId).ConfigureAwait(false);
            session.Name = completionText;
            await chatHistoryService.UpdateSessionAsync(session).ConfigureAwait(false);
            logger.LogInformation("Chat session name summarized for session: {SessionId}", sessionId);
            return completionText;
        }

        /// <summary>
        /// Processes documents with kernel memory.
        /// </summary>
        /// <param name="memoryStream">The memory stream of the document.</param>
        /// <param name="filename">The filename of the document.</param>
        /// <param name="currentSession">The current session.</param>
        /// <param name="currentUser">The current user.</param>
        /// <returns>A boolean indicating whether the document was processed successfully.</returns>
        public async Task<bool> ProcessDocsWithKernelMemory(MemoryStream memoryStream, string filename, Session currentSession, User currentUser)
        {
            logger.LogInformation("Processing document: {Filename} for session: {SessionId}, user: {UserId}", filename, currentSession.SessionId, currentUser.Id);
            ArgumentNullException.ThrowIfNull(kernelMemory);

            // Check if the document already exists in the database
            var doc = dbContext.SessionDocuments.FirstOrDefault(d => d.FileNameOrUrl == filename && d.SessionId == currentSession.SessionId);
            if (doc != null)
            {
                return false;
            }

            var docId = Guid.NewGuid().ToString();
            string index = currentSession.SessionId;
            TagCollection tags = new TagCollection { { "user", currentUser.Id } };
            memoryStream.Position = 0;

            // Import the document into kernel memory
            await kernelMemory.ImportDocumentAsync(memoryStream, filename, docId, tags, index).ConfigureAwait(false);

            // Add the document to the database
            dbContext.SessionDocuments.Add(new SessionDocument() { DocId = docId, FileNameOrUrl = filename, SessionId = index });
            await dbContext.SaveChangesAsync().ConfigureAwait(false);

            // Wait until the document is ready
            while (!await kernelMemory.IsDocumentReadyAsync(docId, index).ConfigureAwait(false))
            {
                await Task.Delay(1000).ConfigureAwait(false);
            }

            logger.LogInformation("Document processed successfully: {Filename} for session: {SessionId}", filename, currentSession.SessionId);
            return true;
        }

        /// <summary>
        /// Deletes uploaded documents for a given session.
        /// </summary>
        /// <param name="sessionIdToDelete">The session identifier to delete documents for.</param>
        /// <returns>A boolean indicating whether the documents were deleted successfully.</returns>
        public async Task<bool> DeleteUploadedDocs(string? sessionIdToDelete)
        {
            logger.LogInformation("Deleting uploaded documents for session: {SessionId}", sessionIdToDelete);
            if (!string.IsNullOrEmpty(sessionIdToDelete))
            {
                if (kernelMemory != null)
                    await kernelMemory.DeleteIndexAsync(sessionIdToDelete).ConfigureAwait(false);

                // Remove documents from the database
                var docs = dbContext.SessionDocuments.Where(d => d.SessionId == sessionIdToDelete);
                dbContext.SessionDocuments.RemoveRange(docs);
                await dbContext.SaveChangesAsync().ConfigureAwait(false);

                logger.LogInformation("Uploaded documents deleted for session: {SessionId}", sessionIdToDelete);
                return true;
            }
            return false;
        }

        /// <summary>
        /// Adds an image to the chat.
        /// </summary>
        /// <param name="imageStream">The memory stream of the image.</param>
        /// <param name="uploadImageType">The type of the uploaded image.</param>
        /// <returns>A boolean indicating whether the image was added successfully.</returns>
        public bool AddImageToChat(MemoryStream imageStream, string uploadImageType)
        {
            logger.LogInformation("Adding image to chat with type: {ImageType}", uploadImageType);
            var sendMessage = new ChatMessageContentItemCollection
            {
                new ImageContent { Data = imageStream.ToArray(), MimeType = uploadImageType }
            };
            // Add the image to the chat history
            history.AddUserMessage(sendMessage);
            logger.LogInformation("Image added to chat successfully.");
            return true;
        }
    }
}
