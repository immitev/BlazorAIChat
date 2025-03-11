#nullable enable

using BlazorAIChat.Models;
using BlazorAIChat.Utils;
using Microsoft.Extensions.Options;
using Microsoft.KernelMemory;
using Microsoft.KernelMemory.AI;
using Microsoft.KernelMemory.AI.OpenAI;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using System.Text;

namespace BlazorAIChat.Services
{
    /// <summary>
    /// The AIService class provides various AI-related functionalities such as chat completion, document processing, and session summarization.
    /// </summary>
    public class AIService
    {
#pragma warning disable SKEXP0010, SKEXP0001, SKEXP0020, KMEXP00
        private readonly AppSettings settings;
        private readonly Kernel kernel;
        private readonly MemoryServerless? kernelMemory;
        private readonly HttpClient? httpClient;
        private readonly ITextTokenizer? textTokenizer;
        private readonly IChatCompletionService? chatCompletionService;
        public ChatHistory history { get; private set; } = new ChatHistory();
        private readonly AIChatDBContext dbContext;
        private readonly ChatCompletionAgent sessionSummaryAgent;
        private readonly ChatCompletionAgent promptRewriteAgent;
        private readonly ChatHistoryService chatHistoryService;

        /// <summary>
        /// Initializes a new instance of the <see cref="AIService"/> class.
        /// </summary>
        /// <param name="appSettings">The application settings.</param>
        /// <param name="chatHistoryService">The chat history service.</param>
        /// <param name="httpClientFactory">The HTTP client factory.</param>
        /// <param name="dbContext">The database context.</param>
        public AIService(IOptions<AppSettings> appSettings, ChatHistoryService chatHistoryService, IHttpClientFactory httpClientFactory, AIChatDBContext dbContext)
        {
            this.dbContext = dbContext;
            this.chatHistoryService = chatHistoryService;

            settings = appSettings.Value;
            httpClient = httpClientFactory.CreateClient("retryHttpClient");
            textTokenizer = new GPT4Tokenizer();

            // Create a Kernel builder and add Azure OpenAI services for chat completion and text embedding generation
            var builder = Kernel.CreateBuilder()
                .AddAzureOpenAIChatCompletion(settings.AzureOpenAIChatCompletion.Model, settings.AzureOpenAIChatCompletion.Endpoint, settings.AzureOpenAIChatCompletion.ApiKey, httpClient: httpClient)
                .AddAzureOpenAITextEmbeddingGeneration(settings.AzureOpenAIEmbedding.Model, settings.AzureOpenAIChatCompletion.Endpoint, settings.AzureOpenAIChatCompletion.ApiKey, httpClient: httpClient);

            // Add logging services
            builder.Services.AddLogging(services => services.AddConsole().SetMinimumLevel(LogLevel.Trace));
            kernel = builder.Build();

            // Initialize the session summary agent with specific instructions
            sessionSummaryAgent = new()
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

            promptRewriteAgent = new()
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

            var knnDirectory = "KNN";

            // Build kernel memory with various configurations based on settings
            var kernelMemoryBuilder = new KernelMemoryBuilder()
                .WithAzureOpenAITextEmbeddingGeneration(new AzureOpenAIConfig
                {
                    APIType = AzureOpenAIConfig.APITypes.EmbeddingGeneration,
                    Endpoint = settings.AzureOpenAIChatCompletion.Endpoint,
                    Deployment = settings.AzureOpenAIEmbedding.Model,
                    Auth = AzureOpenAIConfig.AuthTypes.APIKey,
                    APIKey = settings.AzureOpenAIChatCompletion.ApiKey,
                    MaxTokenTotal = settings.AzureOpenAIEmbedding.MaxInputTokens,
                    MaxRetries = 3
                }, httpClient: httpClient)
                .WithAzureOpenAITextGeneration(new AzureOpenAIConfig
                {
                    APIType = AzureOpenAIConfig.APITypes.ChatCompletion,
                    Endpoint = settings.AzureOpenAIChatCompletion.Endpoint,
                    Deployment = settings.AzureOpenAIChatCompletion.Model,
                    Auth = AzureOpenAIConfig.AuthTypes.APIKey,
                    APIKey = settings.AzureOpenAIChatCompletion.ApiKey,
                    MaxTokenTotal = settings.AzureOpenAIChatCompletion.MaxInputTokens,
                    MaxRetries = 3
                }, httpClient: httpClient, textTokenizer: textTokenizer);

            // Configure kernel memory based on specific settings
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
                kernelMemoryBuilder = kernelMemoryBuilder.WithSimpleVectorDb(new Microsoft.KernelMemory.MemoryStorage.DevTools.SimpleVectorDbConfig { StorageType = Microsoft.KernelMemory.FileSystem.DevTools.FileSystemTypes.Disk, Directory = knnDirectory });
            }

            // Add Azure Document Intelligence if supported
            if (!settings.AzureOpenAIChatCompletion.SupportsImages && settings.UsesAzureDocIntelligence)
            {
                kernelMemoryBuilder = kernelMemoryBuilder.WithAzureAIDocIntel(new AzureAIDocIntelConfig()
                {
                    Endpoint = settings.DocumentIntelligence.Endpoint,
                    APIKey = settings.DocumentIntelligence.ApiKey,
                    Auth = AzureAIDocIntelConfig.AuthTypes.APIKey
                });
            }

            kernelMemory = kernelMemoryBuilder.Build<MemoryServerless>();
            chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();
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
        public async Task<IAsyncEnumerable<StreamingChatMessageContent>> GetChatResponseAsync(string prompt, Message message, Session currentSession, User currentUser)
        {
            ArgumentNullException.ThrowIfNull(textTokenizer);
            ArgumentNullException.ThrowIfNull(chatCompletionService);

            // Perform Retrieval-Augmented Generation (RAG) on the prompt
            await DoRAG(prompt, message, currentSession, currentUser).ConfigureAwait(false);

            // Clean up the chat history to fit within the token limit
            history = AIUtils.CleanUpHistory(history, textTokenizer, settings.AzureOpenAIChatCompletion.MaxInputTokens);

            // Get the chat response as a stream of messages
            return chatCompletionService.GetStreamingChatMessageContentsAsync(history);
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
            var urls = StringUtils.GetURLsFromString(prompt);
            SearchResult? searchData = null;
            if (urls.Count > 0)
            {
                // Process URLs with kernel memory
                await ProcessURLsWithKernelMemory(urls, currentSession, currentUser).ConfigureAwait(false);
                // Generate a new prompt without URLs
                string messageToProcessNoURLS = await GenerateNewPromptForMessagesWithUrl(prompt).ConfigureAwait(false);
                // Search the kernel memory with the new prompt
                searchData = await kernelMemory!.SearchAsync(messageToProcessNoURLS, currentSession.Id).ConfigureAwait(false);
            }
            else
            {
                // Search the kernel memory with the original prompt
                searchData = await kernelMemory!.SearchAsync(prompt, currentSession.Id).ConfigureAwait(false);
            }

            if (searchData != null && !searchData.NoResult)
            {
                if (searchData.Results.Count > 0)
                {
                    history.AddUserMessage("No matter what the question or request, base the response only on the information below.");
                    foreach (var result in searchData.Results)
                    {
                        foreach (var p in result.Partitions)
                        {
                            // Add the search result text to the chat history
                            history.AddUserMessage(p.Text);
                        }

                        // Add citations to the message
                        if (result.SourceName.ToLower() != "content.url")
                            message.Citations.Add($"{result.SourceName} ({(result.Partitions.Max(x => x.Relevance) * 100).ToString("F2")}%)");
                        else
                            message.Citations.Add($"<a href='{result.SourceUrl}' target='_blank'>{result.SourceUrl}</a> ({(result.Partitions.Max(x => x.Relevance) * 100).ToString("F2")}%)");
                    }
                    history.AddUserMessage("----------------------------------");
                }
            }

            // Add the original prompt to the chat history
            history.AddUserMessage(prompt);
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

            // Get the updated prompt from the prompt rewrite agent
            await foreach (var response in promptRewriteAgent.InvokeAsync(sessionSummary).ConfigureAwait(false))
            {
                output.Append(response.ToString());
            }
            var completionText = output.ToString();

            return completionText;
        }

        /// <summary>
        /// Summarizes the chat session name asynchronously.
        /// </summary>
        /// <param name="sessionId">The session identifier.</param>
        /// <returns>The summarized session name.</returns>
        public async Task<string> SummarizeChatSessionNameAsync(string? sessionId)
        {
            ArgumentNullException.ThrowIfNull(sessionId);
            ArgumentNullException.ThrowIfNull(chatCompletionService);

            // Get all messages from the session
            var messages = await chatHistoryService.GetSessionMessagesAsync(sessionId).ConfigureAwait(false);
            var conversationText = string.Join(" ", messages.Select(m => m.Prompt + " " + m.Completion));

            // Create a chat history with the entire conversation
            ChatHistory sessionSummary = new() { new ChatMessageContent(AuthorRole.User, conversationText) };
            StringBuilder output = new();
            // Get the summary from the session summary agent
            await foreach (var response in sessionSummaryAgent.InvokeAsync(sessionSummary).ConfigureAwait(false))
            {
                output.Append(response.ToString());
            }
            var completionText = output.ToString();

            // Update the session name with the summary
            var session = await chatHistoryService.GetSessionAsync(sessionId).ConfigureAwait(false);
            session.Name = completionText;
            await chatHistoryService.UpdateSessionAsync(session).ConfigureAwait(false);
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

            return true;
        }

        /// <summary>
        /// Deletes uploaded documents for a given session.
        /// </summary>
        /// <param name="sessionIdToDelete">The session identifier to delete documents for.</param>
        /// <returns>A boolean indicating whether the documents were deleted successfully.</returns>
        public async Task<bool> DeleteUploadedDocs(string? sessionIdToDelete)
        {
            if (!string.IsNullOrEmpty(sessionIdToDelete))
            {
                if (kernelMemory != null)
                    await kernelMemory.DeleteIndexAsync(sessionIdToDelete).ConfigureAwait(false);

                // Remove documents from the database
                var docs = dbContext.SessionDocuments.Where(d => d.SessionId == sessionIdToDelete);
                dbContext.SessionDocuments.RemoveRange(docs);
                await dbContext.SaveChangesAsync().ConfigureAwait(false);

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
            var sendMessage = new ChatMessageContentItemCollection
            {
                new ImageContent { Data = imageStream.ToArray(), MimeType = uploadImageType }
            };
            // Add the image to the chat history
            history.AddUserMessage(sendMessage);
            return true;
        }
    }
}
