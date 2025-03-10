using BlazorAIChat.Models;
using Microsoft.Extensions.Options;
using Microsoft.KernelMemory.AI;
using Microsoft.KernelMemory;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.KernelMemory.AI.OpenAI;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.EntityFrameworkCore;
using BlazorAIChat.Utils;
using System.Threading.Tasks;
using UglyToad.PdfPig.Fonts.TrueType.Names;
using Microsoft.SemanticKernel.Services;
using System.Net.Http;
using System.Text;
using Microsoft.SemanticKernel.Connectors.OpenAI;

namespace BlazorAIChat.Services
{
    public class AIService
    {
#pragma warning disable SKEXP0010, SKEXP0001, SKEXP0020, KMEXP00
        private readonly AppSettings settings;
        private Kernel kernel;
        private MemoryServerless? kernelMemory = null;
        private HttpClient? httpClient;
        private ITextTokenizer? textTokenizer;
        private IChatCompletionService? chatCompletionService = null;
        public ChatHistory history { get; private set; } = new ChatHistory();
        private AIChatDBContext dbContext;
        private ChatCompletionAgent sessionSummaryAgent;
        private readonly ChatHistoryService chatHistoryService;

        public AIService(IOptions<AppSettings> appSettings, ChatHistoryService chatHistoryService,IHttpClientFactory httpClientFactory, AIChatDBContext dbContext)
        {
            this.dbContext = dbContext;
            this.chatHistoryService = chatHistoryService;

            //Get the app settings from the appsettings.json file or App Service configuration
            settings = appSettings.Value;

            //setup the HttpClient that has Retry policy
            httpClient = httpClientFactory.CreateClient("retryHttpClient");

            //Setup the Tokenizer to use
            textTokenizer = new GPT4Tokenizer();

            //Configurate Semantic Kernel
            // Create a kernel builder with Azure OpenAI chat completion. Both chat completion and embedding use the same OpenAI endpoint and key.
            var builder = Kernel.CreateBuilder()
            .AddAzureOpenAIChatCompletion(settings.AzureOpenAIChatCompletion.Model, settings.AzureOpenAIChatCompletion.Endpoint, settings.AzureOpenAIChatCompletion.ApiKey, httpClient: httpClient)
            .AddAzureOpenAITextEmbeddingGeneration(settings.AzureOpenAIEmbedding.Model, settings.AzureOpenAIChatCompletion.Endpoint, settings.AzureOpenAIChatCompletion.ApiKey, httpClient: httpClient);

            // Add enterprise components
            builder.Services.AddLogging(services => services.AddConsole().SetMinimumLevel(LogLevel.Trace));

            // Build the kernel
            kernel = builder.Build();

            //Define chat completion Agent for creating a chat session name
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

            //Set file directory for storing knowledge if PostgreSQL or Azure AI Search is not used
            var knnDirectory = "KNN";

            //Setup the memory store
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
            },
                httpClient: httpClient)
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


            //If Azure AI Search is configured, we use that for storage
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
                //Use PostgreSQL DB memory store
                //NOTE: You must have enabled pgvector extension in your PostgreSQL database for this to work.
                kernelMemoryBuilder = kernelMemoryBuilder.WithPostgresMemoryDb(new PostgresConfig()
                {
                    ConnectionString = settings.ConnectionStrings.PostgreSQL
                });
            }
            else
            {
                //Use file memory store
                kernelMemoryBuilder = kernelMemoryBuilder.WithSimpleVectorDb(new Microsoft.KernelMemory.MemoryStorage.DevTools.SimpleVectorDbConfig { StorageType = Microsoft.KernelMemory.FileSystem.DevTools.FileSystemTypes.Disk, Directory = knnDirectory });
            }

            //Configure document intelligence if configured
            if (!settings.AzureOpenAIChatCompletion.SupportsImages && settings.UsesAzureDocIntelligence)
            {
                kernelMemoryBuilder = kernelMemoryBuilder.WithAzureAIDocIntel(new AzureAIDocIntelConfig()
                {
                    Endpoint = settings.DocumentIntelligence.Endpoint,
                    APIKey = settings.DocumentIntelligence.ApiKey,
                    Auth = AzureAIDocIntelConfig.AuthTypes.APIKey
                });
            }

            //Build the memory store
            kernelMemory = kernelMemoryBuilder.Build<MemoryServerless>();

            //Get reference to the chat completion service
            chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();
        }


        private async Task ProcessURLsWithKernelMemory(List<string> urls, Session? currentSession, User currentUser)
        {
            if (kernelMemory != null && currentSession!=null)
            {
                string index = currentSession.SessionId;
                TagCollection tags = new TagCollection();
                tags.Add("user", currentUser.Id);
                foreach (var url in urls)
                {
                    //check if the url is already in the session
                    var doc = dbContext.SessionDocuments.FirstOrDefault(d => d.FileNameOrUrl == url && d.SessionId == index);
                    if (doc == null)
                    {
                        var docId = Guid.NewGuid().ToString();
                        await kernelMemory.ImportWebPageAsync(url, docId, tags, index);
                        while (!await kernelMemory.IsDocumentReadyAsync(docId, index))
                        {
                            Thread.Sleep(500);
                        }
                        dbContext.SessionDocuments.Add(new SessionDocument() { DocId = docId, FileNameOrUrl = url, SessionId = index });
                        dbContext.SaveChanges();
                    }
                }
            }
        }

        public async Task<IAsyncEnumerable<StreamingChatMessageContent>> GetChatResponseAsync(string prompt, Message message,Session currentSession,User currentUser)
        {
            ArgumentNullException.ThrowIfNull(textTokenizer);
            ArgumentNullException.ThrowIfNull(chatCompletionService);

            await DoRAG(prompt, message, currentSession, currentUser);
            
            //Clean up history and remove old non-system messages in order to stay below our MaxInputTokens limit.
            //This is not the most efficient way to do this, but it is simple and works for this demo.
            history = AIUtils.CleanUpHistory(history, textTokenizer, settings.AzureOpenAIChatCompletion.MaxInputTokens);
            
            return chatCompletionService.GetStreamingChatMessageContentsAsync(history);
        }

        private async Task DoRAG(string prompt, Message message, Session currentSession, User currentUser)
        {
           

            //Pull out any URLs from the message
            var urls = StringUtils.GetURLsFromString(prompt);

            //If we have URLs, we need to add them to kernel memory
            SearchResult? searchData = null;
            if (urls.Count > 0)
            {
                await ProcessURLsWithKernelMemory(urls,currentSession,currentUser);

                //Remove urls from messageToProcess string.
                string messageToProcessNoURLS = await GenerateNewPromptForMessagesWithUrl(prompt);
              
                searchData = await kernelMemory!.SearchAsync(messageToProcessNoURLS, currentSession.Id);
            }
            else
            {
                searchData = await kernelMemory!.SearchAsync(prompt, currentSession.Id);
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
                            history.AddUserMessage(p.Text);
                        }

                        //Add the source to the message
                        if (result.SourceName.ToLower() != "content.url")
                            message.Citations.Add($"{result.SourceName} ({(result.Partitions.Max(x => x.Relevance) * 100).ToString("F2")}%)");
                        else
                            message.Citations.Add($"<a href='{result.SourceUrl}' target='_blank'>{result.SourceUrl}</a> ({(result.Partitions.Max(x => x.Relevance) * 100).ToString("F2")}%)");

                    }
                    history.AddUserMessage("----------------------------------");
                }
            }

            //Add the user message to the chat history
            history.AddUserMessage(prompt);
        }

        private async Task<string> GenerateNewPromptForMessagesWithUrl(string theMessage)
        {
            ArgumentNullException.ThrowIfNull(chatCompletionService);

            string completionText = string.Empty;
            var skChatHistory = new ChatHistory();
            skChatHistory.AddSystemMessage("Rewrite the user prompt to remove all URLs but still make the question or request understandable.");
            skChatHistory.AddUserMessage(theMessage);
            PromptExecutionSettings settings = new()
            {
                ExtensionData = new Dictionary<string, object>()
                {
                    { "Temperature", 0.3 }
                }
            };

            var result = await chatCompletionService.GetChatMessageContentAsync(skChatHistory, settings);
            completionText = result.Items[0].ToString()!;
            return completionText;
        }

        public async Task<string> SummarizeChatSessionNameAsync(string? sessionId)
        {
            ArgumentNullException.ThrowIfNull(sessionId);
            ArgumentNullException.ThrowIfNull(chatCompletionService);

            //Get the messages for the session
            List<Message> messages = await chatHistoryService.GetSessionMessagesAsync(sessionId);

            //Create a conversation string from the messages
            string conversationText = string.Join(" ", messages.Select(m => m.Prompt + " " + m.Completion));

            //Use sessionSummaryAgent to summarize the conversation into a session title
            ChatHistory sessionSummary = [new ChatMessageContent(AuthorRole.User, conversationText)];
            StringBuilder output = new();
            await foreach (ChatMessageContent response in sessionSummaryAgent.InvokeAsync(sessionSummary))
            {
                output.Append(response.ToString());
            }
            string completionText = output.ToString();

            Session session = await chatHistoryService.GetSessionAsync(sessionId);
            session.Name = completionText;
            await chatHistoryService.UpdateSessionAsync(session);
            return completionText;
        }

        //Processes the uploaded file with Kernel Memory and stores the embeddings in the configured storage system.
        public async Task<bool> ProcessDocsWithKernelMemory(MemoryStream memoryStream, string filename, Session currentSession, User currentUser)
        {
            ArgumentNullException.ThrowIfNull(kernelMemory);

            //Let's see if the document already is in the session, If so give a notification and then return.
            var doc = dbContext.SessionDocuments.FirstOrDefault(d => d.FileNameOrUrl == filename && d.SessionId == currentSession.SessionId);
            if (doc != null)
            {
                //ShowAlert("The document you uploaded is already in the chat session.", AlertTypeEnum.warning);
                return false;
            }

            //Prep variables for processing
            var docId = Guid.NewGuid().ToString();
            string index = currentSession.SessionId;
            TagCollection tags = new TagCollection();
            tags.Add("user", currentUser.Id);
            memoryStream.Position = 0;

            await kernelMemory.ImportDocumentAsync(memoryStream, filename, docId, tags, index);

            //Add record to database about document
            dbContext.SessionDocuments.Add(new SessionDocument() { DocId = docId, FileNameOrUrl = filename, SessionId = index });
            dbContext.SaveChanges();

            while (!await kernelMemory.IsDocumentReadyAsync(docId, index))
            {
                Thread.Sleep(1000);
            }

            return true;
    
        }

        // Deletes the uploaded documents from the memory store
        public async Task<bool> DeleteUploadedDocs(string sessionIdToDelete)
        {
            if (!string.IsNullOrEmpty(sessionIdToDelete))
            {

                if (kernelMemory != null)
                    await kernelMemory.DeleteIndexAsync(sessionIdToDelete);

                //Delete the documents from the database
                var docs = dbContext.SessionDocuments.Where(d => d.SessionId == sessionIdToDelete);
                dbContext.SessionDocuments.RemoveRange(docs);
                dbContext.SaveChanges();

                return true;
            }
            return false;
        }

        public bool AddImageToChat(MemoryStream imageStream, string uploadImageType)
        {
            //Add the image to the chat history so the AI can process it
            var sendMessage = new ChatMessageContentItemCollection
            {
                new ImageContent(){Data=imageStream.ToArray(), MimeType = uploadImageType }
            };
            history.AddUserMessage(sendMessage);
            return true;
        }

    }
}