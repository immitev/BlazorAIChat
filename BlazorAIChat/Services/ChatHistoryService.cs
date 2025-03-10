using Azure.Core;
using Azure.Identity;
using BlazorAIChat.Models;
using Microsoft.Azure.Cosmos;
using Microsoft.Azure.Cosmos.Fluent;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using Container = Microsoft.Azure.Cosmos.Container;

namespace BlazorAIChat.Services
{
    public class ChatHistoryService
    {
        private readonly Container? _chatContainer;
        private readonly AIChatDBContext _aIChatDBContext;
        private readonly bool _usesCosmosDb;

        /// <summary>
        /// Constructor for ChatHistoryService.
        /// Initializes the service with the provided settings.
        /// </summary>
        /// <param name="settings">Application settings.</param>
        public ChatHistoryService(IOptions<AppSettings> settings)
        {
            var appSettings = settings.Value;
            _aIChatDBContext = new AIChatDBContext(settings);

            var connectionString = appSettings.ConnectionStrings.CosmosDb;
            var databaseName = appSettings.CosmosDb.Database;
            var chatContainerName = appSettings.CosmosDb.Container;

            if (!string.IsNullOrEmpty(connectionString) && !string.IsNullOrEmpty(databaseName) && !string.IsNullOrEmpty(chatContainerName))
            {
                _usesCosmosDb = true;

                CosmosSerializationOptions options = new()
                {
                    PropertyNamingPolicy = CosmosPropertyNamingPolicy.CamelCase
                };

                CosmosClient client = new CosmosClientBuilder(connectionString)
                    .WithSerializerOptions(options)
                    .Build();

                Database database = client.GetDatabase(databaseName)!;
                Container chatContainer = database.CreateContainerIfNotExistsAsync(
                    id: chatContainerName,
                    partitionKeyPath: "/sessionId"
                ).GetAwaiter().GetResult();

                _chatContainer = chatContainer ?? throw new ArgumentException("Unable to connect to existing Azure Cosmos DB container or database.");
            }
        }

        /// <summary>
        /// Creates a new chat session.
        /// </summary>
        /// <param name="session">Chat session item to create.</param>
        /// <returns>Newly created chat session item.</returns>
        public async Task<Session> InsertSessionAsync(Session session)
        {
            if (_usesCosmosDb)
            {
                PartitionKey partitionKey = new(session.SessionId);
                return await _chatContainer!.CreateItemAsync(
                    item: session,
                    partitionKey: partitionKey
                );
            }
            else
            {
                _aIChatDBContext.Sessions.Add(session);
                await _aIChatDBContext.SaveChangesAsync();
                return session;
            }
        }

        /// <summary>
        /// Creates a new chat message.
        /// </summary>
        /// <param name="message">Chat message item to create.</param>
        /// <returns>Newly created chat message item.</returns>
        public async Task<Message> InsertMessageAsync(Message message)
        {
            if (_usesCosmosDb)
            {
                PartitionKey partitionKey = new(message.SessionId);
                Message newMessage = message with { TimeStamp = DateTime.UtcNow };
                return await _chatContainer!.CreateItemAsync(
                    item: newMessage,
                    partitionKey: partitionKey
                );
            }
            else
            {
                _aIChatDBContext.Messages.Add(message);
                await _aIChatDBContext.SaveChangesAsync();
                return message;
            }
        }

        /// <summary>
        /// Gets a list of all current chat sessions for a specified user.
        /// </summary>
        /// <param name="userId">User identifier to filter sessions.</param>
        /// <returns>List of distinct chat session items.</returns>
        public async Task<List<Session>> GetSessionsAsync(string userId)
        {
            if (_usesCosmosDb)
            {
                QueryDefinition query = new QueryDefinition("SELECT DISTINCT * FROM c WHERE c.type = @type and c.userId=@userId")
                    .WithParameter("@type", nameof(Session))
                    .WithParameter("@userId", userId);

                FeedIterator<Session> response = _chatContainer!.GetItemQueryIterator<Session>(query);

                List<Session> output = new();
                while (response.HasMoreResults)
                {
                    FeedResponse<Session> results = await response.ReadNextAsync();
                    output.AddRange(results);
                }
                return output;
            }
            else
            {
                return await _aIChatDBContext.Sessions.Where(s => s.UserId == userId).ToListAsync();
            }
        }

        /// <summary>
        /// Gets a list of all current chat messages for a specified session identifier.
        /// </summary>
        /// <param name="sessionId">Chat session identifier used to filter messages.</param>
        /// <returns>List of chat message items for the specified session.</returns>
        public async Task<List<Message>> GetSessionMessagesAsync(string sessionId)
        {
            if (_usesCosmosDb)
            {
                QueryDefinition query = new QueryDefinition("SELECT * FROM c WHERE c.sessionId = @sessionId AND c.type = @type")
                    .WithParameter("@sessionId", sessionId)
                    .WithParameter("@type", nameof(Message));

                FeedIterator<Message> results = _chatContainer!.GetItemQueryIterator<Message>(query);

                List<Message> output = new();
                while (results.HasMoreResults)
                {
                    FeedResponse<Message> response = await results.ReadNextAsync();
                    output.AddRange(response);
                }
                return output;
            }
            else
            {
                return await _aIChatDBContext.Messages.Where(m => m.SessionId == sessionId).ToListAsync();
            }
        }

        /// <summary>
        /// Updates an existing chat session.
        /// </summary>
        /// <param name="session">Chat session item to update.</param>
        /// <returns>Revised created chat session item.</returns>
        public async Task<Session> UpdateSessionAsync(Session session)
        {
            if (_usesCosmosDb)
            {
                PartitionKey partitionKey = new(session.SessionId);
                return await _chatContainer!.ReplaceItemAsync(
                    item: session,
                    id: session.Id,
                    partitionKey: partitionKey
                );
            }
            else
            {
                _aIChatDBContext.Sessions.Update(session);
                await _aIChatDBContext.SaveChangesAsync();
                return session;
            }
        }

        /// <summary>
        /// Returns an existing chat session.
        /// </summary>
        /// <param name="sessionId">Chat session id for the session to return.</param>
        /// <returns>Chat session item.</returns>
        public async Task<Session> GetSessionAsync(string sessionId)
        {
            if (_usesCosmosDb)
            {
                PartitionKey partitionKey = new(sessionId);
                var results = await _chatContainer!.ReadItemAsync<Session>(
                    partitionKey: partitionKey,
                    id: sessionId
                );
                return results;
            }
            else
            {
                var results = await _aIChatDBContext.Sessions.FindAsync(sessionId);
                return results ?? new Session();
            }
        }

        /// <summary>
        /// Batch create chat message and update session.
        /// </summary>
        /// <param name="messages">Chat message and session items to create or replace.</param>
        public async Task UpsertSessionBatchAsync(params dynamic[] messages)
        {
            if (_usesCosmosDb)
            {
                // Make sure items are all in the same partition
                if (messages.Select(m => m.SessionId).Distinct().Count() > 1)
                {
                    throw new ArgumentException("All items must have the same partition key.");
                }

                PartitionKey partitionKey = new(messages[0].SessionId);
                TransactionalBatch batch = _chatContainer!.CreateTransactionalBatch(partitionKey);

                foreach (var message in messages)
                {
                    batch.UpsertItem(item: message);
                }

                await batch.ExecuteAsync();
            }
            else
            {
                foreach (var message in messages)
                {
                    switch (message)
                    {
                        case Session session:
                            if (await _aIChatDBContext.Sessions.FindAsync(session.SessionId) == null)
                            {
                                _aIChatDBContext.Sessions.Add(session);
                            }
                            else
                            {
                                _aIChatDBContext.Sessions.Update(session);
                            }
                            break;
                        case Message msg:
                            if (await _aIChatDBContext.Messages.FindAsync(msg.Id) == null)
                            {
                                _aIChatDBContext.Messages.Add(msg);
                            }
                            else
                            {
                                _aIChatDBContext.Messages.Update(msg);
                            }
                            break;
                    }
                }
                await _aIChatDBContext.SaveChangesAsync();
            }
        }

        /// <summary>
        /// Batch deletes an existing chat session and all related messages.
        /// </summary>
        /// <param name="sessionId">Chat session identifier used to flag messages and sessions for deletion.</param>
        public async Task DeleteSessionAndMessagesAsync(string sessionId)
        {
            if (_usesCosmosDb)
            {
                PartitionKey partitionKey = new(sessionId);

                QueryDefinition query = new QueryDefinition("SELECT VALUE c.id FROM c WHERE c.sessionId = @sessionId")
                    .WithParameter("@sessionId", sessionId);

                FeedIterator<string> response = _chatContainer!.GetItemQueryIterator<string>(query);

                TransactionalBatch batch = _chatContainer!.CreateTransactionalBatch(partitionKey);
                while (response.HasMoreResults)
                {
                    FeedResponse<string> results = await response.ReadNextAsync();
                    foreach (var itemId in results)
                    {
                        batch.DeleteItem(id: itemId);
                    }
                }
                await batch.ExecuteAsync();
            }
            else
            {
                // Delete all messages for the session
                List<Message> messages = await _aIChatDBContext.Messages.Where(m => m.SessionId == sessionId).ToListAsync();
                if (messages.Count > 0)
                {
                    _aIChatDBContext.Messages.RemoveRange(messages);
                    await _aIChatDBContext.SaveChangesAsync();
                }

                // Delete the session
                var session = await _aIChatDBContext.Sessions.FindAsync(sessionId);
                if (session != null)
                {
                    _aIChatDBContext.Sessions.Remove(session);
                    await _aIChatDBContext.SaveChangesAsync();
                }
            }
        }
    }
}
