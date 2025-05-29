#pragma warning disable SKEXP0010, SKEXP0001, SKEXP0020, KMEXP00
using Microsoft.SemanticKernel;
using BlazorAIChat.Models;
using ModelContextProtocol.Client;

namespace BlazorAIChat.Services
{
    public class McpPluginProvider
    {
        public List<KernelPlugin> Plugins { get; } = new();

        public async Task InitializeAsync(AppSettings appSettings, IHttpClientFactory httpClientFactory, ILogger logger)
        {
            foreach (var server in appSettings.MCPServers)
            {
                try
                {
                    IMcpClient mcpClient;
                    if (server.Type.ToLower() == "stdio")
                    {
                        mcpClient = await McpClientFactory.CreateAsync(new StdioClientTransport(new()
                        {
                            Name = server.Name,
                            Command = server.Endpoint,
                            Arguments = server.Args,
                            EnvironmentVariables = server.Env,
                        }));
                    }
                    else if (server.Type.ToLower() == "sse")
                    {
                        // Use defaultHttpClient for SSE (no retry policy)
                        var httpClient = httpClientFactory.CreateClient("defaultHttpClient");
                        mcpClient = await McpClientFactory.CreateAsync(
                            new SseClientTransport(httpClient: httpClient, transportOptions: new SseClientTransportOptions()
                            {
                                Endpoint = new Uri(server.Endpoint),
                                AdditionalHeaders = server.Headers
                            }),
                            new McpClientOptions()
                            {
                                ClientInfo = new() { Name = server.Name, Version = server.Version }
                            });
                    }
                    else
                    {
                        // If you have other types that use HttpClient, use this client
                        throw new NotSupportedException($"Unsupported server type: {server.Type}");
                    }

                    IList<McpClientTool> tools = await mcpClient.ListToolsAsync();
                    var plugin = KernelPluginFactory.CreateFromFunctions(
                        server.Name,
                        tools.Select(tool => tool.AsKernelFunction())
                    );
                    Plugins.Add(plugin);
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, $"Error connecting to MCP server {server.Name}: {ex.Message}");
                }
            }
        }
    }
}
