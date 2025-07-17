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
            if (appSettings.Mcp?.Servers == null)
                return;

            foreach (var serverEntry in appSettings.Mcp.Servers)
            {
                var serverName = serverEntry.Key;
                var server = serverEntry.Value;
                try
                {
                    IMcpClient mcpClient;
                    if (server.Type.ToLower() == "stdio" || string.IsNullOrEmpty(server.Type))
                    {
                        mcpClient = await McpClientFactory.CreateAsync(new StdioClientTransport(new()
                        {
                            Name = serverName,
                            Command = server.Command ?? string.Empty,
                            Arguments = server.Args ?? new List<string>(),
                            EnvironmentVariables = server.Env ?? new Dictionary<string, string>()
                        }));
                    }
                    else if (server.Type.ToLower() == "sse")
                    {
                        var httpClient = httpClientFactory.CreateClient("defaultHttpClient");
                        mcpClient = await McpClientFactory.CreateAsync(
                            new SseClientTransport(httpClient: httpClient, transportOptions: new SseClientTransportOptions()
                            {
                                Endpoint = new Uri(server.Url ?? string.Empty),
                                AdditionalHeaders = server.Headers ?? new Dictionary<string, string>()
                            }),
                            new McpClientOptions()
                            {
                                ClientInfo = new() { Name = serverName, Version = "1.0.0.0" }
                            });
                    }
                    else
                    {
                        throw new NotSupportedException($"Unsupported server type: {server.Type}");
                    }

                    IList<McpClientTool> tools = await mcpClient.ListToolsAsync();
                    var plugin = KernelPluginFactory.CreateFromFunctions(
                        serverName,
                        tools.Select(tool => tool.AsKernelFunction())
                    );
                    Plugins.Add(plugin);
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, $"Error connecting to MCP server {serverName}: {ex.Message}");
                }
            }
        }
    }
}
