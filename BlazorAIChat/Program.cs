#pragma warning disable SKEXP0010, SKEXP0001, SKEXP0020, KMEXP00
using BlazorAIChat;
using BlazorAIChat.Authentication;
using BlazorAIChat.Components;
using BlazorAIChat.Models;
using BlazorAIChat.Services;
using BlazorAIChat.Utils;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using Microsoft.SemanticKernel;
using ModelContextProtocol.Client;

var builder = WebApplication.CreateBuilder(args);

// Configure logging to default to the console
builder.Logging.ClearProviders();
builder.Logging.AddConsole();

builder.Services.Configure<AppSettings>(builder.Configuration);
builder.Services.AddHttpClient("retryHttpClient").AddPolicyHandler(RetryHelper.GetRetryPolicy());
builder.Services.AddDbContext<AIChatDBContext>();
builder.Services.AddScoped<UserService>();
builder.Services.AddSingleton<ChatHistoryService>();
builder.Services.AddScoped<AIService>();

// Register SemanticKernel as a singleton with only synchronous dependencies
builder.Services.AddSingleton<Kernel>(serviceProvider =>
{
    var appSettings = serviceProvider.GetRequiredService<IOptions<AppSettings>>().Value;
    var httpClientFactory = serviceProvider.GetRequiredService<IHttpClientFactory>();
    var httpClient = httpClientFactory.CreateClient("retryHttpClient");

    var kernelBuilder = Kernel.CreateBuilder()
        .AddAzureOpenAIChatCompletion(appSettings.AzureOpenAIChatCompletion.DeploymentName, appSettings.AzureOpenAIChatCompletion.Endpoint, appSettings.AzureOpenAIChatCompletion.ApiKey, httpClient: httpClient)
        .AddAzureOpenAIEmbeddingGenerator(appSettings.AzureOpenAIEmbedding.DeploymentName, appSettings.AzureOpenAIChatCompletion.Endpoint, appSettings.AzureOpenAIChatCompletion.ApiKey, httpClient: httpClient);
    
    kernelBuilder.Services.AddLogging(services => services.AddConsole().SetMinimumLevel(LogLevel.Trace));
    return kernelBuilder.Build();
});

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents()
    .AddCircuitOptions(options => options.DetailedErrors = true);

builder.Services.AddCascadingAuthenticationState();

builder.Services.AddAuthentication(CookieAuthenticationDefaults.AuthenticationScheme)
    .AddCookie();

builder.Services.AddAuthorizationCore();

builder.Services.Configure<FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = int.MaxValue;
});

var app = builder.Build();

//setup EF database and migrate to latest version
using (var scope = app.Services.CreateScope())
{
    var services = scope.ServiceProvider;
    var context = services.GetRequiredService<AIChatDBContext>();
    context.Database.Migrate();

    // Configure MCP clients and tools asynchronously and inject into Kernel
    var kernel = services.GetRequiredService<Kernel>();
    var appSettings = services.GetRequiredService<IOptions<AppSettings>>().Value;
    var httpClientFactory = services.GetRequiredService<IHttpClientFactory>();
    var logger = services.GetRequiredService<ILogger<Program>>();
    var httpClient = httpClientFactory.CreateClient("retryHttpClient");

    await ConfigureMcpClientsAndToolsAsync(appSettings, httpClient, logger, kernel);
}

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();

app.UseStaticFiles();
app.UseAntiforgery();

app.UseAuthentication();

//Add easy auth middleware
app.UseMiddleware<EasyAuthMiddleware>();

app.UseAuthorization();

app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

//Before we start the app, ensure that the KNN folder exists on the filesystem
if (!Directory.Exists("KNN"))
{
    Directory.CreateDirectory("KNN");
}
if (!Directory.Exists("SFS"))
{
    Directory.CreateDirectory("SFS");
}

app.Run();



// This method configures the MCP clients and tools based on the provided app settings
static async Task ConfigureMcpClientsAndToolsAsync(AppSettings appSettings, HttpClient httpClient, ILogger logger, Kernel kernel)
{
    foreach (var server in appSettings.MCPServers)
    {
        try
        {
            logger.LogInformation("Configuring MCP client for server: {ServerName}", server.Name);
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
                throw new NotSupportedException($"Unsupported server type: {server.Type}");
            }

            IList<McpClientTool> tools = await mcpClient.ListToolsAsync();
            kernel.Plugins.AddFromFunctions($"{server.Name}", tools.Select(tool => tool.AsKernelFunction()));
        }
        catch (Exception ex)
        {
            logger.LogError(ex, $"Error connecting to MCP server {server.Name}: {ex.Message}");
        }
    }
}
