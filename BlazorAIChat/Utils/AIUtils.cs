using Microsoft.KernelMemory.AI;
using Microsoft.SemanticKernel.ChatCompletion;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace BlazorAIChat.Utils
{
    public static class AIUtils
    {
#pragma warning disable KMEXP00

        public static async Task<ChatHistory> CleanUpHistoryAsync(
            ChatHistory history,
            IChatCompletionService chatCompletionService,
            int targetMessageCount,
            int? thresholdCount,
            CancellationToken cancellationToken = default)
        {
            var reducer = new ChatHistorySummarizationReducer(
                chatCompletionService,
                targetMessageCount,
                thresholdCount);

            var reduced = await reducer.ReduceAsync(history, cancellationToken).ConfigureAwait(false);

            //if reduced is null, we just pass the current chat history back, otherwise we send back the new reduced history
            if (reduced is null)
                return history;
            else
                return new ChatHistory(reduced);

        }
    }
}