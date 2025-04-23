namespace BlazorAIChat.Services
{
    using System.Security.Claims;
    using BlazorAIChat.Models;
    using BlazorAIChat.Utils;
    using Microsoft.AspNetCore.Components.Authorization;
    using Microsoft.Extensions.Logging;

    public class UserService
    {
        private readonly AuthenticationStateProvider _authenticationStateProvider;
        private readonly ILogger<UserService> _logger;
        private ClaimsPrincipal? userPrincipal;

        public UserService(AuthenticationStateProvider authenticationStateProvider, ILogger<UserService> logger)
        {
            _authenticationStateProvider = authenticationStateProvider;
            _logger = logger;
            _logger.LogInformation("UserService initialized");
        }

        public async Task<User> GetCurrentUserAsync()
        {
            _logger.LogInformation("Getting current user");

            try
            {
                var authState = await _authenticationStateProvider.GetAuthenticationStateAsync();
                userPrincipal = authState.User;

                if (userPrincipal.Identity?.IsAuthenticated == true)
                {
                    var user = UserUtils.ConvertPrincipalToUser(userPrincipal);
                    _logger.LogInformation("Authenticated user retrieved: {UserId}", user.Id);
                    return user;
                }
                else
                {
                    _logger.LogWarning("User is not authenticated, returning guest user");
                    return new User
                    {
                        Id = "Guest User",
                        Name = "Guest User",
                        Role = UserRoles.Guest
                    };
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error occurred while getting current user");
                throw;
            }
        }

        public bool DoesUserNeedToRequestAccess(User user, Config config, bool requireEasyAuth)
        {
            _logger.LogInformation("Checking if user needs to request access: UserId={UserId}, RequireEasyAuth={RequireEasyAuth}", user.Id, requireEasyAuth);

            try
            {
                var result = user.Role == UserRoles.Guest && requireEasyAuth && userPrincipal?.Identity?.IsAuthenticated == true && config.RequireAccountApprovals;
                _logger.LogInformation("User needs to request access: {Result}", result);
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error occurred while checking if user needs to request access: UserId={UserId}", user.Id);
                throw;
            }
        }

        public bool IsUserAccountExpired(User user, Config config, bool requireEasyAuth)
        {
            _logger.LogInformation("Checking if user account is expired: UserId={UserId}, RequireEasyAuth={RequireEasyAuth}", user.Id, requireEasyAuth);

            try
            {
                if (!requireEasyAuth)
                {
                    _logger.LogInformation("EasyAuth is not required, account is not expired");
                    return false;
                }

                if (user.Role == UserRoles.Admin)
                {
                    _logger.LogInformation("User is an admin, account is not expired");
                    return false;
                }

                if (config.ExpirationDays == 0)
                {
                    _logger.LogInformation("Expiration days is set to 0, account is not expired");
                    return false;
                }

                if (!config.RequireAccountApprovals)
                {
                    _logger.LogInformation("Account approvals are not required, account is not expired");
                    return false;
                }

                var isExpired = user.DateApproved is not null && user.DateApproved.Value.AddDays(config.ExpirationDays) <= DateTime.Now;
                _logger.LogInformation("Account expiration status: {IsExpired}", isExpired);
                return isExpired;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error occurred while checking if user account is expired: UserId={UserId}", user.Id);
                throw;
            }
        }
    }
}
