using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace BlazorAIChat.Migrations
{
    /// <inheritdoc />
    public partial class AddedCompetionTimestamp : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<DateTime>(
                name: "CompletionTimeStamp",
                table: "Messages",
                type: "TEXT",
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "CompletionTimeStamp",
                table: "Messages");
        }
    }
}
