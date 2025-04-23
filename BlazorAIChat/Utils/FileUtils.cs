﻿using System.IO;
using System.Text;

namespace BlazorAIChat.Utils
{
    public static class FileUtils
    {
        public static string GetMimeTypeFromImage(Stream stream)
        {
            stream.Position = 0;

            var bmp = Encoding.ASCII.GetBytes("BM");     // BMP
            var gif = Encoding.ASCII.GetBytes("GIF");    // GIF
            var png = new byte[] { 137, 80, 78, 71 };    // PNG
            var tiff = new byte[] { 73, 73, 42 };         // TIFF
            var tiff2 = new byte[] { 77, 77, 42 };         // TIFF
            var jpeg = new byte[] { 255, 216, 255, 224 }; // jpeg
            var jpeg2 = new byte[] { 255, 216, 255, 225 }; // jpeg canon

            var buffer = new byte[4];
            int bytesRead = stream.Read(buffer, 0, buffer.Length);
            if (bytesRead < buffer.Length)
            {
                throw new InvalidOperationException("Failed to read the expected number of bytes from the stream.");
            }

            stream.Position = 0;

            if (bmp.SequenceEqual(buffer.Take(bmp.Length)))
                return "image/bmp";

            if (gif.SequenceEqual(buffer.Take(gif.Length)))
                return "image/gif";

            if (png.SequenceEqual(buffer.Take(png.Length)))
                return "image/png";

            if (tiff.SequenceEqual(buffer.Take(tiff.Length)))
                return "image/tiff";

            if (tiff2.SequenceEqual(buffer.Take(tiff2.Length)))
                return "image/tiff";

            if (jpeg.SequenceEqual(buffer.Take(jpeg.Length)))
                return "image/jpeg";

            if (jpeg2.SequenceEqual(buffer.Take(jpeg2.Length)))
                return "image/jpeg";

            return string.Empty;
        }

        public static string GetIconForFileType(string filename)
        {


            if (filename.EndsWith(".pdf"))
                return "/images/pdf_256x256.png";

            if (filename.EndsWith(".docx"))
                return "/images/word_256x256.png";

            if (filename.EndsWith(".xlsx"))
                return "/images/excel_256x256.png";

            if (filename.EndsWith(".pptx"))
                return "/images/powerpoint_256x256.png";

            if (filename.EndsWith("txt"))
                return "/images/txt_256x256.png";

            return string.Empty;
        }

        public static async Task<string> GetUrlContentTypeAsync(string url)
        {
            using (var httpClient = new HttpClient())
            {
                var request = new HttpRequestMessage(HttpMethod.Head, url);
                var response = await httpClient.SendAsync(request);
                if (response.IsSuccessStatusCode)
                    return response.Content.Headers.ContentType?.ToString() ?? string.Empty;
                else
                    return "404";
            }
        }

        public static async Task<MemoryStream?> GetDocStreamFromURLAsync(string url)
        {
            // Get the contents from the URL.  If it is not text or html, return a memory stream of the contents, else return null.
            using (var httpClient = new HttpClient())
            {
                var response = await httpClient.GetAsync(url);
                if (response.IsSuccessStatusCode)
                {
                    var contentType = response.Content.Headers.ContentType?.ToString() ?? string.Empty;
                    if (contentType.Contains("text") || contentType.Contains("html"))
                        return null;
                    else
                    {
                        var stream = new MemoryStream();
                        await response.Content.CopyToAsync(stream);
                        stream.Position = 0;
                        return stream;
                    }
                }
                else
                    return null;

            }
        }
    }
}
