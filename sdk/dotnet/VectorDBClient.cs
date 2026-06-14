using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace VectorDB.Client
{
    public class VectorDBClient : IDisposable
    {
        private readonly HttpClient _http;
        private readonly string _baseUrl;

        public VectorDBClient(string baseUrl = "http://localhost:8000", string apiKey = null)
        {
            _baseUrl = baseUrl.TrimEnd('/');
            _http = new HttpClient();
            if (!string.IsNullOrEmpty(apiKey))
                _http.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
        }

        public async Task<JsonElement> CreateVectorAsync(List<double> vector, Dictionary<string, object> metadata = null)
        {
            var body = new Dictionary<string, object> { ["vector"] = vector };
            if (metadata != null)
                body["metadata"] = metadata;
            var json = JsonSerializer.Serialize(body);
            var res = await _http.PostAsync($"{_baseUrl}/vectors",
                new StringContent(json, Encoding.UTF8, "application/json"));
            res.EnsureSuccessStatusCode();
            return JsonSerializer.Deserialize<JsonElement>(await res.Content.ReadAsStringAsync());
        }

        public async Task<JsonElement> SearchAsync(List<double> queryVector, int k = 5)
        {
            var body = JsonSerializer.Serialize(new { vector = queryVector, k });
            var colls = JsonSerializer.Deserialize<JsonElement>(
                await _http.GetStringAsync($"{_baseUrl}/collections"));
            var collId = "default";
            if (colls.TryGetProperty("collections", out var arr) && arr.GetArrayLength() > 0)
            {
                collId = arr[0].TryGetProperty("collection_id", out var id)
                    ? id.GetString()
                    : arr[0].GetProperty("id").GetString();
            }
            var res = await _http.PostAsync($"{_baseUrl}/collections/{collId}/search",
                new StringContent(body, Encoding.UTF8, "application/json"));
            res.EnsureSuccessStatusCode();
            return JsonSerializer.Deserialize<JsonElement>(await res.Content.ReadAsStringAsync());
        }

        public async Task<JsonElement> ListCollectionsAsync()
        {
            var json = await _http.GetStringAsync($"{_baseUrl}/collections");
            return JsonSerializer.Deserialize<JsonElement>(json);
        }

        public async Task<JsonElement> AddMemoryAsync(string text, List<string> categories = null)
        {
            var body = JsonSerializer.Serialize(new
            {
                text,
                categories = categories ?? new List<string>()
            });
            var res = await _http.PostAsync($"{_baseUrl}/memories",
                new StringContent(body, Encoding.UTF8, "application/json"));
            res.EnsureSuccessStatusCode();
            return JsonSerializer.Deserialize<JsonElement>(await res.Content.ReadAsStringAsync());
        }

        public async Task<JsonElement> SearchMemoriesAsync(string query)
        {
            var body = JsonSerializer.Serialize(new { query });
            var res = await _http.PostAsync($"{_baseUrl}/memories/search",
                new StringContent(body, Encoding.UTF8, "application/json"));
            res.EnsureSuccessStatusCode();
            return JsonSerializer.Deserialize<JsonElement>(await res.Content.ReadAsStringAsync());
        }

        public void Dispose() => _http?.Dispose();
    }
}
