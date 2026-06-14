using System.Collections.Generic;
using System.Text.Json;

namespace VectorDB.Client.Models
{
    public class SearchResult
    {
        public string VectorId { get; set; }
        public string Id { get; set; }
        public double Distance { get; set; }
        public Dictionary<string, JsonElement> Metadata { get; set; }
    }
}
