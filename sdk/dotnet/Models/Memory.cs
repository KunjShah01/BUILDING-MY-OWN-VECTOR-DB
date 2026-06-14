using System.Collections.Generic;

namespace VectorDB.Client.Models
{
    public class Memory
    {
        public string MemoryId { get; set; }
        public string Text { get; set; }
        public List<string> Categories { get; set; }
    }
}
