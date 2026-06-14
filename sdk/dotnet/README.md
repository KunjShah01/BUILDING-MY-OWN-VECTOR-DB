# VectorDB .NET Client

.NET SDK for the Vector DB API.

## Usage

```csharp
using VectorDB.Client;
using System.Collections.Generic;

var client = new VectorDBClient("http://localhost:8000", "your-api-key");

// Create a vector
var result = await client.CreateVectorAsync(new List<double> { 0.1, 0.2, 0.3 });

// Search
var results = await client.SearchAsync(new List<double> { 0.1, 0.2, 0.3 }, 5);

// List collections
var collections = await client.ListCollectionsAsync();

// Add a memory
var mem = await client.AddMemoryAsync("Remember this", new List<string> { "general" });

// Search memories
var mems = await client.SearchMemoriesAsync("find this");

client.Dispose();
```

## Build

```bash
dotnet build
dotnet test
```
