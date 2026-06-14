# VectorDB Rust Client

Rust SDK for the Vector DB API.

## Usage

```rust
use vectordb_client::client::VectorDBClient;
use serde_json::json;

let client = VectorDBClient::new("http://localhost:8000", Some("your-api-key"));

// Create a vector
let result = client.create_vector(vec![0.1, 0.2, 0.3], Some(json!({"key": "value"})));
println!("{:?}", result);

// Search
let results = client.search(vec![0.1, 0.2, 0.3], 5);
println!("{:?}", results);

// List collections
let collections = client.list_collections();
println!("{:?}", collections);

// Add a memory
let mem = client.add_memory("Remember this", vec!["general"]);
println!("{:?}", mem);

// Search memories
let mems = client.search_memories("find this");
println!("{:?}", mems);
```

## Build

```bash
cargo build
cargo test
```
