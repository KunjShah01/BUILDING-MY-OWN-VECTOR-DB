use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::{json, Value};

use crate::models::*;

pub struct VectorDBClient {
    client: Client,
    base_url: String,
}

impl VectorDBClient {
    pub fn new(base_url: &str, api_key: Option<&str>) -> Self {
        let mut headers = HeaderMap::new();
        if let Some(key) = api_key {
            let val = HeaderValue::from_str(&format!("Bearer {}", key)).unwrap();
            headers.insert("Authorization", val);
        }
        VectorDBClient {
            client: Client::builder().default_headers(headers).build().unwrap(),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    pub fn create_vector(&self, vector: Vec<f64>, metadata: Option<Value>) -> Result<Value, String> {
        let mut body = json!({"vector": vector});
        if let Some(m) = metadata {
            body["metadata"] = m;
        }
        self.client
            .post(format!("{}/vectors", self.base_url))
            .json(&body)
            .send()
            .map_err(|e| e.to_string())?
            .json::<Value>()
            .map_err(|e| e.to_string())
    }

    pub fn search(&self, query_vector: Vec<f64>, k: u32) -> Result<Vec<SearchResult>, String> {
        let body = json!({"vector": query_vector, "k": k});
        let colls: Value = self
            .client
            .get(format!("{}/collections", self.base_url))
            .send()
            .map_err(|e| e.to_string())?
            .json::<Value>()
            .map_err(|e| e.to_string())?;
        let coll_id = colls["collections"][0]["collection_id"]
            .as_str()
            .or_else(|| colls["collections"][0]["id"].as_str())
            .unwrap_or("default");
        self.client
            .post(format!("{}/collections/{}/search", self.base_url, coll_id))
            .json(&body)
            .send()
            .map_err(|e| e.to_string())?
            .json::<Vec<SearchResult>>()
            .map_err(|e| e.to_string())
    }

    pub fn list_collections(&self) -> Result<Value, String> {
        self.client
            .get(format!("{}/collections", self.base_url))
            .send()
            .map_err(|e| e.to_string())?
            .json::<Value>()
            .map_err(|e| e.to_string())
    }

    pub fn add_memory(&self, text: &str, categories: Vec<&str>) -> Result<Value, String> {
        let body = json!({"text": text, "categories": categories});
        self.client
            .post(format!("{}/memories", self.base_url))
            .json(&body)
            .send()
            .map_err(|e| e.to_string())?
            .json::<Value>()
            .map_err(|e| e.to_string())
    }

    pub fn search_memories(&self, query: &str) -> Result<Value, String> {
        let body = json!({"query": query});
        self.client
            .post(format!("{}/memories/search", self.base_url))
            .json(&body)
            .send()
            .map_err(|e| e.to_string())?
            .json::<Value>()
            .map_err(|e| e.to_string())
    }
}
