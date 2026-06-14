use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub vector_id: Option<String>,
    pub id: Option<String>,
    pub distance: f64,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Collection {
    pub id: Option<String>,
    pub collection_id: Option<String>,
    pub name: String,
    pub modality: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Memory {
    pub memory_id: String,
    pub text: String,
    pub categories: Vec<String>,
}
