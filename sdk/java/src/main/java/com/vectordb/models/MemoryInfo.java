package com.vectordb.models;

import java.util.List;

public class MemoryInfo {
    private String memoryId;
    private String text;
    private List<String> categories;

    public MemoryInfo() {}

    public MemoryInfo(String memoryId, String text, List<String> categories) {
        this.memoryId = memoryId;
        this.text = text;
        this.categories = categories;
    }

    public String getMemoryId() { return memoryId; }
    public void setMemoryId(String memoryId) { this.memoryId = memoryId; }

    public String getText() { return text; }
    public void setText(String text) { this.text = text; }

    public List<String> getCategories() { return categories; }
    public void setCategories(List<String> categories) { this.categories = categories; }
}
