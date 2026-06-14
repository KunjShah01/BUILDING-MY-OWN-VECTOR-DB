package com.vectordb.models;

import java.util.Map;

public class VectorSearchResult {
    private String vectorId;
    private String id;
    private double distance;
    private Map<String, Object> metadata;

    public VectorSearchResult() {}

    public VectorSearchResult(String vectorId, double distance, Map<String, Object> metadata) {
        this.vectorId = vectorId;
        this.distance = distance;
        this.metadata = metadata;
    }

    public String getVectorId() { return vectorId; }
    public void setVectorId(String vectorId) { this.vectorId = vectorId; }

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public double getDistance() { return distance; }
    public void setDistance(double distance) { this.distance = distance; }

    public Map<String, Object> getMetadata() { return metadata; }
    public void setMetadata(Map<String, Object> metadata) { this.metadata = metadata; }
}
