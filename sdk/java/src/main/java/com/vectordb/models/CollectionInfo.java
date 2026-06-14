package com.vectordb.models;

public class CollectionInfo {
    private String id;
    private String collectionId;
    private String name;
    private String modality;

    public CollectionInfo() {}

    public CollectionInfo(String id, String name, String modality) {
        this.id = id;
        this.name = name;
        this.modality = modality;
    }

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getCollectionId() { return collectionId; }
    public void setCollectionId(String collectionId) { this.collectionId = collectionId; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getModality() { return modality; }
    public void setModality(String modality) { this.modality = modality; }
}
