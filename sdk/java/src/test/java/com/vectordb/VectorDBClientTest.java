package com.vectordb;

import org.junit.Test;
import static org.junit.Assert.*;

public class VectorDBClientTest {

    @Test
    public void testClientCreation() {
        VectorDBClient client = new VectorDBClient("http://localhost:8000", "test-key");
        assertNotNull(client);
        client.close();
    }

    @Test(expected = Exception.class)
    public void testCreateVectorFailsWithBadUrl() throws Exception {
        VectorDBClient client = new VectorDBClient("http://localhost:1", "test-key");
        client.createVector(java.util.Arrays.asList(1.0, 2.0, 3.0), null);
        client.close();
    }

    @Test(expected = Exception.class)
    public void testSearchFailsWithBadUrl() throws Exception {
        VectorDBClient client = new VectorDBClient("http://localhost:1", "test-key");
        client.search(java.util.Arrays.asList(1.0, 2.0, 3.0), 5);
        client.close();
    }

    @Test(expected = Exception.class)
    public void testListCollectionsFailsWithBadUrl() throws Exception {
        VectorDBClient client = new VectorDBClient("http://localhost:1", "test-key");
        client.listCollections();
        client.close();
    }

    @Test(expected = Exception.class)
    public void testAddMemoryFailsWithBadUrl() throws Exception {
        VectorDBClient client = new VectorDBClient("http://localhost:1", "test-key");
        client.addMemory("test memory", java.util.Arrays.asList("general"));
        client.close();
    }

    @Test(expected = Exception.class)
    public void testSearchMemoriesFailsWithBadUrl() throws Exception {
        VectorDBClient client = new VectorDBClient("http://localhost:1", "test-key");
        client.searchMemories("test query");
        client.close();
    }

    @Test
    public void testCloseDoesNotThrow() {
        VectorDBClient client = new VectorDBClient("http://localhost:8000", "test-key");
        client.close();
    }
}
