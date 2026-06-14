# VectorDB Java Client

Java SDK for the Vector DB API.

## Usage

```java
import com.vectordb.VectorDBClient;
import java.util.*;

VectorDBClient client = new VectorDBClient("http://localhost:8000", "your-api-key");

// Create a vector
String result = client.createVector(Arrays.asList(0.1, 0.2, 0.3), Map.of("key", "value"));

// Search
String results = client.search(Arrays.asList(0.1, 0.2, 0.3), 5);

// List collections
String collections = client.listCollections();

// Add a memory
String memResult = client.addMemory("Remember this", Arrays.asList("general"));

// Search memories
String memSearch = client.searchMemories("find this");

client.close();
```

## Build

```bash
mvn clean package
```
