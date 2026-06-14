package com.vectordb;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.vectordb.models.CollectionInfo;
import com.vectordb.models.MemoryInfo;
import com.vectordb.models.VectorSearchResult;
import okhttp3.*;

import java.io.IOException;
import java.lang.reflect.Type;
import java.util.*;
import java.util.concurrent.TimeUnit;

public class VectorDBClient implements AutoCloseable {
    private final OkHttpClient client;
    private final String baseUrl;
    private final Gson gson;

    public VectorDBClient(String baseUrl, String apiKey) {
        this.baseUrl = baseUrl.replaceAll("/+$", "");
        this.gson = new Gson();

        OkHttpClient.Builder builder = new OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(30, TimeUnit.SECONDS);

        if (apiKey != null && !apiKey.isEmpty()) {
            builder.addInterceptor(chain -> {
                Request request = chain.request().newBuilder()
                        .addHeader("Authorization", "Bearer " + apiKey)
                        .build();
                return chain.proceed(request);
            });
        }

        this.client = builder.build();
    }

    private String doPost(String path, Object body) throws IOException {
        String json = gson.toJson(body);
        Request request = new Request.Builder()
                .url(baseUrl + path)
                .post(RequestBody.create(json, MediaType.get("application/json")))
                .build();
        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("Unexpected code " + response.code() + ": " + response.body().string());
            }
            return response.body().string();
        }
    }

    private String doGet(String path) throws IOException {
        Request request = new Request.Builder()
                .url(baseUrl + path)
                .get()
                .build();
        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("Unexpected code " + response.code() + ": " + response.body().string());
            }
            return response.body().string();
        }
    }

    public String createVector(List<Double> vector, Map<String, Object> metadata) throws IOException {
        Map<String, Object> body = new HashMap<>();
        body.put("vector", vector);
        if (metadata != null) {
            body.put("metadata", metadata);
        }
        return doPost("/vectors", body);
    }

    public String search(List<Double> queryVector, int k) throws IOException {
        String collsJson = doGet("/collections");
        Type mapType = new TypeToken<Map<String, Object>>(){}.getType();
        Map<String, Object> colls = gson.fromJson(collsJson, mapType);

        String collId = "default";
        if (colls.containsKey("collections")) {
            List<Map<String, Object>> list = (List<Map<String, Object>>) colls.get("collections");
            if (list != null && !list.isEmpty()) {
                Map<String, Object> first = list.get(0);
                collId = first.containsKey("collection_id")
                        ? (String) first.get("collection_id")
                        : (String) first.get("id");
            }
        }

        Map<String, Object> body = new HashMap<>();
        body.put("vector", queryVector);
        body.put("k", k);
        return doPost("/collections/" + collId + "/search", body);
    }

    public String listCollections() throws IOException {
        return doGet("/collections");
    }

    public String addMemory(String text, List<String> categories) throws IOException {
        Map<String, Object> body = new HashMap<>();
        body.put("text", text);
        body.put("categories", categories != null ? categories : new ArrayList<>());
        return doPost("/memories", body);
    }

    public String searchMemories(String query) throws IOException {
        Map<String, Object> body = new HashMap<>();
        body.put("query", query);
        return doPost("/memories/search", body);
    }

    @Override
    public void close() {
        client.dispatcher().executorService().shutdown();
        client.connectionPool().evictAll();
    }
}
