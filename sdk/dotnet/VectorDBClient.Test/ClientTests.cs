using Xunit;
using VectorDB.Client;
using System;

namespace VectorDB.Client.Test
{
    public class ClientTests
    {
        [Fact]
        public void CanCreateClient()
        {
            var client = new VectorDBClient("http://localhost:8000", "test-key");
            Assert.NotNull(client);
            client.Dispose();
        }

        [Fact]
        public void DefaultConstructorWorks()
        {
            var client = new VectorDBClient();
            Assert.NotNull(client);
            client.Dispose();
        }

        [Fact]
        public void DisposeDoesNotThrow()
        {
            var client = new VectorDBClient();
            var ex = Record.Exception(() => client.Dispose());
            Assert.Null(ex);
        }
    }
}
