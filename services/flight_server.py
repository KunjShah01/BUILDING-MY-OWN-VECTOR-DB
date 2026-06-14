"""Apache Arrow Flight server for zero-copy vector transfer."""
import logging
import os
import tempfile
from typing import Optional

import pyarrow as pa
import pyarrow.flight as flight

logger = logging.getLogger(__name__)


class VectorFlightServer(flight.FlightServerBase):
    def __init__(self, location: str = "grpc://0.0.0.0:50051", db_session_factory=None):
        super().__init__(location)
        self._location = location
        self._db_session_factory = db_session_factory
        self._tables: dict = {}
        logger.info("Arrow Flight server listening on %s", location)

    def do_get(self, context, ticket):
        """Export vectors as Arrow record batches."""
        table = self._tables.get(ticket.ticket.decode())
        if table is None:
            return flight.GeneratorStream(pa.schema([]), iter([]))
        return flight.RecordBatchStream(table)

    def do_put(self, context, descriptor, reader, writer):
        """Import vectors from Arrow record batches."""
        table = reader.read_all()
        name = descriptor.path[0].decode() if descriptor.path else f"table_{len(self._tables)}"
        self._tables[name] = table
        logger.info("Imported %d rows into table '%s'", table.num_rows, name)
        writer.write(pa.array([table.num_rows]))

    def list_tables(self) -> list:
        return list(self._tables.keys())

    def get_table(self, name: str) -> Optional[pa.Table]:
        return self._tables.get(name)


def start_flight_server(db_session_factory=None, host: str = "0.0.0.0", port: int = 50051):
    location = f"grpc://{host}:{port}"
    server = VectorFlightServer(location, db_session_factory)
    return server
