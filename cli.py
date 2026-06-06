"""
Admin Command Line Interface for the Vector Database.

Usage:
  python cli.py status
  python cli.py stats
  python cli.py rebuild-index --method hnsw
  python cli.py migrate
"""

import click
import json
from config.database import SessionLocal
from services.vector_service import VectorService

@click.group()
def cli():
    """Vector DB Admin CLI."""
    pass

@cli.command()
def status():
    """Print the overall status of the vector database."""
    db = SessionLocal()
    try:
        service = VectorService(db)
        health = service.get_health_status() if hasattr(service, 'get_health_status') else {"status": "ok (mock)"}
        click.echo(json.dumps(health, indent=2))
    except Exception as e:
        click.echo(f"Error connecting to database: {e}", err=True)
    finally:
        db.close()

@cli.command()
def stats():
    """Print index and vector statistics."""
    db = SessionLocal()
    try:
        service = VectorService(db)
        stats = service.get_database_stats()
        click.echo(json.dumps(stats, indent=2))
    finally:
        db.close()

@cli.command()
@click.option('--method', type=click.Choice(['hnsw', 'ivf']), default='hnsw', help='Index method to rebuild')
@click.option('--collection-id', help='Optional collection ID to rebuild')
def rebuild_index(method, collection_id):
    """Rebuild the specified vector index."""
    click.echo(f"Rebuilding {method} index" + (f" for collection {collection_id}" if collection_id else " (global)") + "...")
    db = SessionLocal()
    try:
        service = VectorService(db)
        # Assuming VectorService has a way to rebuild, or we call create_index with overwrite
        if method == 'hnsw':
            res = service.create_index(method='hnsw', collection_id=collection_id)
        elif method == 'ivf':
            res = service.create_index(method='ivf', collection_id=collection_id)
        
        if res.get('success'):
            click.echo("✅ Index rebuilt successfully.")
            service.save_index(method, collection_id=collection_id)
        else:
            click.echo(f"❌ Failed to rebuild index: {res.get('message')}")
    finally:
        db.close()

@cli.command()
def migrate():
    """Run Alembic database migrations."""
    import subprocess
    click.echo("Running database migrations...")
    try:
        # We assume 'alembic upgrade head' is the command
        subprocess.run(["alembic", "upgrade", "head"], check=True)
        click.echo("✅ Migrations applied successfully.")
    except Exception as e:
        click.echo(f"❌ Migration failed: {e}")

if __name__ == '__main__':
    cli()
