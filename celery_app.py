"""
Celery configuration for background tasks
Handles index building, bulk operations, and scheduled maintenance

Graceful shutdown: On SIGTERM/SIGINT, the worker will:
  1. Stop accepting new tasks
  2. Wait up to worker_shutdown_timeout for in-flight tasks to complete
  3. Revoke any remaining tasks back to the broker
  4. Clean up connections and exit

Docker compose stop_grace_period should be set higher than the longest
expected task runtime (default: 5 minutes).
"""
import os
import sys
import logging

from celery import Celery
from celery.signals import (
    task_prerun,
    task_postrun,
    task_failure,
    worker_shutdown,
    worker_ready,
    worker_init,
    task_revoked,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import get_settings

settings = get_settings()

# Create Celery app
celery_app = Celery(
    'ann_search_engine',
    broker=settings.celery_broker_url or settings.redis_url,
    backend=settings.celery_result_backend or settings.redis_url,
    include=[
        'tasks.index_tasks',
        'tasks.vector_tasks',
        'tasks.maintenance_tasks',
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task execution
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Task execution settings
    task_always_eager=settings.celery_task_always_eager,  # For testing
    task_store_eager_result=True,
    task_ignore_result=False,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3000,  # 50 minutes soft limit

    # Worker graceful shutdown settings
    worker_cancel_long_running_tasks_on_connection_loss=True,

    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_max_memory_per_child=500000,  # 500MB

    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_extended=True,

    # Broker settings
    broker_connection_retry_on_startup=True,
    broker_heartbeat=30,

    # Task routing — use separate queues for workload isolation
    task_routes={
        'tasks.index_tasks.*': {'queue': 'index'},
        'tasks.vector_tasks.*': {'queue': 'vector'},
        'tasks.maintenance_tasks.*': {'queue': 'maintenance'},
    },

    # Scheduled tasks
    beat_schedule={
        'health-check': {
            'task': 'tasks.maintenance_tasks.health_check',
            'schedule': 60.0,  # Every minute
        },
        'cleanup-old-logs': {
            'task': 'tasks.maintenance_tasks.cleanup_old_logs',
            'schedule': 3600.0,  # Every hour
        },
        'backup-indexes': {
            'task': 'tasks.maintenance_tasks.backup_indexes',
            'schedule': 86400.0,  # Every day
        },
        'update-statistics': {
            'task': 'tasks.maintenance_tasks.update_statistics',
            'schedule': 300.0,  # Every 5 minutes
        },
    },
)


# ====================== Task Lifecycle Signals ======================


@task_prerun.connect
def task_prerun_handler(task_id, task, args, kwargs, **extras):
    """Log task start"""
    logger.info(f"Task started: {task.name}[{task_id}]")


@task_postrun.connect
def task_postrun_handler(task_id, task, args, kwargs, retval, state, **extras):
    """Log task completion"""
    logger.info(f"Task completed: {task.name}[{task_id}] with state {state}")


@task_failure.connect
def task_failure_handler(task_id, exception, args, kwargs, traceback, einfo, **extras):
    """Log task failure"""
    logger.error(f"Task failed: {task_id} with exception {exception}")


@task_revoked.connect
def task_revoked_handler(
    request, terminated, signum, expired, **kwargs
):
    """
    Called when a task is revoked during shutdown.
    Logs the revoked task so operators can re-enqueue if needed.

    Args:
        terminated: True if killed by SIGTERM/SIGINT during warm shutdown.
        signum: Signal number that caused termination (e.g., 15 for SIGTERM).
        expired: True if task expired before execution.
    """
    reason = "unknown"
    if terminated:
        reason = f"terminated by signal {signum}"
    elif expired:
        reason = "expired"
    else:
        reason = "manually revoked"

    logger.warning(
        "Task revoked: %s — %s",
        getattr(request, "id", "unknown"),
        reason,
    )


# ====================== Worker Lifecycle Signals ======================


@worker_init.connect
def worker_init_handler(sender, **kwargs):
    """
    Called when the worker starts initialising.
    Sets up custom signal handlers for graceful OS-level shutdown.
    """
    hostname = getattr(sender, "hostname", "unknown")
    logger.info("Celery worker initialising: %s", hostname)


@worker_ready.connect
def worker_ready_handler(sender, **kwargs):
    """
    Called when the worker is fully initialised and ready to accept tasks.
    """
    hostname = getattr(sender, "hostname", "unknown")
    concurrency = getattr(sender, "concurrency", "?")
    queues = getattr(sender, "task_consumer", None)
    queue_names = []
    if queues and hasattr(queues, "queues"):
        queue_names = list(queues.queues.keys())
    logger.info(
        "Celery worker ready: %s | concurrency=%s | queues=%s",
        hostname,
        concurrency,
        queue_names or "(default)",
    )


@worker_shutdown.connect
def worker_shutdown_handler(sender, **kwargs):
    """
    Called when the worker begins shutting down.
    This is fired after SIGTERM is received and before the process exits.
    In-flight tasks receive `worker_cancel_long_running_tasks_on_connection_loss`
    signal and are given time to finish within the warm-shutdown period.

    Docker compose sends SIGTERM, then waits `stop_grace_period` before SIGKILL.
    Ensure stop_grace_period >= longest expected task runtime (default 5 minutes).
    """
    hostname = getattr(sender, "hostname", "unknown")
    logger.info(
        "Celery worker shutting down: %s | Waiting for in-flight tasks to complete...",
        hostname,
    )

    # Log remaining pool stats if available
    pool = getattr(sender, "pool", None)
    if pool:
        try:
            num_processes = getattr(pool, "num_processes", None)
            if num_processes:
                logger.debug("Worker pool size during shutdown: %s", num_processes)
        except Exception:
            pass


if __name__ == '__main__':
    celery_app.start()
