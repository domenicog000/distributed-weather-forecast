import docker

client = docker.from_env()

db = client.containers.get("mongodb")
chief = client.containers.get("chief")
worker = client.containers.get("worker")

db.stop()
chief.stop()
worker.stop()