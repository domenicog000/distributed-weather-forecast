import tarfile
import docker
import json
import os

def get_metrics():
    #source_path = "/Worker/metrics.json"
    #destination_path = "D:/Università/Tesi/Distributed Learning/Metrics"
    source_to_destination_path = {
        #"/Worker/metrics.json": "D:/Università/Tesi/Distributed Learning/Metrics",
        "/Worker/train": "D:/Università/Tesi/Distributed Learning/Tensorboard",
        "/Worker/validation": "D:/Università/Tesi/Distributed Learning/Tensorboard"
    }
    
    client = docker.DockerClient()

    containers = client.containers.list()
    for container in containers:
        for source, destination in source_to_destination_path.items():
            container = client.containers.get(container.name)
            try:
                stream, stat = container.get_archive(source)

            except:
                continue
            temp_tar_path = './temp.tar'
            with open(temp_tar_path, 'wb') as f:
                for chunk in stream:
                    f.write(chunk)
            
            with tarfile.open(temp_tar_path, 'r') as tar:
                tar.extractall(path = f"{destination}/96 to 48/4 Workers")
            os.remove(temp_tar_path)
        
        client.close()


if __name__ == '__main__':
    get_metrics()