from kubernetes.utils import create_from_yaml
from kubernetes import client, config
import yaml
import json
import os

NUM_WORKERS = 2

configurations_dir = 'D:/Università/Tesi/Distributed Learning/Kubernetes/Configurations'
deployments_dir = 'D:/Università/Tesi/Distributed Learning/Kubernetes/Deployments'
services_dir = 'D:/Università/Tesi/Distributed Learning/Kubernetes/Services'
volumes_dir = 'D:/Università/Tesi/Distributed Learning/Kubernetes/Volumes'

def create_resources():
    dirs = [configurations_dir, deployments_dir, services_dir]

    api_client = client.ApiClient()

    for dir in dirs:
        filenames = os.listdir(dir)
        for filename in filenames:
            filepath = os.path.join(dir, filename)
            create_from_yaml(api_client, filepath)

def create_config(template):
    tf_config = json.loads(template["data"]["TF_CONFIG"])
    IPs = list(tf_config["cluster"]["worker"])
    new_port = 8082

    for i in range(2, NUM_WORKERS):
        hostname = f"worker{i}"
        IPs.append(f"{hostname}:{new_port}")
        new_port+=1
    template["metadata"]["name"] = "chief"
    tf_config["task"]["index"] = 0
    tf_config["cluster"]["worker"] = IPs
    template["data"]["TF_CONFIG"] = json.dumps(tf_config)
    with open(f"{configurations_dir}/chief.yaml", "w") as file:
        yaml.dump(template, file)

    for i in range(1, NUM_WORKERS):
        tf_config = json.loads(template["data"]["TF_CONFIG"])
        template["metadata"]["name"] = f"worker{i}"
        tf_config["task"]["index"] = i
        tf_config["cluster"]["worker"] = IPs
        template["data"]["TF_CONFIG"] = json.dumps(tf_config)
        with open(f"{configurations_dir}/{template['metadata']['name']}.yaml", "w") as file:
            yaml.dump(template, file)

def create_dpls(template):
    for i in range(2, NUM_WORKERS):
        worker = f"worker{i}"
        template["metadata"]["name"] = worker
        template["spec"]["selector"]["matchLabels"]["node"] = worker
        template["spec"]["template"]["metadata"]["labels"]["node"] = worker
        containers = template["spec"]["template"]["spec"]["containers"]
        containers[0]["name"] = worker
        containers[0]["envFrom"][0]["configMapRef"]["name"] = worker

        with open(f"{deployments_dir}/{worker}.yaml", "w") as file:
            yaml.dump(template, file)

def create_srv(template):
    new_port = 8082
    for i in range(2, NUM_WORKERS):

        worker = f"worker{i}"
        template["metadata"]["name"] = worker
        template["metadata"]["labels"]["node"] = worker
        template["spec"]["selector"]["node"] = worker
        template["spec"]["ports"][0]["port"] = new_port
        template["spec"]["ports"][0]["targetPort"] = new_port
        new_port += 1

        with open(f"{services_dir}/{worker}.yaml", "w") as file:
            yaml.dump(template, file)

def create_files():
    if NUM_WORKERS < 2:
        exit()
    elif NUM_WORKERS == 2:
        return

    with open(f"{configurations_dir}/worker1.yaml", "r") as file:
            template = yaml.safe_load(file)
            create_config(template)

    with open(f"{deployments_dir}/worker1.yaml", "r") as file:
            template = yaml.safe_load(file)
            create_dpls(template)

    with open(f"{services_dir}/worker1.yaml", "r") as file:
            template = yaml.safe_load(file)
            create_srv(template)


if __name__ == '__main__':
    config.load_kube_config()
    
    create_files()
    
    create_resources()
    print("Resources applied")