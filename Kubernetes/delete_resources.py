from kubernetes import client, config
import json
import yaml
import os

configurations_dir = 'D:/Università/Tesi/Distributed Learning/Kubernetes/Configurations'
deployments_dir = 'D:/Università/Tesi/Distributed Learning/Kubernetes/Deployments'
services_dir = 'D:/Università/Tesi/Distributed Learning/Kubernetes/Services'

def delete_config(api, name):
    api.delete_namespaced_config_map(
        name = name,
        namespace = "default"
    )

def delete_service(api, name):
    api.delete_namespaced_service(
        name = name,
        namespace = "default"
    )

def delete_deployment(api, name):
    api.delete_namespaced_deployment(
                name = name,
                namespace = "default",
                body = client.V1DeleteOptions()
    )
    
def delete_resources(api_appsv1, api_corev1):
    for name in os.listdir(deployments_dir):
        delete_deployment(api_appsv1, name[:-5])
            
    for name in os.listdir(configurations_dir):
        delete_config(api_corev1, name[:-5])
        
    for name in os.listdir(services_dir):
        delete_service(api_corev1, name[:-5])

def reset_config():
    filenames = os.listdir(configurations_dir)
    filenames.remove("distributed-model.yaml")
    for filename in filenames:
        with open(f"{configurations_dir}/{filename}", "r") as file:
            template = yaml.safe_load(file)
        filepath = os.path.join(configurations_dir, filename)
        tf_config = json.loads(template["data"]["TF_CONFIG"])
        template["metadata"]["name"] = filename[:-5]
        IPs = ["chief:8080", "worker1:8081"]
        tf_config["cluster"]["worker"] = IPs
        template["data"]["TF_CONFIG"] = json.dumps(tf_config)
        
        with open(filepath, "w") as file:
            yaml.dump(template, file)

def remove_files():
    dirs = [configurations_dir, deployments_dir, services_dir]
    files = ["db.yaml", "chief.yaml", "worker1.yaml", "distributed-model.yaml", "serving.yaml"]

    for dir in dirs:
        for filename in os.listdir(dir):
            if filename not in files:
                filepath = os.path.join(dir, filename)
                os.remove(filepath)

if __name__ == '__main__':

    config.load_kube_config()
    api_corev1 = client.CoreV1Api()
    api_client = client.ApiClient()
    api_appsv1 = client.AppsV1Api(api_client)
    
    delete_resources(api_appsv1, api_corev1)
    print("Resources deleted")
    
    remove_files()
    reset_config()
