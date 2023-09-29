from kubernetes import client, config
import subprocess
import threading

def get_pods_and_containers(namespace):
    config.load_kube_config()
    
    core_api = client.CoreV1Api()
    
    pods = core_api.list_namespaced_pod(namespace, label_selector = "node").items

    dpl = {}
    for pod in pods:
        for container in pod.spec.containers:
            if container.name[:6] == "chief"  or container.name[:6] == "worker":
                dpl[pod.metadata.name] = container.name

    return dpl


def start_training(kubectl_command, container):
    process = subprocess.Popen(kubectl_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    #output = output.decode('utf-8')

    #with open(f"D:/Università/Tesi/Distributed Learning/Training/{container}.txt", "w") as file:
    #    file.write(output)
        
    #print("Error:", error)

def execute_command_in_container(dpl):

    for pod, container in dpl.items():
        command = "python worker.py"
        kubectl_command = f"kubectl exec -it {pod} -c {container} -- {command}"
        thread = threading.Thread(target = start_training, args = (kubectl_command,container))
        thread.start()

if __name__ == '__main__':
    namespace = "default"
    dpl = get_pods_and_containers(namespace)
    execute_command_in_container(dpl)