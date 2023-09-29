import docker

client = docker.from_env()

#database = client.containers.run(
#    "mongodb:v1",
#    detach = True,
#    network = "distributed-network",
#    name = "mongo",
#    hostname='172.20.0.4',
#    ports = {"27017/tcp": 27017},
#    volumes = ['mongo:/data/db']
#    #volumes = ['mongo:/data/db', "D:/Università/Tesi/Distributed Learning/Database/mongod.conf:/etc/mongod.conf.orig"]
#)

chief = client.containers.run(
    "chief:v1",
    detach = True,
    network = "distributed-network",
    name = "chief",
    hostname = '172.20.0.2',
    ports = {"8080/tcp": 8080}
)

worker = client.containers.run(
    "worker:v1",
    detach = True,
    network = "distributed-network",
    name = "worker",
    hostname='172.20.0.3',
    ports= {"8081/tcp": 8081}
)


