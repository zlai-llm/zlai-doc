> 使用docker安装neo4j

```docker
docker run -d --name neo4j 
    -p 7474:7474 
    -p 7687:7687 
    -e NEO4J_AUTH=neo4j/12345678 
    -e dbms.connector.bolt.tls_level=OPTIONAL 
    neo4j:latest
```

> 访问neo4j

http://localhost:7474/browser/

> 进入neo4j

