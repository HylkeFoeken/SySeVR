#/bin/sh
#########################
# kills neo4j out of memory process
# starts neo4j instance
#########################

killall -9 /usr/lib/jvm/zulu-7-amd64/bin/java
/home/hylke/thesis/neo4j-community-2.1.5/bin/neo4j console &