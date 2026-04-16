docker stop neo4j

docker exec --interactive --tty neo4j \
  neo4j-admin database import full neo4j \
    --nodes=Company=/import_csv/companies.csv \
    --nodes=Investor=/import_csv/investors.csv \
    --nodes=Patent=/import_csv/patents.csv \
    --nodes=EventType=/import_csv/eventtypes.csv \
    --relationships=INVEST=/import_csv/invest_rels.csv \
    --relationships=HAVE=/import_csv/have_rels.csv \
    --overwrite-destination \
    --high-parallel-io=on \
    --verbose